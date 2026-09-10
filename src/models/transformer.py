"""
Rung 5 of the model ladder: the small Transformer from roadmap section 26
-- previous 60 trading days, d_model=64, 4 attention heads, 2 encoder
layers. Same fit_predict() interface and same reasoning as lstm.py.

This is a "getting the mechanics right" milestone (self-attention over a
real leakage-safe feature sequence), not a claim that a Transformer suits
~2,500 rows of daily data -- roadmap section 26 explicitly expects it may
not beat the simpler baselines, and says that's a legitimate finding, not
a failure.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from windowing import make_windows  # noqa: E402


class _PositionalEncoding(nn.Module):
    """Standard sinusoidal positions (Vaswani et al. 2017) -- attention
    itself has no notion of order, so the model needs this to tell "5 days
    ago" apart from "50 days ago"."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


class _TransformerNet(nn.Module):
    def __init__(self, n_features: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoding = _PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        last_step = x[:, -1, :]  # representation after attending over the whole window
        return self.head(last_step).squeeze(-1)  # logit


class TransformerModel:
    def __init__(self, window: int = 60, d_model: int = 64, n_heads: int = 4, n_layers: int = 2,
                 epochs: int = 15, lr: float = 1e-3, batch_size: int = 64, seed: int = 42):
        self.window = window
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed

    def fit_predict(
        self, X_full: pd.DataFrame, y_full: pd.Series,
        train_idx: np.ndarray, test_idx: np.ndarray,
    ) -> np.ndarray:
        torch.manual_seed(self.seed)

        mean = X_full.iloc[train_idx].mean()
        std = X_full.iloc[train_idx].std().replace(0, 1.0)
        X_scaled = ((X_full - mean) / std).to_numpy()
        y_arr = y_full.to_numpy()

        X_seq, y_seq, t_index = make_windows(X_scaled, y_arr, self.window)
        train_mask = np.isin(t_index, train_idx)
        test_mask = np.isin(t_index, test_idx)

        model = _TransformerNet(n_features=X_scaled.shape[1], d_model=self.d_model,
                                 n_heads=self.n_heads, n_layers=self.n_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        X_train_t = torch.from_numpy(X_seq[train_mask])
        y_train_t = torch.from_numpy(y_seq[train_mask])
        n_train = len(X_train_t)

        model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n_train)
            for start in range(0, n_train, self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                optimizer.zero_grad()
                logits = model(X_train_t[batch_idx])
                loss = loss_fn(logits, y_train_t[batch_idx])
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            X_test_t = torch.from_numpy(X_seq[test_mask])
            probs = torch.sigmoid(model(X_test_t)).numpy()
        return probs
