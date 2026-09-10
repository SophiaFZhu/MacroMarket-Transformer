"""
Rung 4 of the model ladder: an LSTM over the previous `window` trading
days of features, predicting direction at the label's horizon.

Interface note: unlike baseline.py/logistic.py, this exposes fit_predict()
instead of separate fit()/predict_proba(). A sequence model's test-time
predictions need lookback context (the window of days *before* each test
row) that legitimately reaches back before the fold's train/test boundary
-- that's still only past information relative to each prediction day, not
leakage, but it doesn't fit the same one-row-in-one-row-out shape as the
flat models. Keeping that reshaping inside one method here (rather than
forcing a fake-uniform API) is more honest about what's actually going on.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from windowing import make_windows  # noqa: E402


class _LSTMNet(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]  # representation after seeing the whole window
        return self.head(last_step).squeeze(-1)  # logit


class LSTMModel:
    def __init__(self, window: int = 60, hidden_size: int = 32, num_layers: int = 1,
                 epochs: int = 15, lr: float = 1e-3, batch_size: int = 64, seed: int = 42):
        self.window = window
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed

    def fit_predict(
        self, X_full: pd.DataFrame, y_full: pd.Series,
        train_idx: np.ndarray, test_idx: np.ndarray,
    ) -> np.ndarray:
        torch.manual_seed(self.seed)

        # Standardize using train-fold statistics only -- fitting the
        # scaler on the full series (including test) would leak the test
        # period's distribution into training, same class of bug as
        # feature leakage, just at the preprocessing step instead.
        mean = X_full.iloc[train_idx].mean()
        std = X_full.iloc[train_idx].std().replace(0, 1.0)
        X_scaled = ((X_full - mean) / std).to_numpy()
        y_arr = y_full.to_numpy()

        X_seq, y_seq, t_index = make_windows(X_scaled, y_arr, self.window)
        train_mask = np.isin(t_index, train_idx)
        test_mask = np.isin(t_index, test_idx)

        model = _LSTMNet(n_features=X_scaled.shape[1], hidden_size=self.hidden_size,
                          num_layers=self.num_layers)
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
