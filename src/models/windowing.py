"""
Shared by lstm.py and transformer.py: turn the flat daily feature table
into overlapping (window, n_features) sequences, one per prediction day.

Not in the PDF's file list -- added because both sequence models need the
exact same windowing logic and it shouldn't be duplicated.
"""

import numpy as np


def make_windows(X: np.ndarray, y: np.ndarray, window: int):
    """X: (n_days, n_features), y: (n_days,).
    Returns X_seq: (n_windows, window, n_features), y_seq: (n_windows,),
    t_index: (n_windows,) -- the original row position each window/label
    corresponds to (its last day). Row t's window is X[t-window+1 : t+1],
    i.e. only t and earlier -- never a future row."""
    n = len(X)
    n_windows = n - window + 1
    n_features = X.shape[1]
    X_seq = np.empty((n_windows, window, n_features), dtype=np.float32)
    for i in range(n_windows):
        X_seq[i] = X[i:i + window]
    t_index = np.arange(window - 1, n)
    y_seq = y[t_index].astype(np.float32)
    return X_seq, y_seq, t_index
