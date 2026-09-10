"""
Walk-forward (expanding-window) train/test splits, with a purge gap so no
training row's label peeks into the test period.

Roadmap ref section 27: "Walk-Forward Backtest". Why not a random/shuffled
train-test split: this is time-series data, and a random split would let
the model train on data from *after* the day it's being tested on --
exactly the look-ahead bias Phase 2 spent so much effort avoiding, just
reintroduced at the model-evaluation stage instead of the feature stage.

Why the purge gap: a training row dated t's label depends on the SPY
return from t to t+horizon. If t is within `horizon` days of the test
window's start, that label was computed using price data that overlaps
the test period -- so it's dropped from the training fold entirely
(`purge`, from the finance-ML literature: Lopez de Prado 2018).
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray


def walk_forward_splits(
    n_samples: int, n_splits: int = 5, min_train_size: int = 500, purge: int = 5
) -> list[Fold]:
    """Expanding window: fold k trains on [0, test_start_k), purges the
    last `purge` rows before the boundary, and tests on the next
    contiguous test block. Test blocks are equal-sized and cover
    everything after min_train_size."""
    test_region = n_samples - min_train_size
    if test_region < n_splits:
        raise ValueError(
            f"Not enough data for {n_splits} folds: {n_samples} rows, "
            f"min_train_size={min_train_size}"
        )
    test_size = test_region // n_splits

    folds = []
    for k in range(n_splits):
        test_start = min_train_size + k * test_size
        test_end = n_samples if k == n_splits - 1 else test_start + test_size
        train_end = max(0, test_start - purge)
        folds.append(Fold(
            train_idx=np.arange(0, train_end),
            test_idx=np.arange(test_start, test_end),
        ))
    return folds
