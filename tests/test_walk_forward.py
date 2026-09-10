"""
The purge gap is what keeps walk-forward evaluation honest: a training
row's label depends on `horizon` days of future price data, so any
training row within `horizon` days of the test window must be dropped or
its label overlaps the test period.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "validation"))
from walk_forward import walk_forward_splits  # noqa: E402


def test_folds_are_chronological_and_non_overlapping():
    folds = walk_forward_splits(n_samples=1000, n_splits=4, min_train_size=200, purge=5)
    for fold in folds:
        assert fold.train_idx.max() < fold.test_idx.min()
        assert not np.isin(fold.train_idx, fold.test_idx).any()


def test_purge_gap_is_respected():
    horizon = 5
    folds = walk_forward_splits(n_samples=1000, n_splits=4, min_train_size=200, purge=horizon)
    for fold in folds:
        gap = fold.test_idx.min() - fold.train_idx.max()
        assert gap > horizon, f"train/test gap {gap} does not exceed horizon {horizon}"


def test_test_windows_cover_everything_after_min_train_size():
    folds = walk_forward_splits(n_samples=1000, n_splits=4, min_train_size=200, purge=5)
    all_test = np.concatenate([f.test_idx for f in folds])
    assert all_test.min() == 200
    assert all_test.max() == 999
    assert len(np.unique(all_test)) == len(all_test)  # no fold tests the same row twice
