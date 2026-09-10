"""
Rung 1 (naive) and rung 3 (random forest) of the model ladder. Naive
should always be run and reported -- it's the floor everything else has
to beat to mean anything (roadmap section 26).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class NaiveMajorityModel:
    """Predicts the training set's most common class, every time. Zero
    information about the input -- if a "smarter" model can't beat this,
    it has learned nothing useful."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveMajorityModel":
        self.majority_prob_ = float(y.mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.majority_prob_)


class PersistenceModel:
    """Predicts "up" whenever the most recent realized 5-day return was
    positive, "down" otherwise -- the classic naive time-series baseline
    (tomorrow looks like today), using no fitted parameters at all."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PersistenceModel":
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return (X["spy_return_5d"] > 0).astype(float).to_numpy()


class RandomForestModel:
    def __init__(self, **kwargs):
        defaults = dict(n_estimators=300, max_depth=5, min_samples_leaf=20, random_state=42)
        defaults.update(kwargs)
        self.model = RandomForestClassifier(**defaults)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
