"""
Rung 2 of the model ladder. From the roadmap's classical-ML section
(10.2): P(Y=1 | x), a linear decision boundary in standardized feature
space. Standardizing (section 6: z = (x - mean) / std) matters here more
than for tree models, since logistic regression's coefficients are scaled
by each feature's range.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class LogisticModel:
    def __init__(self, **kwargs):
        defaults = dict(max_iter=1000, C=1.0)
        defaults.update(kwargs)
        self.model = make_pipeline(StandardScaler(), LogisticRegression(**defaults))

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticModel":
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]
