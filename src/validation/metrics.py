"""
Classification metrics for the SPY-direction target. Kept in one place so
every model in the ladder is scored identically -- a fair comparison is
the whole point of the ladder (roadmap section 26: "Never compare
Transformer vs. Nothing").
"""

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def summarize(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """y_prob = predicted P(SPY up). Returns accuracy at a 0.5 threshold,
    AUC (threshold-independent), and log loss (penalizes overconfidence)."""
    y_pred = (y_prob >= 0.5).astype(int)
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "n": len(y_true),
    }
    # AUC/log-loss need both classes present in the fold; small folds can
    # fail this, so degrade gracefully instead of crashing a whole run.
    if len(np.unique(y_true)) > 1:
        result["auc"] = roc_auc_score(y_true, y_prob)
        result["log_loss"] = log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6))
    else:
        result["auc"] = float("nan")
        result["log_loss"] = float("nan")
    return result
