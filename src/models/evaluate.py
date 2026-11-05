"""Classification metrics, with the buying (minority) class in focus."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Scorer names for cross-validation, buying-class metrics first.
CV_SCORING = {
    "pr_auc": "average_precision",
    "recall": "recall",
    "precision": "precision",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "accuracy": "accuracy",
}


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    """Return the metric suite for predicted buyer probabilities.

    PR-AUC and ROC-AUC use the raw probabilities, so they do not depend on
    the threshold. Precision, recall, F1 and accuracy do.
    """
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
    }
