"""Classification metrics computed with Scikit-Learn."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

DECISION_THRESHOLD = 0.5


def classification_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float = DECISION_THRESHOLD
) -> dict[str, float]:
    """Score relevance predictions as a binary classification problem.

    Args:
        labels: Ground-truth relevance (``0``/``1``).
        scores: Predicted probabilities.
        threshold: Probability above which an interaction counts as relevant.

    Returns:
        Mapping with accuracy, precision, recall, F1, ROC AUC and log loss.
    """
    predictions = (scores >= threshold).astype(int)
    truth = labels.astype(int)
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "precision": float(precision_score(truth, predictions, zero_division=0)),
        "recall": float(recall_score(truth, predictions, zero_division=0)),
        "f1": float(f1_score(truth, predictions, zero_division=0)),
        "roc_auc": _roc_auc(truth, scores),
        "log_loss": _log_loss(truth, scores),
    }


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC AUC, degrading gracefully on single-class inputs.

    Args:
        labels: Ground-truth relevance.
        scores: Predicted probabilities.

    Returns:
        The ROC AUC, or ``0.5`` when only one class is present.
    """
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def _log_loss(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute the log loss with clipped probabilities.

    Args:
        labels: Ground-truth relevance.
        scores: Predicted probabilities.

    Returns:
        The log loss.
    """
    clipped = np.clip(scores, 1e-7, 1 - 1e-7)
    return float(log_loss(labels, clipped, labels=[0, 1]))
