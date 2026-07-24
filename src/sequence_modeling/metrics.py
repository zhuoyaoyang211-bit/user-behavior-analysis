"""Evaluation metrics for binary purchase prediction."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


THRESHOLD = 0.5


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = THRESHOLD,
) -> dict[str, float | int]:
    """Compute binary classification metrics.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted positive probabilities.
        threshold: Classification threshold.

    Returns:
        Dictionary containing AUC, PR-AUC, logloss, threshold metrics,
        and confusion matrix counts.
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics: dict[str, float | int] = {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc_ap": average_precision_score(y_true, y_prob),
        "log_loss": log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15)),
        "accuracy_at_0_5": accuracy_score(y_true, y_pred),
        "precision_at_0_5": precision_score(y_true, y_pred, zero_division=0),
        "recall_at_0_5": recall_score(y_true, y_pred, zero_division=0),
        "f1_at_0_5": f1_score(y_true, y_pred, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return metrics


def compute_threshold_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list[float],
) -> list[dict[str, float | int]]:
    """Compute threshold metrics for multiple cutoffs.

    Args:
        y_true: Ground-truth binary labels.
        y_prob: Predicted positive probabilities.
        thresholds: Probability cutoffs to evaluate.

    Returns:
        One metric dictionary per threshold.
    """
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
        rows.append(
            {
                "threshold": threshold,
                "roc_auc": metrics["roc_auc"],
                "pr_auc_ap": metrics["pr_auc_ap"],
                "log_loss": metrics["log_loss"],
                "accuracy": metrics["accuracy_at_0_5"],
                "precision": metrics["precision_at_0_5"],
                "recall": metrics["recall_at_0_5"],
                "f1": metrics["f1_at_0_5"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
            }
        )
    return rows
