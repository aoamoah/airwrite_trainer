"""Metrics computed identically for every model so comparisons are fair."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """y_true: 1 = writing. y_prob: probability of writing."""
    if len(y_true) == 0:
        return {"n_samples": 0, "threshold": threshold, "accuracy": None,
                "precision_writing": None, "recall_writing": None,
                "f1_writing": None, "f1_macro": None, "roc_auc": None,
                "confusion_matrix": [[0, 0], [0, 0]],
                "classification_report": "(empty split)"}
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "n_samples": int(len(y_true)),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_writing": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_writing": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_writing": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["not_writing", "writing"], zero_division=0
        ),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:  # single-class split
        metrics["roc_auc"] = None
    return metrics
