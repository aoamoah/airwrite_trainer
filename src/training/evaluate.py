"""Metrics computed identically for every model so comparisons are fair.

Two things here are deliberate.

**The operating point is fitted, for every model.** The rule baselines have
always fitted their threshold; the learned models used to be scored at a flat
0.5 while the writing prior across participants ranges from 10% to 84%. That
handicapped the learned models in exactly the comparison the study is about.
`fit_threshold` is now applied to all of them, always on validation data —
never on the test fold.

**Both operating-point-free and operating-point-dependent metrics are kept.**
ROC AUC flatters a detector on an imbalanced problem; average precision (the
area under the precision-recall curve) does not, and its baseline is the
class prior rather than 0.5. F2 is reported alongside F1 because missing a
writing episode costs the user more than a brief false positive, and because
it is what the closest published work reports.
"""

import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report,
)

DEFAULT_THRESHOLD = 0.5


def fit_threshold(y_true: np.ndarray, y_prob: np.ndarray,
                  beta: float = 1.0, steps: int = 199) -> tuple[float, float]:
    """Threshold maximising F-beta of the writing class. Returns (thr, score).

    Candidates are quantiles of the observed scores, which concentrates the
    search where the data actually is. Falls back to 0.5 when the split is
    single-class or empty — there is nothing to fit against.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return DEFAULT_THRESHOLD, float("nan")
    qs = np.linspace(0.01, 0.99, steps)
    candidates = np.unique(np.quantile(y_prob, qs))
    scores = [fbeta_score(y_true, (y_prob >= t).astype(int), beta=beta,
                          zero_division=0) for t in candidates]
    best = int(np.argmax(scores))
    return float(candidates[best]), float(scores[best])


def evaluate_binary(y_true: np.ndarray, y_prob: np.ndarray,
                    threshold: float = DEFAULT_THRESHOLD) -> dict:
    """y_true: 1 = writing. y_prob: probability of writing."""
    if len(y_true) == 0:
        return {"n_samples": 0, "threshold": threshold, "accuracy": None,
                "precision_writing": None, "recall_writing": None,
                "f1_writing": None, "f2_writing": None, "f1_macro": None,
                "roc_auc": None, "average_precision": None,
                "positive_rate": None, "predicted_positive_rate": None,
                "confusion_matrix": [[0, 0], [0, 0]],
                "classification_report": "(empty split)"}
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "n_samples": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_writing": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_writing": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_writing": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2_writing": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        # The prior the metrics sit against, and what the model actually fired
        # on — a detector that predicts 80% writing on a 30% prior is broken in
        # a way F1 alone can hide
        "positive_rate": float(np.mean(y_true)),
        "predicted_positive_rate": float(np.mean(y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        # labels=[0, 1] is load-bearing: subsets such as the "no hand
        # detected" breakdown are frequently single-class, and without it
        # sklearn raises instead of reporting a class with zero support
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1],
            target_names=["not_writing", "writing"], zero_division=0
        ),
    }
    # Both are undefined on a single-class split. Depending on the sklearn
    # version that surfaces as a ValueError or as a warning plus NaN, and a
    # NaN would poison every fold average it reaches — so normalise both
    # outcomes to None, which the aggregation already skips.
    for key, fn in (("roc_auc", roc_auc_score),
                    ("average_precision", average_precision_score)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                value = float(fn(y_true, y_prob))
        except ValueError:
            value = float("nan")
        metrics[key] = value if np.isfinite(value) else None
    return metrics
