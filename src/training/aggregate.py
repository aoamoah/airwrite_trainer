"""Fold aggregation.

With participant-level cross-validation a configuration is no longer one
number, it is a distribution over held-out participants. The spread is part
of the result: a model that scores 0.90 on one participant and 0.55 on the
next has not been shown to generalise, however good the mean looks.
"""

from dataclasses import dataclass, field

import numpy as np

METRIC_KEYS = ("accuracy", "precision_writing", "recall_writing",
               "f1_writing", "f2_writing", "f1_macro", "roc_auc",
               "average_precision", "positive_rate",
               "predicted_positive_rate", "threshold")

# Event-level scalars worth a mean across folds. The precision/recall/F1/F2
# blocks under each matching criterion are averaged separately, in
# `_aggregate_events`.
EVENT_SCALARS = ("n_true_events", "n_pred_events", "median_iou_of_matches",
                 "median_onset_error_frames", "median_onset_delay_frames",
                 "mean_fragments_per_true_event",
                 "true_events_missed_entirely", "median_true_event_frames",
                 # pause events and segmental metrics (src/training/segmental.py)
                 "n_true_pauses", "n_pred_pauses", "median_true_pause_frames",
                 "edit_score", "seg_f1@10", "seg_f1@25", "seg_f1@50")
# Blocks holding precision/recall/F1/F2 under one matching criterion
CRITERION_PREFIXES = ("iou@", "onset@", "pause@")


@dataclass
class Aggregate:
    dataset: str
    model: str
    window: dict | None
    n_folds: int
    metrics: dict = field(default_factory=dict)     # key -> mean/std/min/max/values
    per_fold: list = field(default_factory=list)
    pooled_confusion: list = field(default_factory=lambda: [[0, 0], [0, 0]])
    train_seconds_total: float = 0.0
    epochs_trained: list = field(default_factory=list)
    by_source: dict = field(default_factory=dict)   # combined runs only
    by_detection: dict = field(default_factory=dict)  # hand seen vs not
    events: dict = field(default_factory=dict)      # episode-level metrics
    smoothed: dict = field(default_factory=dict)    # frame metrics after smoothing
    smoothed_events: dict = field(default_factory=dict)

    @property
    def key(self):
        return (self.dataset, self.model, _window_key(self.window))

    def mean(self, metric: str):
        return self.metrics.get(metric, {}).get("mean")

    def std(self, metric: str):
        return self.metrics.get(metric, {}).get("std")


def _window_key(window: dict | None):
    return (window["size"], window["stride"]) if window else None


def _mean_std(values: list) -> dict:
    values = [float(v) for v in values if v is not None]
    if not values:
        return {}
    return {"mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "n": len(values)}


def _aggregate_events(runs: list) -> dict:
    """Average the episode-level metrics over folds.

    Kept separate from the frame metrics because the matching criteria are
    nested one level deeper, and because a fold with no writing episodes at
    all contributes nothing rather than a zero — averaging in a zero for
    "there was nothing to detect" would understate every model equally but
    misleadingly.
    """
    blocks = [r.event_metrics for r in runs if getattr(r, "event_metrics", None)]
    if not blocks:
        return {}
    out: dict = {}
    criteria = sorted({k for b in blocks for k in b
                       if k.startswith(CRITERION_PREFIXES)})
    for crit in criteria:
        out[crit] = {
            metric: stats
            for metric in ("precision", "recall", "f1", "f2")
            if (stats := _mean_std([b.get(crit, {}).get(metric) for b in blocks]))
        }
    for key in EVENT_SCALARS:
        if stats := _mean_std([b.get(key) for b in blocks]):
            out[key] = stats
    out["folds_with_events"] = len(blocks)
    return out


def aggregate_results(results: list) -> list[Aggregate]:
    """Group RunResults by (dataset, model, window) and summarise across folds."""
    order, buckets = [], {}
    for r in results:
        key = (r.dataset, r.model, _window_key(r.window))
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(r)

    aggregates = []
    for key in order:
        runs = buckets[key]
        first = runs[0]
        agg = Aggregate(
            dataset=first.dataset, model=first.model, window=first.window,
            n_folds=len(runs),
            train_seconds_total=float(sum(r.train_seconds for r in runs)),
            epochs_trained=[r.epochs_trained for r in runs
                            if r.epochs_trained is not None],
        )
        for metric in METRIC_KEYS:
            values = [r.test_metrics.get(metric) for r in runs]
            values = [v for v in values if v is not None]
            if not values:
                continue
            agg.metrics[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "values": [float(v) for v in values],
            }
        pooled = np.zeros((2, 2), dtype=np.int64)
        for r in runs:
            cm = r.test_metrics.get("confusion_matrix")
            if cm:
                pooled += np.array(cm, dtype=np.int64)
        agg.pooled_confusion = pooled.tolist()

        # Combined runs: how the pooled model did on each corpus separately
        per_source: dict[str, list] = {}
        for r in runs:
            for src, m in (r.extra.get("test_metrics_by_source") or {}).items():
                if m.get("f1_writing") is not None:
                    per_source.setdefault(src, []).append(m)
        agg.by_source = {
            src: {
                "folds": len(ms),
                **{metric: {"mean": float(np.mean(vals)),
                            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
                   for metric in ("accuracy", "precision_writing",
                                  "recall_writing", "f1_writing")
                   if (vals := [m[metric] for m in ms if m.get(metric) is not None])},
                "n_test": int(sum(m.get("n_samples", 0) for m in ms)),
            }
            for src, ms in sorted(per_source.items())
        }
        agg.events = _aggregate_events(runs)

        # The same summaries again for the temporally post-processed
        # predictions. The gap between the two is the share of the problem
        # that is temporal consistency rather than discrimination.
        smoothed = [r.smoothed_metrics for r in runs
                    if getattr(r, "smoothed_metrics", None)]
        if smoothed:
            agg.smoothed = {
                metric: stats for metric in METRIC_KEYS
                if (stats := _mean_std([m.get(metric) for m in smoothed]))
            }
        agg.smoothed_events = _aggregate_events(
            [type("R", (), {"event_metrics": r.smoothed_events})()
             for r in runs if getattr(r, "smoothed_events", None)])

        # Frames where MediaPipe saw a hand, versus frames where it did not.
        # "No hand" is very nearly a free correct answer on this data, so a
        # score that collapses once those frames are removed is a score that
        # will collapse in the app.
        per_detection: dict[str, list] = {}
        for r in runs:
            for state, m in (r.extra.get("test_metrics_by_detection") or {}).items():
                per_detection.setdefault(state, []).append(m)
        agg.by_detection = {
            state: {
                "folds": len(ms),
                "n_test": int(sum(m.get("n_samples", 0) for m in ms)),
                **{metric: stats for metric in
                   ("positive_rate", "accuracy", "precision_writing",
                    "recall_writing", "f1_writing")
                   if (stats := _mean_std([m.get(metric) for m in ms]))},
            }
            for state, ms in sorted(per_detection.items())
        }

        agg.per_fold = [
            {
                "fold": r.fold,
                "test_participants": r.extra.get("test_participants", []),
                "n_test": r.n_test,
                "test_writing_prior": r.extra.get("test_writing_prior"),
                **{m: r.test_metrics.get(m) for m in
                   ("accuracy", "precision_writing", "recall_writing",
                    "f1_writing", "f2_writing", "roc_auc",
                    "average_precision", "threshold")},
                "train_seconds": r.train_seconds,
                "epochs_trained": r.epochs_trained,
            }
            for r in runs
        ]
        aggregates.append(agg)
    return aggregates
