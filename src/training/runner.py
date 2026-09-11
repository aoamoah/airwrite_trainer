"""Orchestrates every model x dataset x window x fold combination and
collects everything the report needs.

The unit of evaluation is a fold, not a single split: every configuration is
trained once per held-out participant group (see `src.data.splits.make_folds`).
Window tensors are built once per window size and indexed per fold, so the
extra folds cost training time only.

Every model's decision threshold is fitted on validation data and applied to
test. Previously only the rule baselines had a fitted operating point while
the learned models were scored at a flat 0.5 — on a task whose writing prior
runs from 10% to 84% across participants, that was a handicap applied to one
side of the very comparison the study exists to make.
"""

import gc
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.loaders import LoadedDataset
from src.data.preprocess import preprocess, make_windows
from src.data.splits import Fold, make_folds
from src.models.builders import build_random_forest, build_recurrent
from src.models.heuristics import BASELINES
from src.training.evaluate import evaluate_binary, fit_threshold, DEFAULT_THRESHOLD
from src.training.events import event_metrics
from src.training.segmental import (
    DEFAULT_MIN_PAUSE_FRAMES, DEFAULT_PAUSE_TOLERANCES, DEFAULT_SEGMENT_OVERLAPS,
    pause_event_metrics, segmental_metrics, split_unscored,
)
from src.training.smoothing import smooth

SPLITS = ("train", "val", "test")
LEARNED_MODELS = ("random_forest", "lstm", "gru")


@dataclass
class RunResult:
    dataset: str
    model: str
    window: dict | None            # None for frame-level models
    train_seconds: float
    n_train: int
    n_val: int
    n_test: int
    test_metrics: dict
    fold: str | None = None
    val_metrics: dict | None = None
    train_metrics: dict | None = None
    history: dict | None = None    # keras per-epoch history
    epochs_trained: int | None = None
    feature_importances: list | None = None
    model_path: str | None = None
    event_metrics: dict | None = None
    smoothed_metrics: dict | None = None
    smoothed_events: dict | None = None
    extra: dict = field(default_factory=dict)


def _order_frames(frames: pd.DataFrame) -> pd.DataFrame:
    """Sort by session then frame, and reset the index.

    Event-level scoring reads contiguous runs straight off row order, so the
    row order has to be the timeline. Everything else here is order-agnostic,
    which is exactly why an out-of-order frame would have gone unnoticed.
    """
    by = ["group"] + (["frame_index"] if "frame_index" in frames.columns else [])
    return frames.sort_values(by, kind="stable").reset_index(drop=True)


def _fit_operating_point(cfg: dict, y_val, p_val, y_train, p_train) -> tuple[float, dict]:
    """Threshold for a learned model: fitted on val, never on test.

    Falls back to the training split only when a fold has no validation
    participants, and to 0.5 when neither split has both classes. Which of
    those happened is recorded, because a fold scored at 0.5 is not
    comparable with one scored at a fitted point.
    """
    beta = float(cfg.get("evaluation", {}).get("threshold_beta", 1.0))
    if not cfg.get("evaluation", {}).get("fit_threshold", True):
        return DEFAULT_THRESHOLD, {"threshold_fitted_on": "not fitted (disabled)"}
    if y_val is not None and len(y_val) and len(np.unique(y_val)) > 1:
        thr, score = fit_threshold(y_val, p_val, beta=beta)
        return thr, {"threshold_fitted_on": f"val split, maximising F{beta:g}(writing)",
                     "threshold_fit_score": round(score, 4)}
    if len(y_train) and len(np.unique(y_train)) > 1:
        thr, score = fit_threshold(y_train, p_train, beta=beta)
        return thr, {"threshold_fitted_on": "train split (fold has no usable val)",
                     "threshold_fit_score": round(score, 4)}
    return DEFAULT_THRESHOLD, {"threshold_fitted_on": "0.5 (no two-class split to fit on)"}


def _event_cfg(cfg: dict) -> dict:
    ev = dict(cfg.get("evaluation", {}).get("events", {}) or {})
    return {
        "iou_threshold": float(ev.get("iou_threshold", 0.5)),
        "tolerance_frames": int(ev.get("tolerance_frames", 12)),
        "min_event_frames": int(ev.get("min_event_frames", 3)),
    }


def _timeline_metrics(cfg, y_true, y_pred, groups, scored=None) -> dict | None:
    """Everything scored on the timeline rather than per frame: writing
    episodes, pause events (the pen-in-air protocol) and segmental Edit /
    F1@k (the action-segmentation standard). Unscored frames are cut out
    first, so nothing is matched across an unsure stretch."""
    if not cfg.get("evaluation", {}).get("event_metrics", True) or not len(y_true):
        return None
    y_true, y_pred, groups = split_unscored(y_true, y_pred, groups, scored)
    if not len(y_true):
        return None
    ev = dict(cfg.get("evaluation", {}).get("events", {}) or {})
    out = event_metrics(y_true, y_pred, groups, **_event_cfg(cfg))
    out.update(pause_event_metrics(
        y_true, y_pred, groups,
        tolerances=tuple(ev.get("pause_tolerances_frames", DEFAULT_PAUSE_TOLERANCES)),
        min_frames=int(ev.get("min_pause_frames", DEFAULT_MIN_PAUSE_FRAMES))))
    out.update(segmental_metrics(
        y_true, y_pred, groups,
        overlaps=tuple(ev.get("segment_overlaps", DEFAULT_SEGMENT_OVERLAPS))))
    return out


def _frame_events(cfg, y_true, y_prob, groups, threshold, scored=None) -> dict | None:
    """Timeline metrics for a frame-level model."""
    if not len(y_true):
        return None
    return _timeline_metrics(cfg, np.asarray(y_true),
                             (np.asarray(y_prob) >= threshold).astype(int),
                             np.asarray(groups, dtype=object), scored)


def _smoothing_cfg(cfg: dict) -> dict | None:
    sm = cfg.get("evaluation", {}).get("smoothing")
    if sm is None:
        sm = {}
    if sm.get("enabled", True) is False:
        return None
    return {k: sm[k] for k in ("band", "min_on", "min_off", "low", "high")
            if k in sm}


def _smoothed(cfg, y_true, y_prob, groups, threshold, scored=None):
    """Frame and timeline metrics after temporal post-processing.

    Reported alongside the raw numbers rather than replacing them: the raw
    scores say what the model learned, the smoothed ones say what a user
    would actually experience, and the gap between them is how much of the
    problem is temporal consistency rather than discrimination.

    Smoothing runs over the whole timeline, unsure frames included — the app
    has no labels and smooths everything — and scoring then skips them.
    """
    params = _smoothing_cfg(cfg)
    if params is None or not len(y_true):
        return None, None
    y_true = np.asarray(y_true)
    groups = np.asarray(groups, dtype=object)
    pred = smooth(np.asarray(y_prob), groups, threshold, params)
    mask = (np.ones(len(y_true), bool) if scored is None
            else np.asarray(scored, dtype=bool))
    metrics = evaluate_binary(y_true[mask], pred[mask].astype(float), threshold=0.5)
    metrics["smoothing"] = params
    return metrics, _timeline_metrics(cfg, y_true, pred, groups, scored)


def _to_percentiles(reference: np.ndarray, scores: np.ndarray,
                    threshold: float) -> tuple[np.ndarray, float]:
    """Map raw scores and their threshold onto the reference's percentiles.

    Smoothing works in probability units — a hysteresis band of 0.2 around
    the threshold, clamped to [0.01, 0.99]. The rule baselines score on raw
    signals instead (fingertip speed in hand-widths per second, routinely
    above 1), where that band and clamp are meaningless: the old code turned
    the speed rule's hysteresis into nonsense and its smoothed numbers with
    it. The empirical CDF of the training signal is monotone, so decisions at
    the threshold are unchanged, and it puts the band in comparable units
    (±10 percentile points) for every model."""
    ref = np.sort(np.asarray(reference, dtype=np.float64))
    if not len(ref):
        return np.asarray(scores, dtype=np.float64), threshold
    cdf = lambda x: np.searchsorted(ref, x, side="left") / len(ref)
    return cdf(np.asarray(scores, dtype=np.float64)), float(cdf(threshold))


class _Progress:
    """Progress reporting for runs that take tens of minutes.

    A full LOPO sweep is 13 folds x 3 window sizes x 2 architectures on top
    of the frame-level models, and until it finished it printed nothing at
    all — indistinguishable from a hang. On a terminal this redraws one line;
    redirected to a log it prints one line per completed unit of work, so
    `tail -f` stays useful without the hundreds of per-fold lines the summary
    deliberately avoids.
    """

    def __init__(self, total: int, label: str):
        self.total = max(total, 1)
        self.label = label
        self.done = 0
        self.t0 = time.time()
        self.tty = sys.stdout.isatty()

    def step(self, what: str):
        self.done += 1
        elapsed = time.time() - self.t0
        rate = elapsed / self.done
        remaining = rate * (self.total - self.done)
        msg = (f"[{100 * self.done / self.total:3.0f}%] {self.label} "
               f"{self.done}/{self.total} · {what} · "
               f"{self._fmt(elapsed)} elapsed, ~{self._fmt(remaining)} left")
        if self.tty:
            print("\r" + msg.ljust(96)[:96], end="", flush=True)
        else:
            print(msg, flush=True)

    def close(self):
        if self.tty and self.done:
            print("\r" + " " * 96 + "\r", end="", flush=True)

    @staticmethod
    def _fmt(seconds: float) -> str:
        m, sec = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m" if h else (f"{m}m{sec:02d}s" if m else f"{sec}s")


def _hold_windows_on_frames(ends: np.ndarray, probs: np.ndarray,
                            frame_groups: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Expand window scores to frame resolution, holding each until the next.

    A window model emits one score every `stride` frames. Scoring episodes on
    that sparse timeline would silently reinterpret the units — an onset
    tolerance of 12 would mean 12 *windows*, which is 60 frames at stride 5,
    five times the intended 400 ms. So each score is held across the frames
    up to the next window, which is both the correct unit and exactly what a
    live app does with a sliding buffer.
    """
    if len(ends) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0)
    order = np.argsort(ends, kind="stable")
    ends, probs = ends[order], probs[order]
    # Hold until the next window of the same session; the last window of a
    # session covers only its own frame
    next_end = np.empty(len(ends), dtype=np.int64)
    next_end[:-1] = ends[1:]
    next_end[-1] = ends[-1] + 1
    same = np.empty(len(ends), dtype=bool)
    same[:-1] = frame_groups[ends[1:]] == frame_groups[ends[:-1]]
    same[-1] = False
    stop = np.where(same, next_end, ends + 1)
    counts = np.maximum(stop - ends, 1)
    rows = np.concatenate([np.arange(e, e + c) for e, c in zip(ends, counts)])
    held = np.repeat(probs, counts)
    return rows, held


def run_dataset(
    ds: LoadedDataset,
    cfg: dict,
    out_dir: Path,
    quick: bool = False,
    folds_override: list[Fold] | None = None,
    design_override: dict | None = None,
) -> tuple[list[RunResult], dict]:
    """Train all configured models on one dataset, across all folds.

    `folds_override` replaces the participant cross-validation with folds
    built elsewhere — used by the cross-corpus transfer arm, where the split
    is by corpus rather than by participant.
    """
    seed = cfg["seed"]
    results: list[RunResult] = []

    frames, feature_columns, prep_notes = preprocess(
        _order_frames(ds.frames), ds.feature_columns, cfg["preprocessing"]
    )
    frames = _order_frames(frames)

    if folds_override is not None:
        folds, design = folds_override, dict(design_override or {})
    else:
        folds, design = make_folds(
            frames, cfg.get("evaluation", {}), cfg["split"], seed)

    details = {
        "preprocessing": prep_notes,
        "feature_count": len(feature_columns),
        "feature_groups": _feature_groups(feature_columns),
        "evaluation": design,
        "folds": [],
    }

    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    save_models = cfg.get("report", {}).get("save_models", "first_fold")

    rnn_kinds = [k for k in ("lstm", "gru") if k in cfg["models"]]
    baseline_kinds = [k for k in BASELINES if k in cfg["models"]]
    per_fold = len(baseline_kinds) + (1 if "random_forest" in cfg["models"] else 0)
    progress = _Progress(
        len(folds) * (per_fold + len(cfg["windows"]) * len(rnn_kinds)),
        f"{ds.name}")
    # Combined runs carry a `source` column; single-corpus runs have one value
    group_source = (frames.groupby("group", sort=False)["source"].first().to_dict()
                    if "source" in frames and frames["source"].nunique() > 1
                    else None)

    # ---- pass 1: fold bookkeeping and the frame-level models ----
    # Splits are held as a row-aligned Series rather than a copy of `frames`
    # per fold: on a 240k-frame dataset each copy is tens of megabytes.
    contexts = []
    for i, fold in enumerate(folds):
        split = fold.row_split(frames)
        fold_notes = dict(fold.info)

        if (split == "val").sum() == 0:
            # No val participants: train for the full epoch budget rather than
            # early-stop against test, which would leak the test set into
            # model selection. Flagged so the report says so.
            fold_notes["val_fallback"] = (
                "no val participants; early stopping disabled, trained for the "
                "full epoch budget (test is never used for model selection)")

        details["folds"].append(_fold_record(fold, frames, split, fold_notes))
        fold_extra = _fold_extra(fold, frames, split)
        # First fold owns the exported model filenames; later folds either
        # skip saving or get fold-suffixed names
        tag = None if (i == 0 or save_models == "none") else fold.name
        writing = save_models == "all" or (i == 0 and save_models != "none")
        contexts.append({
            "fold": fold, "extra": fold_extra, "tag": tag, "writing": writing,
            # group -> split, all the window pass needs from this fold
            "group_split": split.groupby(frames["group"].values,
                                         sort=False).first().to_dict(),
        })

        if "random_forest" in cfg["models"]:
            results.append(_run_random_forest(
                ds.name, fold, frames, split, feature_columns, cfg, seed,
                models_dir, quick, writing, tag, fold_extra))
            progress.step(f"{fold.name} random_forest")

        for kind in baseline_kinds:
            results.append(_run_baseline(
                kind, ds.name, fold, frames, split, cfg, fold_extra))
            progress.step(f"{fold.name} {kind}")

    # ---- pass 2: the recurrent models, one window config at a time ----
    # Window tensors are fold-independent, so each is built once and reused
    # across folds — but only one is held at a time. Keeping all three
    # resident costs ~750 MB on a 240k-frame dataset, which is the difference
    # between fitting in RAM and swapping.
    frame_truth = (frames["label"] == "writing").to_numpy(dtype=np.int8)
    frame_groups = frames["group"].to_numpy(dtype=object)
    frame_scored = _scored_mask(frames)
    label_mode = cfg.get("evaluation", {}).get("window_label", "last")
    details["window_label"] = label_mode
    for window_cfg in cfg["windows"]:
        if not rnn_kinds:
            break
        X, y, groups, ends = make_windows(
            frames, feature_columns, window_cfg["size"], window_cfg["stride"],
            label_mode=label_mode)
        w_source = (np.array([group_source[g] for g in groups])
                    if group_source else None)
        for ctx in contexts:
            w_split = np.array([ctx["group_split"][g] for g in groups])
            # Slice once per fold, not once per architecture: LSTM and GRU
            # train on exactly the same arrays
            arrays = {s: (X[w_split == s], y[w_split == s]) for s in SPLITS}
            test_sources = (w_source[w_split == "test"]
                            if w_source is not None else None)
            test_ends = ends[w_split == "test"]
            for kind in rnn_kinds:
                results.append(_run_recurrent(
                    kind, ds.name, ctx["fold"], arrays, window_cfg, cfg,
                    seed, models_dir, quick, ctx["writing"], ctx["tag"],
                    ctx["extra"], test_sources, test_ends,
                    frame_truth, frame_groups, frame_scored))
                progress.step(f"{ctx['fold'].name} {kind} "
                              f"w{window_cfg['size']}")
            del arrays
        del X, y, groups, ends
        gc.collect()

    # Results come out grouped by window; sort back into fold order so the
    # report and results.json read the way the run was described
    progress.close()
    order = {f.name: i for i, f in enumerate(folds)}
    results.sort(key=lambda r: order.get(r.fold, 0))
    return results, details


def _feature_groups(feature_columns: list[str]) -> dict:
    """Count features by family, so the report says what the model actually saw."""
    groups = {"pose": 0, "motion": 0, "velocity": 0, "other": 0}
    for c in feature_columns:
        if c.startswith("mot_"):
            groups["motion"] += 1
        elif c.startswith("d_"):
            groups["velocity"] += 1
        elif c.startswith("l") and c[1:2].isdigit():
            groups["pose"] += 1
        else:
            groups["other"] += 1
    return {k: v for k, v in groups.items() if v}


def _prior(part: pd.DataFrame) -> float | None:
    if "scored" in part:
        part = part[part["scored"].to_numpy(dtype=bool)]
    if part.empty:
        return None
    return round(float((part["label"] == "writing").mean()), 4)


def _fold_record(fold: Fold, frames: pd.DataFrame, split: pd.Series,
                 notes: dict) -> dict:
    """Per-fold provenance: who is where, and the class prior of each split.

    The priors are logged for every fold precisely so an unrepresentative
    validation prior cannot silently reappear.
    """
    record = {"fold": fold.name, **notes}
    seen: dict[str, set[str]] = {}
    for s in SPLITS:
        part = frames[(split == s).to_numpy()]
        people = sorted(part["participant"].unique().tolist())
        record[f"{s}_participants"] = fold.participants(s) or people
        record[f"{s}_frames"] = int(len(part))
        record[f"{s}_writing_prior"] = _prior(part)
        for p in people:
            seen.setdefault(p, set()).add(s)

    # Participants on both sides of the split. Zero by construction under
    # participant folds; under a session split this is the leakage, counted
    leaked = sorted(p for p, splits in seen.items() if len(splits) > 1)
    record["participants_in_multiple_splits"] = leaked
    if leaked:
        leaked_frames = int(frames["participant"].isin(leaked).sum())
        record["leaked_participant_frame_share"] = round(
            leaked_frames / max(len(frames), 1), 4)
    return record


def _fold_extra(fold: Fold, frames: pd.DataFrame, split: pd.Series) -> dict:
    part = lambda s: frames[(split == s).to_numpy()]
    test = part("test")
    extra = {
        "test_participants": (fold.participants("test")
                              or sorted(test["participant"].unique().tolist())),
        "test_writing_prior": _prior(test),
        "val_writing_prior": _prior(part("val")),
        "train_writing_prior": _prior(part("train")),
    }
    if "hand_detected" in test.columns and len(test):
        detected = test["hand_detected"].to_numpy().astype(bool)
        extra["test_hand_detection_rate"] = round(float(detected.mean()), 4)
    return extra


def _by_source(y_true, y_prob, sources, threshold=DEFAULT_THRESHOLD) -> dict | None:
    """Test metrics broken down by originating corpus.

    Only meaningful for a combined run: it answers whether pooling helped the
    corpus you care about, or only lifted the average.
    """
    if sources is None or len(y_true) == 0:
        return None
    unique = pd.unique(sources)
    if len(unique) < 2:
        return None
    out = {}
    for src in sorted(unique):
        m = sources == src
        if not m.any():
            continue
        metrics = evaluate_binary(y_true[m], y_prob[m], threshold=threshold)
        out[str(src)] = {k: metrics[k] for k in
                         ("n_samples", "accuracy", "precision_writing",
                          "recall_writing", "f1_writing", "roc_auc")}
    return out


def _by_detection(y_true, y_prob, detected, threshold) -> dict | None:
    """Test metrics split by whether MediaPipe found a hand at all.

    P(writing | no hand) is 0.03 against 0.38 when a hand is present, so
    "no hand" is very nearly a free correct answer and a model can lean on it
    instead of learning writing. The live app sees a *lower* detection rate
    than training did — the collector extracts with inter-frame tracking the
    app cannot match — so any score propped up by these frames will not
    survive deployment. Splitting them out is how that shows up.
    """
    if detected is None or not len(y_true):
        return None
    detected = np.asarray(detected).astype(bool)
    if detected.all() or not detected.any():
        return None
    out = {}
    for label, mask in (("hand_detected", detected), ("no_hand", ~detected)):
        m = evaluate_binary(y_true[mask], y_prob[mask], threshold=threshold)
        out[label] = {k: m[k] for k in
                      ("n_samples", "positive_rate", "accuracy",
                       "precision_writing", "recall_writing", "f1_writing")}
    return out


def _scored_mask(part: pd.DataFrame) -> np.ndarray:
    return (part["scored"].to_numpy(dtype=bool) if "scored" in part
            else np.ones(len(part), bool))


def _split_frame_arrays(frames, split, columns):
    """Scored rows of each split — what a frame model trains and is fitted on."""
    out = {}
    for s in SPLITS:
        part = frames[(split == s).to_numpy()]
        part = part[_scored_mask(part)]
        out[s] = (
            part[columns].to_numpy(dtype=np.float32),
            (part["label"] == "writing").to_numpy(dtype=np.int8),
        )
    return out


def _run_random_forest(ds_name, fold, frames, split, feature_columns, cfg, seed,
                       models_dir, quick, writing, tag, fold_extra):
    params = dict(cfg["models"]["random_forest"])
    if quick:
        params["n_estimators"] = min(params.get("n_estimators", 300), 50)

    arrays = _split_frame_arrays(frames, split, feature_columns)
    X_train, y_train = arrays["train"]
    X_val, y_val = arrays["val"]
    X_test, y_test = arrays["test"]

    model = build_random_forest(params, seed)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_seconds = time.time() - t0

    def probs(X):
        return model.predict_proba(X)[:, 1] if len(X) else np.empty(0)

    p_train, p_val, p_test = probs(X_train), probs(X_val), probs(X_test)

    # Where to fit the decision threshold. Neither option is free:
    #
    #   val   — unbiased with respect to participant (the held-out person is
    #           genuinely unlike anyone fitted on), but only two people, and
    #           measurably unstable: the threshold swung 0.25-0.54 across
    #           folds and every high-threshold fold collapsed to near-zero
    #           recall, because a point tuned on two people does not transfer.
    #   oob   — out-of-bag predictions are out-of-sample per row and there are
    #           ~44k of them instead of ~8k, but the row's *participant* was
    #           still in training, so the estimate carries some optimism.
    #   train — largest sample, most optimistic; the forest has fitted it.
    #
    # Which transfers better is an empirical question, so it is a setting
    # rather than an assumption. See the threshold_source comparison in the
    # report.
    source = cfg.get("evaluation", {}).get("threshold_source", "oob")
    threshold, thr_notes = None, {}
    if source == "oob":
        oob = getattr(model, "oob_decision_function_", None)
        if oob is not None and len(oob) == len(y_train):
            p_oob = oob[:, 1]
            usable = np.isfinite(p_oob)
            if usable.sum() > 100 and len(np.unique(y_train[usable])) > 1:
                threshold, thr_notes = _fit_operating_point(
                    cfg, y_train[usable], p_oob[usable], y_train, p_train)
                thr_notes["threshold_fitted_on"] = (
                    f"out-of-bag predictions over {int(usable.sum()):,} "
                    "training rows (out-of-sample per row; participants seen)")
    elif source == "train":
        threshold, thr_notes = _fit_operating_point(
            cfg, y_train, p_train, y_train, p_train)
        thr_notes["threshold_fitted_on"] = "train split (optimistic — model fitted it)"
    if threshold is None:
        threshold, thr_notes = _fit_operating_point(cfg, y_val, p_val, y_train, p_train)

    importances = sorted(
        zip(feature_columns, model.feature_importances_.tolist()),
        key=lambda t: t[1], reverse=True,
    )
    path = None
    if writing:
        suffix = f"_{tag}" if tag else ""
        path = models_dir / f"{ds_name}_random_forest{suffix}.joblib"
        joblib.dump(model, path)

    # The whole test timeline, unsure frames included, is predicted — the
    # timeline metrics need it unbroken — but only scored frames are scored
    test_part = frames[(split == "test").to_numpy()]
    test_scored = _scored_mask(test_part)
    test_truth = (test_part["label"] == "writing").to_numpy(dtype=np.int8)
    test_groups = test_part["group"].to_numpy(dtype=object)
    p_timeline = probs(test_part[feature_columns].to_numpy(dtype=np.float32))
    scored_part = test_part[test_scored]
    detected = (scored_part["hand_detected"].to_numpy()
                if "hand_detected" in scored_part else None)

    return RunResult(
        dataset=ds_name, model="random_forest", window=None, fold=fold.name,
        train_seconds=train_seconds,
        n_train=len(y_train), n_val=len(y_val), n_test=len(y_test),
        test_metrics=evaluate_binary(y_test, p_test, threshold=threshold),
        val_metrics=(evaluate_binary(y_val, p_val, threshold=threshold)
                     if len(y_val) else None),
        train_metrics=evaluate_binary(y_train, p_train, threshold=threshold),
        feature_importances=importances[:20],
        model_path=str(path) if path else None,
        event_metrics=_frame_events(cfg, test_truth, p_timeline, test_groups,
                                    threshold, test_scored),
        **dict(zip(("smoothed_metrics", "smoothed_events"),
                   _smoothed(cfg, test_truth, p_timeline, test_groups,
                             threshold, test_scored))),
        extra={"params": params, "granularity": "per frame", **thr_notes,
               **fold_extra,
               "test_metrics_by_source": _by_source(
                   y_test, p_test,
                   scored_part["source"].to_numpy()
                   if "source" in scored_part else None, threshold),
               "test_metrics_by_detection": _by_detection(
                   y_test, p_test, detected, threshold)},
    )


def _run_baseline(kind, ds_name, fold, frames, split, cfg, fold_extra):
    """Rule-based baseline: fit a threshold on training data, apply it to test.

    A threshold is one free parameter fitted over tens of thousands of
    frames, so unlike a learned model it cannot meaningfully overfit the
    split it is fitted on — and fitting it on the two validation participants
    instead measurably cost about two points of F1 in pure small-sample
    noise. The comparison stays fair because every model now fits its
    operating point on the largest *unbiased* sample available to it: the
    forest uses out-of-bag predictions over all training participants, the
    recurrent models use validation (they have no cheap out-of-sample
    equivalent), and these rules use the training split.
    """
    baseline = BASELINES[kind](cfg["models"].get(kind) or {})
    column = baseline.signal_column

    # Fitted and scored on scored frames; the test timeline is kept whole
    # for the timeline metrics
    timeline = frames[(split == "test").to_numpy()]
    timeline_scored = _scored_mask(timeline)
    timeline_sig = timeline[column].to_numpy(dtype=np.float64)
    timeline_truth = (timeline["label"] == "writing").to_numpy(dtype=np.int8)
    timeline_groups = timeline["group"].to_numpy(dtype=object)
    parts = {s: frames[(split == s).to_numpy()] for s in SPLITS}
    parts = {s: p[_scored_mask(p)] for s, p in parts.items()}
    sig = {s: p[column].to_numpy(dtype=np.float64) for s, p in parts.items()}
    y = {s: (p["label"] == "writing").to_numpy(dtype=np.int8)
         for s, p in parts.items()}

    beta = float(cfg.get("evaluation", {}).get("threshold_beta", 1.0))
    fit_on = "train" if (len(y["train"]) and len(np.unique(y["train"])) > 1) else "val"
    t0 = time.time()
    baseline.fit(sig[fit_on], y[fit_on], beta=beta)
    train_seconds = time.time() - t0

    def metrics(s):
        if not len(y[s]):
            return None
        return evaluate_binary(y[s], baseline.decision_scores(sig[s]),
                               threshold=baseline.threshold)

    detected = (parts["test"]["hand_detected"].to_numpy()
                if "hand_detected" in parts["test"] else None)
    pct_scores, pct_threshold = _to_percentiles(
        baseline.decision_scores(sig[fit_on]),
        baseline.decision_scores(timeline_sig), baseline.threshold)

    return RunResult(
        dataset=ds_name, model=kind, window=None, fold=fold.name,
        train_seconds=train_seconds,
        n_train=len(y["train"]), n_val=len(y["val"]), n_test=len(y["test"]),
        test_metrics=metrics("test"),
        val_metrics=metrics("val"),
        train_metrics=metrics("train"),
        event_metrics=_frame_events(
            cfg, timeline_truth, baseline.decision_scores(timeline_sig),
            timeline_groups, baseline.threshold, timeline_scored),
        **dict(zip(("smoothed_metrics", "smoothed_events"),
                   _smoothed(cfg, timeline_truth, pct_scores, timeline_groups,
                             pct_threshold, timeline_scored))),
        extra={
            "params": {"threshold": round(baseline.threshold, 6),
                       "signal": column, **baseline.params},
            "granularity": "per frame",
            "rule": baseline.description,
            "threshold_fitted_on": (f"{fit_on} split, threshold maximising "
                                    f"F{beta:g}(writing)"),
            "fitted_on": (f"{fit_on} split, threshold maximising "
                          f"F{beta:g}(writing)"),
            **fold_extra,
            "test_metrics_by_source": _by_source(
                y["test"], baseline.decision_scores(sig["test"]),
                parts["test"]["source"].to_numpy()
                if "source" in parts["test"] else None,
                threshold=baseline.threshold),
            "test_metrics_by_detection": _by_detection(
                y["test"], baseline.decision_scores(sig["test"]), detected,
                baseline.threshold),
        },
    )


def _run_recurrent(kind, ds_name, fold, arrays, window_cfg, cfg, seed,
                   models_dir, quick, writing, tag, fold_extra,
                   test_sources=None, test_ends=None,
                   frame_truth=None, frame_groups=None, frame_scored=None):
    from tensorflow import keras

    params = dict(cfg["models"][kind])
    if quick:
        params["epochs"] = min(params.get("epochs", 60), 3)

    X_train, y_train = arrays["train"]
    X_val, y_val = arrays["val"]
    X_test, y_test = arrays["test"]

    n_pos = max(1, int(y_train.sum()))
    n_neg = max(1, len(y_train) - n_pos)
    class_weight = {0: len(y_train) / (2 * n_neg), 1: len(y_train) / (2 * n_pos)}

    keras.backend.clear_session()
    model = build_recurrent(kind, params, window_cfg["size"], X_train.shape[2])
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=params.get("patience", 8),
            restore_best_weights=True,
        )
    ]
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val) if len(y_val) else None,
        epochs=params.get("epochs", 60),
        batch_size=params.get("batch_size", 64),
        class_weight=class_weight,
        callbacks=callbacks if len(y_val) else [],
        verbose=0,
    )
    train_seconds = time.time() - t0

    def probs(Xp):
        return model.predict(Xp, verbose=0).ravel() if len(Xp) else np.empty(0)

    p_train, p_val, p_test = probs(X_train), probs(X_val), probs(X_test)
    threshold, thr_notes = _fit_operating_point(cfg, y_val, p_val, y_train, p_train)

    path = None
    if writing:
        suffix = f"_{tag}" if tag else ""
        path = models_dir / f"{ds_name}_{kind}_w{window_cfg['size']}{suffix}.keras"
        model.save(path)

    # Window scores laid back on the frame timeline, held until the next
    # window — the live app's behaviour, and the only way these numbers are
    # comparable with the frame-level models' event scores
    events = None
    smoothed_metrics = smoothed_events = None
    if test_ends is not None and len(test_ends) and frame_truth is not None:
        rows, held = _hold_windows_on_frames(test_ends, p_test, frame_groups)
        held_scored = frame_scored[rows] if frame_scored is not None else None
        smoothed_metrics, smoothed_events = _smoothed(
            cfg, frame_truth[rows], held, frame_groups[rows], threshold, held_scored)
        events = _timeline_metrics(cfg, frame_truth[rows],
                                   (held >= threshold).astype(int),
                                   frame_groups[rows], held_scored)
    if events is not None:
        events["timeline"] = (
            "window scores expanded to frame resolution, each held until the "
            f"next window (stride {window_cfg['stride']} frames) — so the "
            "tolerance and minimum-episode settings are in real frames and "
            "these numbers are comparable with the frame-level models'")

    return RunResult(
        dataset=ds_name, model=kind, window=dict(window_cfg), fold=fold.name,
        train_seconds=train_seconds,
        n_train=len(y_train), n_val=len(y_val), n_test=len(y_test),
        test_metrics=evaluate_binary(y_test, p_test, threshold=threshold),
        val_metrics=(evaluate_binary(y_val, p_val, threshold=threshold)
                     if len(y_val) else None),
        train_metrics=evaluate_binary(y_train, p_train, threshold=threshold),
        history={k: [float(v) for v in vs] for k, vs in history.history.items()},
        epochs_trained=len(history.history.get("loss", [])),
        model_path=str(path) if path else None,
        event_metrics=events,
        smoothed_metrics=smoothed_metrics,
        smoothed_events=smoothed_events,
        extra={"params": params, "class_weight": class_weight,
               "granularity": f"windows of {window_cfg['size']} frames",
               "window_label": cfg.get("evaluation", {}).get("window_label", "last"),
               **thr_notes, **fold_extra,
               "test_metrics_by_source": _by_source(
                   y_test, p_test, test_sources, threshold),
               "window_writing_prior": {
                   "train": round(float(y_train.mean()), 4) if len(y_train) else None,
                   "val": round(float(y_val.mean()), 4) if len(y_val) else None,
                   "test": round(float(y_test.mean()), 4) if len(y_test) else None,
               }},
    )


# --------------------------------------------------------------------------
# Cross-corpus transfer
# --------------------------------------------------------------------------


def run_transfer(train_ds: LoadedDataset, test_ds: LoadedDataset, cfg: dict,
                 out_dir: Path, quick: bool = False
                 ) -> tuple[list[RunResult], dict]:
    """Train on the whole of one corpus, evaluate on the whole of another.

    This is the experiment that decides whether the data source matters. The
    pipeline extracts identical features from every corpus, which makes it
    tempting to treat provenance as a nuisance variable and pool everything —
    but identical *features* do not imply identical *labels*. If source were
    truly irrelevant, transfer would score close to within-corpus
    cross-validation. Measuring the gap turns that assumption into a result.

    Validation participants are drawn from the training corpus, so the target
    corpus is untouched during model selection — including the threshold fit.
    """
    from src.data.loaders import combine_datasets
    from src.data.splits import Fold, _choose_val_participants

    pooled = combine_datasets({train_ds.name: train_ds, test_ds.name: test_ds},
                              name=f"{train_ds.name}->{test_ds.name}")
    frames = _order_frames(pooled.frames)
    is_test = (frames["source"] == test_ds.name).to_numpy()

    rng = np.random.default_rng(cfg["seed"])
    train_people = sorted(frames.loc[~is_test, "participant"].unique())
    val, val_info = _choose_val_participants(
        frames[~is_test], train_people,
        int(cfg.get("evaluation", {}).get("val_participants", 2)),
        float(cfg.get("evaluation", {}).get("val_fraction", 0.15)), rng)

    assignment = pd.Series("train", index=frames.index)
    assignment[is_test] = "test"
    assignment[(~is_test) & frames["participant"].isin(val).to_numpy()] = "val"

    fold = Fold(
        name=f"transfer_{train_ds.name}_to_{test_ds.name}",
        assignment={}, precomputed_rows=assignment,
        info={"train_corpus": train_ds.name, "test_corpus": test_ds.name,
              "val_participants": val, **val_info},
    )
    design = {
        "scheme": "cross-corpus transfer",
        "folds": 1,
        "unit": "corpus",
        "train_corpus": train_ds.name,
        "test_corpus": test_ds.name,
        "note": ("trained on every participant of the training corpus and "
                 "evaluated on every participant of the target corpus; "
                 "validation (and the threshold fit) come from the training "
                 "corpus only"),
    }
    pooled = LoadedDataset(pooled.name, frames, pooled.feature_columns,
                           {**pooled.meta, "transfer": design})
    return run_dataset(pooled, cfg, out_dir, quick=quick,
                       folds_override=[fold], design_override=design)
