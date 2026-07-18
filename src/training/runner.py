"""Orchestrates every model × dataset × window combination and collects
everything the report needs."""

import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.loaders import LoadedDataset
from src.data.preprocess import preprocess, make_windows
from src.data.splits import assign_groups
from src.models.builders import build_random_forest, build_recurrent
from src.training.evaluate import evaluate_binary


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
    val_metrics: dict | None = None
    train_metrics: dict | None = None
    history: dict | None = None    # keras per-epoch history
    epochs_trained: int | None = None
    feature_importances: list | None = None
    model_path: str | None = None
    extra: dict = field(default_factory=dict)


def run_dataset(
    ds: LoadedDataset,
    cfg: dict,
    out_dir: Path,
    quick: bool = False,
) -> tuple[list[RunResult], dict]:
    """Train all configured models on one dataset. Returns (results, details)."""
    seed = cfg["seed"]
    results: list[RunResult] = []

    frames, feature_columns, prep_notes = preprocess(
        ds.frames, ds.feature_columns, cfg["preprocessing"]
    )

    split_cfg = cfg["split"]
    assignment, split_info = assign_groups(
        frames, split_cfg["method"], tuple(split_cfg["ratios"]), seed,
        chunk_rows=split_cfg.get("chunk_rows", 500),
    )
    frames = frames.assign(split=assignment.values)
    if (frames["split"] == "val").sum() == 0:
        # Too few units for a val split — reuse test for early stopping and
        # flag it in the report
        frames.loc[frames["split"] == "test", "split"] = "val_test"
        frames["split"] = frames["split"].replace({"val_test": "test"})
        split_info["val_fallback"] = "no val units; test set reused for early stopping"

    details = {
        "preprocessing": prep_notes,
        "feature_count": len(feature_columns),
        "split": split_info,
        "class_balance_per_split": {
            s: frames.loc[frames["split"] == s, "label"].value_counts().to_dict()
            for s in ("train", "val", "test")
        },
    }

    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if "random_forest" in cfg["models"]:
        results.append(
            _run_random_forest(ds.name, frames, feature_columns, cfg, seed,
                               models_dir, quick)
        )

    rnn_kinds = [k for k in ("lstm", "gru") if k in cfg["models"]]
    if rnn_kinds:
        group_split = frames.groupby("group")["split"].first().to_dict()
        for window_cfg in cfg["windows"]:
            size, stride = window_cfg["size"], window_cfg["stride"]
            X, y, groups = make_windows(frames, feature_columns, size, stride)
            w_split = np.array([group_split[g] for g in groups])
            for kind in rnn_kinds:
                results.append(
                    _run_recurrent(kind, ds.name, X, y, w_split, window_cfg,
                                   cfg, seed, models_dir, quick)
                )

    return results, details


def _split_frame_arrays(frames, feature_columns):
    out = {}
    for split in ("train", "val", "test"):
        part = frames[frames["split"] == split]
        out[split] = (
            part[feature_columns].to_numpy(dtype=np.float32),
            (part["label"] == "writing").to_numpy(dtype=np.int8),
        )
    return out


def _run_random_forest(ds_name, frames, feature_columns, cfg, seed, models_dir, quick):
    params = dict(cfg["models"]["random_forest"])
    if quick:
        params["n_estimators"] = min(params.get("n_estimators", 300), 50)

    arrays = _split_frame_arrays(frames, feature_columns)
    X_train, y_train = arrays["train"]
    X_val, y_val = arrays["val"]
    X_test, y_test = arrays["test"]

    model = build_random_forest(params, seed)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_seconds = time.time() - t0

    def probs(X):
        return model.predict_proba(X)[:, 1] if len(X) else np.empty(0)

    importances = sorted(
        zip(feature_columns, model.feature_importances_.tolist()),
        key=lambda t: t[1], reverse=True,
    )
    path = models_dir / f"{ds_name}_random_forest.joblib"
    joblib.dump(model, path)

    return RunResult(
        dataset=ds_name, model="random_forest", window=None,
        train_seconds=train_seconds,
        n_train=len(y_train), n_val=len(y_val), n_test=len(y_test),
        test_metrics=evaluate_binary(y_test, probs(X_test)),
        val_metrics=evaluate_binary(y_val, probs(X_val)) if len(y_val) else None,
        train_metrics=evaluate_binary(y_train, probs(X_train)),
        feature_importances=importances[:20],
        model_path=str(path),
        extra={"params": params, "granularity": "per frame"},
    )


def _run_recurrent(kind, ds_name, X, y, w_split, window_cfg, cfg, seed, models_dir, quick):
    from tensorflow import keras

    params = dict(cfg["models"][kind])
    if quick:
        params["epochs"] = min(params.get("epochs", 60), 3)

    tr, va, te = w_split == "train", w_split == "val", w_split == "test"
    X_train, y_train = X[tr], y[tr]
    X_val, y_val = X[va], y[va]
    X_test, y_test = X[te], y[te]

    n_pos = max(1, int(y_train.sum()))
    n_neg = max(1, len(y_train) - n_pos)
    class_weight = {0: len(y_train) / (2 * n_neg), 1: len(y_train) / (2 * n_pos)}

    model = build_recurrent(kind, params, window_cfg["size"], X.shape[2])
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

    path = models_dir / f"{ds_name}_{kind}_w{window_cfg['size']}.keras"
    model.save(path)

    return RunResult(
        dataset=ds_name, model=kind, window=dict(window_cfg),
        train_seconds=train_seconds,
        n_train=len(y_train), n_val=len(y_val), n_test=len(y_test),
        test_metrics=evaluate_binary(y_test, probs(X_test)),
        val_metrics=evaluate_binary(y_val, probs(X_val)) if len(y_val) else None,
        train_metrics=evaluate_binary(y_train, probs(X_train)),
        history={k: [float(v) for v in vs] for k, vs in history.history.items()},
        epochs_trained=len(history.history.get("loss", [])),
        model_path=str(path),
        extra={"params": params, "class_weight": class_weight,
               "granularity": f"windows of {window_cfg['size']} frames"},
    )
