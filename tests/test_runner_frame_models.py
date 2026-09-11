"""The runner end to end with the frame-level models only (a small random
forest and the rule baselines), on synthetic data in a temp directory — the
wiring check for unsure masking and the timeline metrics. No recurrent
models, no GPU, nothing written outside tmp_path."""

import copy

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.loaders import LoadedDataset, LANDMARK_COLUMNS
from src.training.aggregate import aggregate_results
from src.report.builder import build_report
from src.training.runner import run_dataset


def _synthetic(n_people=4, n=400, seed=0):
    rng = np.random.default_rng(seed)
    parts = []
    for p in range(n_people):
        writing = np.zeros(n, bool)
        writing[60:180] = True
        writing[200:330] = True          # pause 180..199
        speed = np.where(writing, 0.02, 0.002) + rng.normal(0, 0.002, n)
        wrist = np.cumsum(np.c_[speed, speed * 0.3], axis=0) % 1.0
        pts = wrist[:, None, :] + rng.normal(0, 0.03, (n, 21, 2))
        z = rng.normal(0, 0.01, (n, 21, 1))
        df = pd.DataFrame(np.concatenate([pts, z], axis=2).reshape(n, 63),
                          columns=LANDMARK_COLUMNS)
        df["frame_index"] = np.arange(n)
        df["timestamp_ms"] = (np.arange(n) * 1000 / 30).astype(int)
        df["hand_detected"] = 1
        label = np.where(writing, "writing", "not_writing").astype(object)
        label[340:350] = "unsure"
        df["label"] = label
        df["scored"] = df["label"] != "unsure"
        df["group"] = f"P{p:03d}/S{p:03d}"
        df["participant"] = f"P{p:03d}"
        df["source"] = "synthetic"
        df["aspect"] = 4 / 3
        parts.append(df)
    frames = pd.concat(parts, ignore_index=True)
    return LoadedDataset("synthetic", frames, ["hand_detected"] + LANDMARK_COLUMNS, {})


def test_frame_models_report_pause_and_segmental_metrics(tmp_path):
    cfg = copy.deepcopy(load_config())
    cfg["models"] = {"random_forest": {"n_estimators": 20, "n_jobs": 1,
                                       "class_weight": "balanced"},
                     "velocity_threshold": {"search_steps": 49}}
    cfg["evaluation"]["threshold_source"] = "val"
    cfg["evaluation"]["val_participants"] = 1
    ds = _synthetic()
    ds.meta["label_granularity"] = {"writing_runs": 8, "writing_run_ms": {"p50": 4000},
                                    "interior_pauses": 4, "interior_pause_ms": {"p50": 633},
                                    "pauses_under_167ms": 0.0, "pauses_under_400ms": 0.0,
                                    "unsure_frames": 40}
    results, details = run_dataset(ds, cfg, tmp_path, quick=True)

    assert results, "no results"
    for r in results:
        ev = r.event_metrics
        assert {"pause@5f", "pause@10f", "pause@12f", "edit_score",
                "seg_f1@10", "seg_f1@25", "seg_f1@50",
                "median_onset_delay_frames"} <= set(ev)
        assert ev["n_true_pauses"] == 1            # one pause per held-out session
        # frame metrics are over scored frames only
        assert r.test_metrics["n_samples"] == 400 - 10
    aggs = aggregate_results(results)
    assert all("pause@12f" in a.events and "edit_score" in a.events for a in aggs)

    report = build_report(cfg, {"synthetic": ds}, {"synthetic": details}, results, tmp_path)
    text = report.read_text()
    for heading in ("Label granularity", "Pause detection (pen-in-air protocol)",
                    "Segmentation quality", "pen-in-air (published)", "OnlineTAS"):
        assert heading in text, heading


def test_rule_scores_are_smoothed_in_percentile_units():
    from src.training.runner import _to_percentiles
    ref = np.linspace(0, 10, 101)                 # a speed signal, not a probability
    scores, thr = _to_percentiles(ref, np.array([0.0, 2.9, 3.0, 7.5, 12.0]), 3.0)
    assert thr == 30 / 101                          # share of the reference below it
    assert list(scores >= thr) == [False, False, True, True, True]   # decisions unchanged
    assert scores.min() >= 0 and scores.max() <= 1
