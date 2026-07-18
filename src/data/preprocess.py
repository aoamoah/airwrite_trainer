"""Feature preprocessing and sequence windowing."""

import re

import numpy as np
import pandas as pd

_LM_COL = re.compile(r"^l(\d+)_([xyz])$")


def _landmark_triples(feature_columns: list[str]) -> list[tuple[str, str, str]]:
    """Group MediaPipe landmark columns into (x, y, z) triples, in order."""
    by_index: dict[int, dict[str, str]] = {}
    for col in feature_columns:
        m = _LM_COL.match(col)
        if m:
            by_index.setdefault(int(m.group(1)), {})[m.group(2)] = col
    return [
        (axes["x"], axes["y"], axes["z"])
        for _, axes in sorted(by_index.items())
        if len(axes) == 3
    ]


def preprocess(frames: pd.DataFrame, feature_columns: list[str], cfg: dict) -> tuple[pd.DataFrame, list[str], dict]:
    """Apply configured preprocessing. Returns (frames, feature_columns, notes)."""
    frames = frames.copy()
    notes = {}

    if cfg.get("drop_undetected") and "hand_detected" in frames.columns:
        before = len(frames)
        frames = frames[frames["hand_detected"] == 1].reset_index(drop=True)
        notes["dropped_undetected_frames"] = before - len(frames)

    triples = _landmark_triples(feature_columns)
    if cfg.get("normalize") == "wrist" and triples:
        wrist_x, wrist_y, wrist_z = triples[0]
        xs = frames[[t[0] for t in triples]].to_numpy()
        ys = frames[[t[1] for t in triples]].to_numpy()
        zs = frames[[t[2] for t in triples]].to_numpy()

        xs = xs - xs[:, [0]]
        ys = ys - ys[:, [0]]
        zs = zs - zs[:, [0]]
        # Scale by hand span (max distance from wrist); zero rows (no hand)
        # stay zero
        span = np.sqrt(xs**2 + ys**2 + zs**2).max(axis=1)
        span[span == 0] = 1.0
        xs /= span[:, None]
        ys /= span[:, None]
        zs /= span[:, None]

        frames[[t[0] for t in triples]] = xs
        frames[[t[1] for t in triples]] = ys
        frames[[t[2] for t in triples]] = zs
        notes["normalization"] = "wrist-origin, hand-span scaled"
    else:
        notes["normalization"] = "none"

    if cfg.get("add_velocity"):
        lm_cols = [c for t in triples for c in t]
        vel = (
            frames.groupby("group", sort=False)[lm_cols]
            .diff()
            .fillna(0.0)
            .add_prefix("d_")
        )
        frames = pd.concat([frames, vel], axis=1)
        feature_columns = feature_columns + list(vel.columns)
        notes["velocity_features"] = len(vel.columns)

    return frames, feature_columns, notes


def make_windows(
    frames: pd.DataFrame,
    feature_columns: list[str],
    size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding windows within each group (never across session bounds).

    Returns (X [n, size, features], y [n] with 1=writing, groups [n]).
    Window label is the majority frame label.
    """
    xs, ys, gs = [], [], []
    for group, gdf in frames.groupby("group", sort=False):
        feats = gdf[feature_columns].to_numpy(dtype=np.float32)
        labels = (gdf["label"] == "writing").to_numpy(dtype=np.int8)
        for start in range(0, len(gdf) - size + 1, stride):
            xs.append(feats[start:start + size])
            ys.append(int(labels[start:start + size].sum() * 2 > size))
            gs.append(group)
    if not xs:
        return (np.empty((0, size, len(feature_columns)), dtype=np.float32),
                np.empty(0, dtype=np.int8), np.empty(0, dtype=object))
    return np.stack(xs), np.array(ys, dtype=np.int8), np.array(gs, dtype=object)
