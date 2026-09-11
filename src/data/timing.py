"""Put every session on one frame rate.

Almost everything downstream counts frames: the motion features' rolling
windows (5/15/31), the recurrent models' windows (15/30/60) and strides, the
smoothing durations, the event tolerances. At 30 fps a 30-frame window is one
second; at 60 fps it is half a second, and at 25 fps 1.2 s. With several
sources recorded on different devices, "the same model" would be seeing
different amounts of time depending on where a session came from.

Resampling each session to `target_fps` makes a frame mean the same duration
everywhere, so every frame-denominated setting becomes a time. The resampler
is sample-and-hold on the real timestamps: at each grid tick it takes the
latest frame at or before that tick. That is causal — a live app can do
exactly the same with its own stream — and it never invents a pose that was
not observed, which interpolation would.

A session already within `tolerance` of the target is left untouched, so a
30 fps session is not shuffled by a few milliseconds of clock jitter.
"""

import numpy as np
import pandas as pd

DEFAULT_TOLERANCE = 0.05


def measured_fps(timestamps_ms: np.ndarray) -> float | None:
    ts = np.asarray(timestamps_ms, dtype=np.float64)
    if len(ts) < 2:
        return None
    dt = np.diff(ts)
    dt = dt[dt > 0]
    return float(1000.0 / np.median(dt)) if len(dt) else None


def resample_session(df: pd.DataFrame, target_fps: float,
                     tolerance: float = DEFAULT_TOLERANCE) -> tuple[pd.DataFrame, dict]:
    """Sample-and-hold one session onto a `target_fps` grid.

    Rows must be in frame order and carry `timestamp_ms`. Returns the
    resampled frame (frame_index renumbered, the original kept as
    `source_frame_index`) and a note of what was done.
    """
    ts = pd.to_numeric(df["timestamp_ms"], errors="coerce").to_numpy(dtype=np.float64)
    fps = measured_fps(ts)
    note = {"measured_fps": round(fps, 3) if fps else None}
    if fps is None or np.isnan(ts).any() or abs(fps - target_fps) <= tolerance * target_fps:
        out = df.copy()
        out["source_frame_index"] = out["frame_index"].to_numpy()
        return out, {**note, "resampled": False}

    step = 1000.0 / target_fps
    grid = np.arange(ts[0], ts[-1] + 1e-6, step)
    # Latest frame at or before each tick; timestamps are non-decreasing
    rows = np.clip(np.searchsorted(ts, grid, side="right") - 1, 0, len(ts) - 1)
    out = df.iloc[rows].copy()
    out["source_frame_index"] = out["frame_index"].to_numpy()
    out["frame_index"] = np.arange(len(out))
    out["timestamp_ms"] = np.round(grid).astype(np.int64)
    out = out.reset_index(drop=True)
    repeated = int((np.diff(rows) == 0).sum())
    return out, {**note, "resampled": True, "frames_before": len(df),
                 "frames_after": len(out), "held_repeats": repeated}
