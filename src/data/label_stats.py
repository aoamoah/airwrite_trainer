"""What a corpus's labels actually mark, measured.

Two corpora can share a label vocabulary and still mean different things by
it. Measured on 2026-09-11: in `dataset` the median pause between writing
runs is 462 ms and 28% of pauses are under 167 ms — the labels mark pauses
inside letters. In `dataset_IPN` the median gap is 11 s and 1.5% are under
167 ms — the labels mark whole gestures. Pooling the two teaches "is someone
in a writing episode", not "is the pen down right now".

So every corpus's granularity is measured at load time and printed in the
report, and a new source can be checked before it is pooled.
"""

import numpy as np
import pandas as pd

PAUSE_THRESHOLDS_MS = (167, 400)


def _runs(labels: np.ndarray):
    """(value, start, end) runs of a 1-D array, end inclusive."""
    if len(labels) == 0:
        return []
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change - 1, len(labels) - 1]
    return [(labels[s], s, e) for s, e in zip(starts, ends)]


def label_granularity(frames: pd.DataFrame) -> dict:
    """Writing-run and interior-pause durations across sessions, in ms.

    A pause is a not_writing run with writing on both sides in the same
    session, the unit the pen-in-air benchmark scores. `unsure` frames are
    neither: a run touching one is left out rather than guessed at.
    """
    writing_ms, pause_ms = [], []
    for _, s in frames.groupby("group", sort=False):
        labels = s["label"].to_numpy()
        ts = pd.to_numeric(s["timestamp_ms"], errors="coerce").to_numpy(dtype=np.float64) \
            if "timestamp_ms" in s else np.arange(len(s)) * (1000.0 / 30.0)
        frame_ms = np.median(np.diff(ts)) if len(ts) > 1 else 1000.0 / 30.0
        runs = _runs(labels)
        for k, (value, a, b) in enumerate(runs):
            ms = (b - a + 1) * frame_ms
            if value == "writing":
                writing_ms.append(ms)
            elif (value == "not_writing" and 0 < k < len(runs) - 1
                  and runs[k - 1][0] == "writing" and runs[k + 1][0] == "writing"):
                pause_ms.append(ms)

    def pct(values):
        if not values:
            return None
        return {f"p{q}": round(float(np.percentile(values, q))) for q in (10, 25, 50, 75, 90)}

    return {
        "writing_runs": len(writing_ms),
        "writing_run_ms": pct(writing_ms),
        "interior_pauses": len(pause_ms),
        "interior_pause_ms": pct(pause_ms),
        **{f"pauses_under_{t}ms": (round(float(np.mean(np.array(pause_ms) < t)), 4)
                                   if pause_ms else None)
           for t in PAUSE_THRESHOLDS_MS},
        "unsure_frames": int((frames["label"] == "unsure").sum()),
    }
