"""Temporal post-processing: turn per-frame scores into stable episodes.

Every model here emits an independent decision per frame (or per window),
and the episode metrics show what that costs: each true writing episode is
recovered as two to four separate predicted fragments. A user experiences
that as the detector "cutting out" mid-stroke, and no amount of extra model
capacity fixes it, because the model is not being asked to be temporally
consistent — nothing in the loss or the decision rule mentions time.

Two standard mechanisms, applied in order:

- **Hysteresis.** Two thresholds instead of one: cross `high` to start
  writing, and fall below `low` to stop. A score hovering around a single
  threshold flips every few frames; with a gap between the two it cannot.
  This is the same dual-threshold scheme the pen-in-air video work uses.
- **Minimum durations.** Runs shorter than `min_on` frames are deleted and
  gaps shorter than `min_off` are filled. A 2-frame dropout mid-stroke is a
  tracking artifact, not the user stopping.

This lives in the trainer rather than the app on purpose: the parameters are
fitted on training data and travel to the app in `inference_spec.json`, so
the app applies a measured configuration instead of hand-tuned constants.
"""

import numpy as np

# Durations stay at or under the shortest scored pause (3 frames), so
# smoothing cannot erase a pause the evaluation expects to be detected
DEFAULTS = {"low": None, "high": None, "min_on": 3, "min_off": 3}


def apply_hysteresis(probs: np.ndarray, groups: np.ndarray,
                     low: float, high: float) -> np.ndarray:
    """Dual-threshold state machine, restarted at every session boundary."""
    probs = np.asarray(probs, dtype=np.float64)
    groups = np.asarray(groups, dtype=object)
    out = np.zeros(len(probs), dtype=np.int8)
    state = 0
    for i in range(len(probs)):
        if i == 0 or groups[i] != groups[i - 1]:
            state = 0
        if state == 0 and probs[i] >= high:
            state = 1
        elif state == 1 and probs[i] < low:
            state = 0
        out[i] = state
    return out


def enforce_durations(pred: np.ndarray, groups: np.ndarray,
                      min_on: int = 5, min_off: int = 5) -> np.ndarray:
    """Drop runs shorter than `min_on`; fill gaps shorter than `min_off`.

    Gaps are filled first: a stroke broken by a 2-frame dropout should be
    healed into one episode before short-run removal decides whether either
    half was long enough to keep.
    """
    pred = np.asarray(pred, dtype=np.int8).copy()
    groups = np.asarray(groups, dtype=object)
    if len(pred) == 0:
        return pred

    starts = [0] + [i for i in range(1, len(pred)) if groups[i] != groups[i - 1]]
    ends = starts[1:] + [len(pred)]
    for s, e in zip(starts, ends):
        seg = pred[s:e]
        for value, minimum in ((0, min_off), (1, min_on)):
            if minimum <= 1:
                continue
            i = 0
            while i < len(seg):
                if seg[i] != value:
                    i += 1
                    continue
                j = i
                while j < len(seg) and seg[j] == value:
                    j += 1
                # Interior runs only: a run touching a session edge may simply
                # be cut off by the recording, not genuinely short
                if j - i < minimum and not (i == 0 and value == 1) and not (j == len(seg) and value == 1):
                    if value == 0 and (i == 0 or j == len(seg)):
                        pass          # leading/trailing silence is real
                    else:
                        seg[i:j] = 1 - value
                i = j
        pred[s:e] = seg
    return pred


def smooth(probs: np.ndarray, groups: np.ndarray, threshold: float,
           params: dict | None = None) -> np.ndarray:
    """Full post-processing chain: hysteresis, then minimum durations.

    `low`/`high` default to a band around the fitted single threshold, so
    smoothing never silently moves the operating point — it only refuses to
    flip inside the band.
    """
    p = {**DEFAULTS, **(params or {})}
    band = float(p.get("band", 0.1))
    high = p["high"] if p["high"] is not None else min(0.99, threshold + band / 2)
    low = p["low"] if p["low"] is not None else max(0.01, threshold - band / 2)
    if low > high:
        low, high = high, low
    pred = apply_hysteresis(probs, groups, low, high)
    return enforce_durations(pred, groups, int(p["min_on"]), int(p["min_off"]))
