"""Event-level scoring: writing *episodes*, not individual frames.

Frame F1 answers "what share of frames were labelled right", which is not
what the live app needs. The app needs to notice that a writing episode
started, and to stop when it ends. A detector can score a respectable frame
F1 while shattering every episode into a dozen flickering fragments, and a
user experiences that as broken. It can also lag every onset by half a second
and lose almost nothing at frame level.

So episodes are matched one-to-one here, and the fragmentation and onset lag
are reported alongside precision and recall. This is also the granularity the
closest published work uses, which makes the numbers comparable rather than
merely adjacent.

Two matching criteria are reported because they answer different questions:

- **IoU** — did the predicted episode cover the right span? Strict about
  duration; a prediction twice as long as the truth fails it.
- **Onset tolerance** — did the detector fire near the right moment? This is
  the interaction-facing question, and the criterion the pen-in-air video
  work reports (they use ~12 frames, about 400 ms).
"""

from dataclasses import dataclass

import numpy as np

DEFAULT_IOU = 0.5
DEFAULT_TOLERANCE_FRAMES = 12     # ~400 ms at 30 fps
DEFAULT_MIN_EVENT_FRAMES = 3      # shorter runs are annotation noise


@dataclass(frozen=True)
class Event:
    group: str
    start: int          # inclusive row position
    end: int            # inclusive row position

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def find_events(y: np.ndarray, groups: np.ndarray,
                min_frames: int = DEFAULT_MIN_EVENT_FRAMES) -> list[Event]:
    """Contiguous runs of 1 within each group, in row order.

    Rows must already be ordered by session and frame; a run never spans a
    group boundary.
    """
    y = np.asarray(y).astype(int)
    groups = np.asarray(groups, dtype=object)
    if len(y) == 0:
        return []

    events, start = [], None
    for i in range(len(y)):
        boundary = i == 0 or groups[i] != groups[i - 1]
        if boundary and start is not None:
            events.append(Event(groups[i - 1], start, i - 1))
            start = None
        if y[i] == 1 and start is None:
            start = i
        elif y[i] == 0 and start is not None:
            events.append(Event(groups[i], start, i - 1))
            start = None
    if start is not None:
        events.append(Event(groups[-1], start, len(y) - 1))

    return [e for e in events if e.length >= min_frames]


def _iou(a: Event, b: Event) -> float:
    if a.group != b.group:
        return 0.0
    inter = min(a.end, b.end) - max(a.start, b.start) + 1
    if inter <= 0:
        return 0.0
    union = max(a.end, b.end) - min(a.start, b.start) + 1
    return inter / union


def _match_iou(true: list[Event], pred: list[Event],
               iou_threshold: float) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching, best overlap first.

    One-to-one matters: without it a single prediction spanning a whole
    session would "detect" every episode in it.
    """
    pairs = [(i, j, _iou(t, p))
             for i, t in enumerate(true) for j, p in enumerate(pred)
             if _iou(t, p) >= iou_threshold]
    pairs.sort(key=lambda x: -x[2])
    used_t, used_p, matched = set(), set(), []
    for i, j, score in pairs:
        if i in used_t or j in used_p:
            continue
        used_t.add(i)
        used_p.add(j)
        matched.append((i, j, score))
    return matched


def _match_onset(true: list[Event], pred: list[Event],
                 tolerance: int) -> list[tuple[int, int, int]]:
    """Greedy one-to-one matching on onset proximity, closest first."""
    pairs = [(i, j, abs(t.start - p.start))
             for i, t in enumerate(true) for j, p in enumerate(pred)
             if t.group == p.group and abs(t.start - p.start) <= tolerance]
    pairs.sort(key=lambda x: x[2])
    used_t, used_p, matched = set(), set(), []
    for i, j, delta in pairs:
        if i in used_t or j in used_p:
            continue
        used_t.add(i)
        used_p.add(j)
        matched.append((i, j, delta))
    return matched


def _prf(n_matched: int, n_true: int, n_pred: int) -> dict:
    precision = n_matched / n_pred if n_pred else 0.0
    recall = n_matched / n_true if n_true else 0.0
    def fbeta(beta):
        b2 = beta * beta
        denom = b2 * precision + recall
        return (1 + b2) * precision * recall / denom if denom else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(fbeta(1), 4), "f2": round(fbeta(2), 4)}


def event_metrics(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray,
                  iou_threshold: float = DEFAULT_IOU,
                  tolerance_frames: int = DEFAULT_TOLERANCE_FRAMES,
                  min_event_frames: int = DEFAULT_MIN_EVENT_FRAMES) -> dict:
    """Episode-level precision/recall/F1/F2 under both matching criteria."""
    true = find_events(y_true, groups, min_event_frames)
    pred = find_events(y_pred, groups, min_event_frames)

    iou_matched = _match_iou(true, pred, iou_threshold)
    onset_matched = _match_onset(true, pred, tolerance_frames)

    # Fragmentation: predicted episodes overlapping each true episode at all.
    # A detector that chops one stroke into six fragments scores badly here
    # even when its frame F1 looks healthy.
    overlaps = [sum(1 for p in pred if _iou(t, p) > 0) for t in true]

    onset_errors = [d for _, _, d in onset_matched]
    # Signed: positive means the detector fired late. The continuous-gesture
    # survey (Emporio et al., CVIU 2025) lists detection delay among the four
    # metrics every online benchmark should report
    onset_delays = [pred[j].start - true[i].start for i, j, _ in onset_matched]
    return {
        "n_true_events": len(true),
        "n_pred_events": len(pred),
        "min_event_frames": min_event_frames,
        f"iou@{iou_threshold}": _prf(len(iou_matched), len(true), len(pred)),
        f"onset@{tolerance_frames}f": _prf(len(onset_matched), len(true), len(pred)),
        "median_iou_of_matches": (round(float(np.median([s for _, _, s in iou_matched])), 4)
                                  if iou_matched else None),
        "median_onset_error_frames": (int(np.median(onset_errors))
                                      if onset_errors else None),
        "median_onset_delay_frames": (float(np.median(onset_delays))
                                      if onset_delays else None),
        "mean_fragments_per_true_event": (round(float(np.mean(overlaps)), 2)
                                          if overlaps else None),
        "true_events_missed_entirely": int(sum(1 for n in overlaps if n == 0)),
        "median_true_event_frames": (int(np.median([e.length for e in true]))
                                     if true else None),
    }
