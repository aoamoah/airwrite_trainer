"""Timeline metrics the thesis is compared on.

Two families, each chosen because a specific body of published work reports
it, so the numbers can be set beside theirs rather than merely near them.

**Pause events** — the protocol of "Detecting Pen-In-Air States from Video"
(arXiv 2606.02342), the closest published task. Its positive class is the
*pause* (pen-up), not the writing: an event is a pause bounded by writing on
both sides, and a prediction matches it when both its boundaries fall within
a temporal tolerance of the true ones. It reports F2 (recall-weighted) at 5,
10 and 12 frames (~167/333/400 ms at 30 fps; 12 frames is the typical
handwriting pause it cites). This is the metric for "detects writing state
even mid-letter": a model that only finds whole writing episodes scores
nothing here.

**Segmental metrics** — the temporal action segmentation standard (Lea et
al. 2017; MS-TCN, Farha & Gall 2019; online: OnlineTAS, NeurIPS 2024). Edit
score is the normalised Levenshtein similarity between the sequences of
segment labels, and punishes over-segmentation directly. F1@k counts a
predicted segment correct when its IoU with an unmatched true segment of the
same label reaches k. Online models characteristically keep frame accuracy
while these collapse — OnlineTAS reports 56.7% frame accuracy at 9.3% F1@50
on Breakfast — which is exactly the live app's "cutting out" symptom. The
implementation follows the MS-TCN evaluation code: segment ends exclusive,
F1 pooled over sessions, Edit averaged over sessions. Both classes count as
segments (there is no background class in a binary task).

Frames labelled `unsure` are removed and the timeline is cut at them, so no
segment or pause is ever formed across an unjudged stretch.
"""

import numpy as np

DEFAULT_PAUSE_TOLERANCES = (5, 10, 12)
DEFAULT_SEGMENT_OVERLAPS = (0.10, 0.25, 0.50)
DEFAULT_MIN_PAUSE_FRAMES = 3


def split_unscored(y_true, y_pred, groups, scored=None):
    """Drop unscored frames; start a new sub-session after each unscored run.

    Returns (y_true, y_pred, groups) restricted to scored frames, with group
    ids suffixed `#k` so runs on either side of an unsure stretch are never
    joined."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    groups = np.asarray(groups, dtype=object)
    if scored is None:
        return y_true, y_pred, groups
    scored = np.asarray(scored, dtype=bool)
    if scored.all():
        return y_true, y_pred, groups
    sub = np.empty(len(groups), dtype=object)
    k, prev_group, prev_scored = 0, None, True
    for i, (g, s) in enumerate(zip(groups, scored)):
        if g != prev_group:
            k = 0
        elif s and not prev_scored:
            k += 1
        sub[i] = f"{g}#{k}"
        prev_group, prev_scored = g, s
    return y_true[scored], y_pred[scored], sub[scored]


def _group_bounds(groups: np.ndarray):
    if len(groups) == 0:
        return []
    change = np.flatnonzero(groups[1:] != groups[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(groups)]
    return list(zip(starts, ends))


def _segments(y: np.ndarray):
    """(label, start, end_exclusive) runs of one session."""
    if len(y) == 0:
        return []
    change = np.flatnonzero(y[1:] != y[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(y)]
    return [(int(y[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def _interior_pauses(y: np.ndarray, min_frames: int):
    """(start, end_exclusive) of 0-runs with a 1-run on both sides."""
    segs = _segments(y)
    return [(s, e) for k, (v, s, e) in enumerate(segs)
            if v == 0 and 0 < k < len(segs) - 1 and e - s >= min_frames]


def _prf(tp: int, n_true: int, n_pred: int) -> dict:
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_true if n_true else 0.0

    def fbeta(beta):
        b2 = beta * beta
        denom = b2 * precision + recall
        return (1 + b2) * precision * recall / denom if denom else 0.0

    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(fbeta(1), 4), "f2": round(fbeta(2), 4)}


def pause_event_metrics(y_true, y_pred, groups,
                        tolerances=DEFAULT_PAUSE_TOLERANCES,
                        min_frames: int = DEFAULT_MIN_PAUSE_FRAMES) -> dict:
    """Pause detection under boundary tolerance, one-to-one matching.

    y is 1 = writing. A true pause is matched by at most one predicted pause
    whose start and end both lie within `tol` frames of its own; closest
    pairs are matched first."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    groups = np.asarray(groups, dtype=object)
    true_all, pred_all = [], []
    for a, b in _group_bounds(groups):
        true_all.append(_interior_pauses(y_true[a:b], min_frames))
        pred_all.append(_interior_pauses(y_pred[a:b], min_frames))
    n_true = sum(len(t) for t in true_all)
    n_pred = sum(len(p) for p in pred_all)

    out = {"n_true_pauses": n_true, "n_pred_pauses": n_pred,
           "min_pause_frames": min_frames}
    lengths = [e - s for t in true_all for s, e in t]
    out["median_true_pause_frames"] = int(np.median(lengths)) if lengths else None
    for tol in tolerances:
        tp = 0
        for true, pred in zip(true_all, pred_all):
            pairs = sorted(
                (abs(ts - ps) + abs(te - pe), i, j)
                for i, (ts, te) in enumerate(true) for j, (ps, pe) in enumerate(pred)
                if abs(ts - ps) <= tol and abs(te - pe) <= tol)
            used_t, used_p = set(), set()
            for _, i, j in pairs:
                if i in used_t or j in used_p:
                    continue
                used_t.add(i)
                used_p.add(j)
            tp += len(used_t)
        out[f"pause@{tol}f"] = _prf(tp, n_true, n_pred)
    return out


def _levenshtein(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def segmental_metrics(y_true, y_pred, groups,
                      overlaps=DEFAULT_SEGMENT_OVERLAPS) -> dict:
    """Edit score and segmental F1@k, MS-TCN style. Values are fractions."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    groups = np.asarray(groups, dtype=object)
    tp = {k: 0 for k in overlaps}
    fp = {k: 0 for k in overlaps}
    fn = {k: 0 for k in overlaps}
    edits = []
    for a, b in _group_bounds(groups):
        gt = _segments(y_true[a:b])
        pr = _segments(y_pred[a:b])
        if not gt:
            continue
        edits.append(1.0 - _levenshtein([s[0] for s in pr], [s[0] for s in gt])
                     / max(len(pr), len(gt)))
        g_lab = np.array([s[0] for s in gt])
        g_start = np.array([s[1] for s in gt])
        g_end = np.array([s[2] for s in gt])
        for k in overlaps:
            hits = np.zeros(len(gt), dtype=bool)
            for lab, ps, pe in pr:
                inter = np.minimum(pe, g_end) - np.maximum(ps, g_start)
                union = np.maximum(pe, g_end) - np.minimum(ps, g_start)
                iou = np.where(g_lab == lab, np.clip(inter, 0, None) / union, 0.0)
                idx = int(np.argmax(iou))
                if iou[idx] >= k and not hits[idx]:
                    tp[k] += 1
                    hits[idx] = True
                else:
                    fp[k] += 1
            fn[k] += int(len(gt) - hits.sum())

    out = {"edit_score": round(float(np.mean(edits)), 4) if edits else None}
    for k in overlaps:
        p = tp[k] / (tp[k] + fp[k]) if tp[k] + fp[k] else 0.0
        r = tp[k] / (tp[k] + fn[k]) if tp[k] + fn[k] else 0.0
        out[f"seg_f1@{int(round(k * 100))}"] = round(2 * p * r / (p + r), 4) if p + r else 0.0
    return out
