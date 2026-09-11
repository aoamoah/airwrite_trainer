import numpy as np
import pytest

from src.training.segmental import (
    pause_event_metrics, segmental_metrics, split_unscored,
)


def g(n, name="s1"):
    return np.array([name] * n, dtype=object)


def test_perfect_prediction_scores_one_everywhere():
    y = np.array([0] * 5 + [1] * 10 + [0] * 4 + [1] * 10 + [0] * 5)
    pause = pause_event_metrics(y, y, g(len(y)))
    assert pause["n_true_pauses"] == 1
    for tol in (5, 10, 12):
        assert pause[f"pause@{tol}f"]["f2"] == 1.0
    seg = segmental_metrics(y, y, g(len(y)))
    assert seg["edit_score"] == 1.0
    assert seg["seg_f1@50"] == 1.0


def test_only_interior_pauses_count():
    # leading/trailing not_writing is rest, not a pause
    y = np.array([0] * 10 + [1] * 10 + [0] * 10)
    assert pause_event_metrics(y, y, g(len(y)))["n_true_pauses"] == 0


def test_pause_boundaries_must_both_fall_within_tolerance():
    truth = np.array([1] * 20 + [0] * 10 + [1] * 20)       # pause 20..29
    late = np.array([1] * 26 + [0] * 10 + [1] * 14)        # pause 26..35: both ends off by 6
    m = pause_event_metrics(truth, late, g(50), tolerances=(5, 10))
    assert m["pause@5f"]["recall"] == 0.0
    assert m["pause@10f"]["recall"] == 1.0


def test_missed_pause_inside_long_writing_prediction():
    # a detector that only finds whole writing episodes finds no pauses
    truth = np.array([0] * 5 + [1] * 10 + [0] * 4 + [1] * 10 + [0] * 5)
    episode_only = np.array([0] * 5 + [1] * 24 + [0] * 5)
    m = pause_event_metrics(truth, episode_only, g(len(truth)))
    assert m["pause@12f"]["recall"] == 0.0 and m["n_pred_pauses"] == 0


def test_short_pauses_below_minimum_are_not_events():
    y = np.array([1] * 10 + [0] * 2 + [1] * 10)
    assert pause_event_metrics(y, y, g(len(y)), min_frames=3)["n_true_pauses"] == 0


def test_pauses_never_span_sessions():
    y = np.array([1] * 10 + [0] * 5 + [1] * 10)
    groups = np.array(["a"] * 12 + ["b"] * 13, dtype=object)
    # the 0-run is cut by the session boundary: no pause bounded by writing
    assert pause_event_metrics(y, y, groups)["n_true_pauses"] == 0


def test_segmental_matches_hand_computation():
    # truth: 0x10 1x10 ; prediction flickers once inside the writing run
    truth = np.array([0] * 10 + [1] * 10)
    pred = np.array([0] * 10 + [1] * 4 + [0] * 2 + [1] * 4)
    m = segmental_metrics(truth, pred, g(20), overlaps=(0.10, 0.50))
    # segments: truth [0,1]; pred [0,1,0,1] -> Levenshtein 2, max len 4
    assert m["edit_score"] == pytest.approx(0.5)
    # pred segs: 0[0,10) iou 1 -> tp; 1[10,14) iou .4 -> tp at .1, fp at .5;
    # 0[14,16) iou with 0-truth = 0 -> fp; 1[16,20) truth 1 already hit -> fp
    # @0.10: tp 2, fp 2, fn 0 -> P .5 R 1 -> F1 .6667
    assert m["seg_f1@10"] == pytest.approx(0.6667, abs=1e-4)
    # @0.50: tp 1, fp 3, fn 1 -> P .25 R .5 -> F1 .3333
    assert m["seg_f1@50"] == pytest.approx(0.3333, abs=1e-4)


def test_over_segmentation_lowers_edit_but_not_accuracy_much():
    truth = np.array([1] * 100)
    pred = truth.copy()
    pred[[20, 40, 60, 80]] = 0                    # four one-frame dropouts
    assert (pred == truth).mean() == 0.96
    assert segmental_metrics(truth, pred, g(100))["edit_score"] < 0.2


def test_split_unscored_cuts_the_timeline():
    y_true = np.array([1, 1, 0, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 0, 1, 1])
    groups = g(6)
    scored = np.array([True, True, False, False, True, True])
    yt, yp, gg = split_unscored(y_true, y_pred, groups, scored)
    assert list(yt) == [1, 1, 1, 1]
    assert len(set(gg)) == 2            # the two writing runs stay separate
    # ...so the unsure stretch cannot be scored as a pause
    assert pause_event_metrics(yt, yp, gg)["n_true_pauses"] == 0
