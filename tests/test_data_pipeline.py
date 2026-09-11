"""Loader, resampling and preprocessing on synthetic schema-v2 exports."""

import json

import numpy as np
import pandas as pd
import pytest

from export_onnx import feature_order
from src.data.loaders import LANDMARK_COLUMNS, WORLD_COLUMNS, load_dataset_folder
from src.data.preprocess import make_windows, preprocess
from src.data.timing import resample_session

BASE_PREP = {"aspect_correct": True, "time_normalize": True, "causal_motion": True,
             "add_motion": True, "pose_features": True, "normalize": "wrist",
             "add_velocity": False, "heuristic_smooth_frames": 5,
             "pose_source": "image", "mirror_left_hands": True,
             "require_resolution": True}


def _hand(n, rng, shift=0.0):
    """A plausible moving right hand: 21 points around a wandering wrist."""
    wrist = 0.4 + 0.1 * np.sin(np.linspace(0, 6, n))[:, None] + shift
    offsets = rng.normal(0, 0.05, (21, 2))
    offsets[0] = 0
    xy = wrist[:, None, :] + offsets[None, :, :]
    z = rng.normal(0, 0.01, (n, 21, 1))
    return np.concatenate([np.broadcast_to(xy, (n, 21, 2)), z], axis=2)


def _write_session(root, p, s, n, fps, labels, meta_video=None, handedness="Right",
                   world=True, seed=0):
    rng = np.random.default_rng(seed)
    d = root / p / s
    d.mkdir(parents=True)
    pts = _hand(n, rng)
    df = pd.DataFrame(pts.reshape(n, 63), columns=LANDMARK_COLUMNS)
    df.insert(0, "frame_index", np.arange(n))
    df.insert(1, "timestamp_ms", (np.arange(n) * 1000.0 / fps).astype(int))
    df.insert(2, "hand_detected", True)
    df.insert(3, "detection_confidence", 0.9)
    df.insert(4, "tracking_confidence", 0.9)
    df["handedness"] = handedness
    if world:
        w = rng.normal(0, 0.03, (n, 63))
        w[:, :3] = 0
        df[WORLD_COLUMNS] = w
    df.to_csv(d / "landmarks.csv", index=False)
    pd.DataFrame({"frame_index": np.arange(n), "label": labels}).to_csv(
        d / "labels.csv", index=False)
    meta = {"schema_version": 2}
    if meta_video:
        meta["video"] = meta_video
    (d / "metadata.json").write_text(json.dumps(meta))


def _labels(n):
    lab = np.array(["not_writing"] * n, dtype=object)
    lab[n // 5: n // 2] = "writing"
    lab[n // 2 + 5: 4 * n // 5] = "writing"
    lab[n // 2 + 1: n // 2 + 3] = "unsure"
    return lab


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "dataset_Test"
    _write_session(root, "P001", "S001", 300, 30, _labels(300),
                   {"width": 640, "height": 480})
    _write_session(root, "P002", "S002", 600, 60, _labels(600),
                   {"width": 1080, "height": 1920}, handedness="Left", seed=1)
    _write_session(root, "P003", "S003", 90, 30, ["maybe"] * 90,
                   {"width": 640, "height": 480}, seed=2)
    return root


def test_loader_reads_v2_metadata_resamples_and_flags_unsure(corpus):
    ds = load_dataset_folder(corpus, "dataset_Test", target_fps=30)
    f = ds.frames
    assert set(f["group"]) == {"P001/S001", "P002/S002"}       # "maybe" session skipped
    assert any("unknown labels" in s for s in ds.meta["sessions_skipped"])
    assert f.loc[f["group"] == "P002/S002", "aspect"].iloc[0] == pytest.approx(1080 / 1920)
    assert ds.meta["aspect_source"] == {"metadata.json": 2}
    # the 60 fps session now has ~half the rows, one every 33 ms
    s2 = f[f["group"] == "P002/S002"]
    assert 295 <= len(s2) <= 305
    assert np.median(np.diff(s2["timestamp_ms"])) in (33, 34)
    assert ds.meta["timing"]["sessions_resampled"] == 1
    # unsure is kept on the timeline but never scored
    assert (~f["scored"]).sum() == (f["label"] == "unsure").sum() > 0
    g = ds.meta["label_granularity"]
    assert g["interior_pauses"] == 0            # the only gap touches an unsure run
    assert g["unsure_frames"] > 0


def test_resampling_is_identity_near_target_and_causal_otherwise():
    n = 100
    df = pd.DataFrame({"frame_index": np.arange(n),
                       "timestamp_ms": (np.arange(n) * 1000 / 29.97).astype(int),
                       "v": np.arange(n)})
    out, note = resample_session(df, 30)
    assert not note["resampled"] and len(out) == n
    df25 = df.assign(timestamp_ms=(np.arange(n) * 40))      # 25 fps -> 30
    out, note = resample_session(df25, 30)
    assert note["resampled"] and note["held_repeats"] > 0
    # sample-and-hold never uses a frame from the future
    src_ts = df25.set_index("frame_index")["timestamp_ms"]
    assert (src_ts.loc[out["source_frame_index"]].to_numpy()
            <= out["timestamp_ms"].to_numpy() + 1).all()


def test_missing_resolution_stops_preprocessing(tmp_path):
    root = tmp_path / "dataset_NoRes"
    _write_session(root, "P001", "S001", 120, 30, _labels(120), meta_video=None)
    ds = load_dataset_folder(root, "dataset_NoRes")
    with pytest.raises(ValueError, match="no recorded resolution"):
        preprocess(ds.frames, ds.feature_columns, BASE_PREP)
    frames, _, notes = preprocess(ds.frames, ds.feature_columns,
                                  {**BASE_PREP, "require_resolution": False})
    assert notes["aspect_sessions_uncorrected"] == ["P001/S001"]


@pytest.mark.parametrize("overrides", [
    {}, {"pose_source": "world"}, {"add_motion": False}, {"pose_features": False},
    {"add_velocity": True},
])
def test_exported_feature_order_matches_preprocessing(corpus, overrides):
    """The inference spec's feature order must be exactly what training used;
    a mismatch feeds the app correctly-shaped nonsense with no error."""
    ds = load_dataset_folder(corpus, "dataset_Test", target_fps=30)
    prep = {**BASE_PREP, **overrides}
    _, columns, _ = preprocess(ds.frames, ds.feature_columns, prep)
    assert columns == feature_order(prep)


def test_left_hand_pose_is_mirrored_into_right_hand_form(corpus):
    ds = load_dataset_folder(corpus, "dataset_Test", target_fps=30)
    frames = ds.frames.copy()
    left = frames["handedness"] == "Left"
    on, _, notes = preprocess(frames, ds.feature_columns, BASE_PREP)
    off, _, _ = preprocess(frames, ds.feature_columns,
                           {**BASE_PREP, "mirror_left_hands": False})
    xs = [c for c in LANDMARK_COLUMNS if c.endswith("_x")]
    np.testing.assert_allclose(on.loc[left, xs], -off.loc[left, xs])
    np.testing.assert_allclose(on.loc[~left, xs], off.loc[~left, xs])
    # speeds do not change under a mirror and must not be touched
    mot = [c for c in on.columns if c.startswith("mot_")]
    np.testing.assert_allclose(on[mot], off[mot])
    assert "mirrored" in notes["mirror_left_hands"]


def test_windows_label_by_last_frame_and_skip_unsure_targets():
    n = 12
    frames = pd.DataFrame({
        "group": ["a"] * n,
        "f": np.arange(n, dtype=float),
        "label": ["writing"] * 8 + ["not_writing"] + ["unsure"] + ["writing"] * 2,
    })
    frames["scored"] = frames["label"] != "unsure"
    X, y, _, ends = make_windows(frames, ["f"], size=4, stride=1, label_mode="last")
    by_end = dict(zip(ends.tolist(), y.tolist()))
    assert by_end[8] == 0                       # the pause frame is the target
    assert 9 not in by_end                      # unsure target: no window
    X, y, _, ends = make_windows(frames, ["f"], size=4, stride=1, label_mode="majority")
    assert dict(zip(ends.tolist(), y.tolist()))[8] == 1   # majority hides it
