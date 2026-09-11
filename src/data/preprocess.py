"""Feature preprocessing and sequence windowing.

Three corrections happen here before anything is handed to a model, in this
order, because each one depends on the last:

1. **Aspect correction.** MediaPipe writes x as a fraction of frame width and
   y as a fraction of frame height, so on any non-square video the two axes
   carry different physical scales. Wrist normalisation cannot undo that — it
   divides both axes by one isotropic hand span. x is rescaled into units of
   image height using each session's resolution, which the collector records
   in metadata.json (older exports: `scan_resolutions.py`). A session whose
   resolution is unknown stops the run rather than being silently left
   uncorrected — with many sources, that is how a portrait phone clip would
   end up on a different scale from everything else.

2. **Motion features.** Air-writing is mostly whole-hand translation, which
   wrist normalisation deletes by construction. Speeds are therefore taken
   from the aspect-corrected but *un*-normalised landmarks, and expressed in
   hand-widths per second — divided by the wrist -> middle-MCP distance so
   they do not depend on how close the hand is to the camera, and by the real
   `dt` from `timestamp_ms` so they do not depend on frame rate. A 30-frame
   window is 1.0 s at 30 fps and 0.5 s at 60 fps; three of the sessions are
   60 fps, and frame-indexed differences put them on a different scale from
   everything else.

3. **Wrist normalisation** of the pose itself, unchanged and still optional,
   so "wrist-normalised pose only" remains a runnable ablation against the
   pose + motion arm.

4. **Canonical hand.** A left hand's pose is the mirror image of a right
   hand's, and so is any hand in a mirrored (front-camera) source. With
   `mirror_left_hands`, poses labelled Left by MediaPipe are mirrored into
   right-hand form. Only the wrist-relative pose is mirrored, never the
   positions the motion features are taken from: speed does not change under
   a mirror, and mirroring absolute positions would turn any flicker in the
   handedness label into a jump of half the frame width.

The pose can come from the image landmarks (default) or from MediaPipe's
world landmarks (`pose_source: world`), which are metric and hand-centred and
so independent of resolution and aspect by construction. Motion features
always use image landmarks, because world landmarks carry no position.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

_LM_COL = re.compile(r"^l(\d+)_([xyz])$")
_WORLD_COL = re.compile(r"^wl(\d+)_([xyz])$")
WORLD_COLUMNS = [f"wl{i}_{axis}" for i in range(21) for axis in ("x", "y", "z")]

# Rolling windows for the motion features, in frames. Short enough to catch a
# single stroke, long enough to survive a dropped detection.
MOTION_MEAN_WINDOWS = (5, 15, 31)
MOTION_SD_WINDOWS = (15, 31)

# dt outside this range is a timestamp glitch, not a real gap. Dividing by a
# 0 ms or 3 s dt produces infinities that poison every downstream statistic.
DT_MIN_MS, DT_MAX_MS = 5.0, 200.0


def _landmark_triples(feature_columns: list[str],
                      pattern: re.Pattern = _LM_COL) -> list[tuple[str, str, str]]:
    """Group MediaPipe landmark columns into (x, y, z) triples, in order."""
    by_index: dict[int, dict[str, str]] = {}
    for col in feature_columns:
        m = pattern.match(col)
        if m:
            by_index.setdefault(int(m.group(1)), {})[m.group(2)] = col
    return [
        (axes["x"], axes["y"], axes["z"])
        for _, axes in sorted(by_index.items())
        if len(axes) == 3
    ]


HEURISTIC_COLUMNS = ("heur_speed", "heur_extension")


def load_resolutions(path: str | Path | None = None) -> dict:
    """Per-session {width, height, aspect} written by scan_resolutions.py."""
    p = Path(path) if path else Path(__file__).parents[2] / "session_resolutions.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text()).get("sessions", {})


def _session_aspect(frames: pd.DataFrame, resolutions: dict) -> pd.Series:
    """Aspect ratio (width / height) per row, 1.0 where unknown.

    Sessions are keyed `<source>/P###/S###`; `group` is already `P###/S###`.
    """
    if not resolutions:
        return pd.Series(1.0, index=frames.index)
    source = (frames["source"] if "source" in frames.columns
              else pd.Series("", index=frames.index))
    keys = source.astype(str) + "/" + frames["group"].astype(str)
    aspect = keys.map(lambda k: (resolutions.get(k) or {}).get("aspect"))
    if aspect.isna().all():  # combined runs namespace the group; try the tail
        keys = frames["group"].astype(str).str.split(":").str[-1]
        aspect = keys.map(
            lambda g: next((v["aspect"] for k, v in resolutions.items()
                            if k.endswith("/" + g)), None))
    return aspect.astype(float).fillna(1.0)


def apply_aspect_correction(frames: pd.DataFrame,
                            triples: list[tuple[str, str, str]],
                            resolutions: dict,
                            require: bool = True) -> tuple[pd.DataFrame, dict]:
    """Rescale x into units of image height, so x and y share a scale.

    The per-session `aspect` column set by the loader wins; `resolutions`
    (session_resolutions.json) fills only sessions it leaves unknown. A
    session still unknown after both stops the run when `require` is set —
    otherwise it is left uncorrected and named in the notes."""
    aspect = (frames["aspect"].astype(float) if "aspect" in frames.columns
              else pd.Series(np.nan, index=frames.index))
    if aspect.isna().any() and resolutions:
        legacy = _session_aspect(frames, resolutions)
        aspect = aspect.fillna(legacy.where(legacy != 1.0))
    missing = sorted(frames.loc[aspect.isna(), "group"].unique().tolist())
    if missing and require:
        raise ValueError(
            f"{len(missing)} session(s) have no recorded resolution, so x "
            f"cannot be aspect-corrected: {missing[:10]}. Re-export them from "
            "the collector (metadata.json schema_version 2 records it), run "
            "scan_resolutions.py, or set preprocessing.require_resolution: "
            "false to leave them uncorrected.")
    aspect = aspect.fillna(1.0)

    x_cols = [t[0] for t in triples]
    detected = _detected_mask(frames)
    scale = aspect.to_numpy()[:, None]
    xs = frames[x_cols].to_numpy()
    # Undetected rows are all-zero sentinels, not coordinates — leave them
    xs = np.where(detected[:, None], xs * scale, xs)
    frames[x_cols] = xs
    frames["aspect"] = aspect

    per_session = (frames.groupby("group", sort=False)["aspect"]
                   .first().round(3).value_counts().to_dict())
    notes = {
        "aspect_correction": "x rescaled by width/height into image-height units",
        "aspect_values_seen": {str(k): int(v) for k, v in sorted(per_session.items())},
    }
    if missing:
        notes["aspect_sessions_uncorrected"] = missing[:20]
    return frames, notes


def _detected_mask(frames: pd.DataFrame) -> np.ndarray:
    if "hand_detected" not in frames.columns:
        return np.ones(len(frames), bool)
    return frames["hand_detected"].to_numpy().astype(bool)


def _seconds_per_frame(frames: pd.DataFrame, time_normalize: bool) -> tuple[np.ndarray, dict]:
    """dt in seconds between consecutive frames of a session.

    Falls back to the median dt of the session wherever `timestamp_ms` is
    missing, non-monotonic, or outside a plausible range.
    """
    if not time_normalize or "timestamp_ms" not in frames.columns:
        return np.ones(len(frames)), {
            "time_normalization": "off — speeds are per frame, not per second"}

    ts = pd.to_numeric(frames["timestamp_ms"], errors="coerce")
    dt = ts.groupby(frames["group"].values, sort=False).diff()
    bad = dt.isna() | (dt < DT_MIN_MS) | (dt > DT_MAX_MS)
    median = (dt.where(~bad).groupby(frames["group"].values, sort=False)
                .transform("median"))
    dt = dt.where(~bad, median).fillna(1000.0 / 30.0)
    dt = dt.clip(DT_MIN_MS, DT_MAX_MS) / 1000.0

    fps = (1.0 / dt).groupby(frames["group"].values, sort=False).median().round(1)
    return dt.to_numpy(), {
        "time_normalization": "speeds divided by real dt from timestamp_ms",
        "dt_repaired_frames": int(bad.sum()),
        "median_fps_per_session": (fps.value_counts().sort_index()
                                   .rename_axis("fps").to_dict()),
    }


def _hand_scale(frames: pd.DataFrame, triples, detected: np.ndarray) -> np.ndarray:
    """Wrist -> middle-finger-MCP distance: one hand width, in frame units.

    Every speed is divided by this, which is what makes the features
    independent of how close the hand is to the camera.
    """
    xy = lambda i: frames[[triples[i][0], triples[i][1]]].to_numpy(dtype=np.float64)
    scale = np.linalg.norm(xy(9) - xy(0), axis=1)
    valid = detected & (scale > 1e-6)
    if valid.any():                       # a plausible value for bad rows
        scale = np.where(valid, scale, np.median(scale[valid]))
    else:
        scale = np.ones_like(scale)
    return scale


def _point_speed(frames, triples, index, scale, dt, detected) -> pd.Series:
    """Speed of one landmark in hand-widths per second, within a session."""
    xy = frames[[triples[index][0], triples[index][1]]].to_numpy(dtype=np.float64)
    pts = pd.DataFrame(xy / scale[:, None], index=frames.index, columns=["x", "y"])
    pts[~detected] = np.nan          # a gap breaks the difference, never spans it
    d = pts.groupby(frames["group"].values, sort=False).diff()
    step = np.sqrt(d["x"] ** 2 + d["y"] ** 2)
    return (step / dt).fillna(0.0)


def motion_features(frames: pd.DataFrame, triples, dt: np.ndarray,
                    causal: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """Fingertip and wrist speed, scale- and time-normalised, plus rolling
    mean and standard deviation at three time scales.

    The standard deviations matter as much as the means: sustained writing is
    not just motion, it is *variable* motion, while a hand travelling to or
    from the page moves fast and smoothly.
    """
    if len(triples) < 21:
        return pd.DataFrame(index=frames.index), []

    # Centred windows look forward, which a live app cannot do: at frame t
    # the trailing mean spans t-30..t, the centred one t-15..t+15. Measured
    # on real sessions the two differ by 25-43% on average, so a model
    # trained on centred windows is being fed something materially different
    # at inference time. `causal=True` trains on what the app can actually
    # compute, trading a little offline accuracy for the absence of a
    # train/serve skew.
    center = not causal
    detected = _detected_mask(frames)
    scale = _hand_scale(frames, triples, detected)
    groups = frames["group"].values

    out, names = {}, []
    for label, index in (("tip", 8), ("wrist", 0)):
        speed = _point_speed(frames, triples, index, scale, dt, detected)
        col = f"mot_{label}_speed"
        out[col] = speed.to_numpy()
        names.append(col)
        rolling = speed.groupby(groups, sort=False)
        for w in MOTION_MEAN_WINDOWS:
            col = f"mot_{label}_speed_m{w}"
            out[col] = rolling.transform(
                lambda s, w=w: s.rolling(w, min_periods=1, center=center).mean()
            ).to_numpy()
            names.append(col)
        for w in MOTION_SD_WINDOWS:
            col = f"mot_{label}_speed_sd{w}"
            out[col] = rolling.transform(
                lambda s, w=w: s.rolling(w, min_periods=2, center=center).std()
            ).fillna(0.0).to_numpy()
            names.append(col)

    return pd.DataFrame(out, index=frames.index), names


def heuristic_signals(frames: pd.DataFrame, triples: list[tuple[str, str, str]],
                      smooth_frames: int = 5,
                      dt: np.ndarray | None = None) -> pd.DataFrame:
    """Signals for the rule-based baselines, from *raw* landmarks.

    Computed before wrist normalisation: that step removes absolute position,
    and fingertip speed is exactly absolute position over time. Frames with
    no hand detected contribute zero to both signals.

    When `dt` is supplied the speed signal is per second rather than per
    frame, so the baselines are held to the same frame-rate correction as the
    learned models and the comparison between them stays honest.
    """
    if len(triples) < 21:
        return pd.DataFrame({c: np.zeros(len(frames)) for c in HEURISTIC_COLUMNS},
                            index=frames.index)

    xy = lambda i: frames[[triples[i][0], triples[i][1]]].to_numpy(dtype=np.float64)
    wrist, index_tip = xy(0), xy(8)
    detected = _detected_mask(frames)

    # --- speed: per-frame fingertip displacement, smoothed, within session ---
    tip = pd.DataFrame(index_tip, index=frames.index, columns=["x", "y"])
    tip[~detected] = np.nan          # gaps break the difference, they don't span it
    tip["group"] = frames["group"].values
    d = tip.groupby("group", sort=False)[["x", "y"]].diff()
    speed = np.sqrt((d["x"] ** 2 + d["y"] ** 2)).fillna(0.0)
    if dt is not None:
        speed = speed / dt
    speed = (speed.groupby(frames["group"].values, sort=False)
                  .transform(lambda s: s.rolling(smooth_frames, min_periods=1,
                                                 center=True).mean()))

    # --- extension: index tip further from wrist than the other tips ---
    palm = np.linalg.norm(xy(9) - wrist, axis=1)   # wrist -> middle-finger MCP
    valid = detected & (palm > 0)
    scale = np.where(valid, palm, 1.0)
    dist = lambda i: np.linalg.norm(xy(i) - wrist, axis=1) / scale
    others = np.mean(np.stack([dist(12), dist(16), dist(20)]), axis=0)
    extension = np.where(valid, dist(8) - others, 0.0)

    return pd.DataFrame({"heur_speed": speed.to_numpy(),
                         "heur_extension": extension}, index=frames.index)


def preprocess(frames: pd.DataFrame, feature_columns: list[str], cfg: dict) -> tuple[pd.DataFrame, list[str], dict]:
    """Apply configured preprocessing. Returns (frames, feature_columns, notes)."""
    frames = frames.copy()
    notes = {}

    pose_source = cfg.get("pose_source", "image")
    if pose_source not in ("image", "world"):
        raise ValueError(f"preprocessing.pose_source must be image or world, got {pose_source!r}")
    if pose_source == "world":
        if not set(WORLD_COLUMNS) <= set(frames.columns) or \
                frames.loc[frames["hand_detected"] == 1, WORLD_COLUMNS].isna().any().any():
            raise ValueError(
                "pose_source: world needs world landmarks (wl*_ columns) in every "
                "session — re-extract with the current collector")

    if cfg.get("drop_undetected") and "hand_detected" in frames.columns:
        before = len(frames)
        frames = frames[frames["hand_detected"] == 1].reset_index(drop=True)
        notes["dropped_undetected_frames"] = before - len(frames)

    triples = _landmark_triples(feature_columns)

    # 1. Aspect: put x and y on one scale before any distance is measured
    if cfg.get("aspect_correct", True) and triples:
        frames, aspect_notes = apply_aspect_correction(
            frames, triples, load_resolutions(cfg.get("resolutions_path")),
            require=bool(cfg.get("require_resolution", True)))
        notes.update(aspect_notes)
    else:
        notes["aspect_correction"] = "off"

    # 2. Real elapsed time per frame, shared by the motion features and the
    #    rule baselines so both are frame-rate independent or neither is
    time_normalize = cfg.get("time_normalize", True)
    dt, dt_notes = _seconds_per_frame(frames, time_normalize)
    notes.update(dt_notes)

    # Before normalisation — these are inputs to the rule-based baselines
    # only, never to the learned models' feature matrix
    signals = heuristic_signals(frames, triples,
                                int(cfg.get("heuristic_smooth_frames", 5)),
                                dt=dt if time_normalize else None)
    frames[list(HEURISTIC_COLUMNS)] = signals
    notes["heuristic_signals"] = (
        "fingertip speed + finger extension, from raw landmarks "
        f"({'hand-widths/s' if time_normalize else 'per frame'}, baselines only)")

    # 3. Motion features, also before wrist normalisation removes translation
    motion_names: list[str] = []
    if cfg.get("add_motion", True) and triples:
        causal = bool(cfg.get("causal_motion", False))
        motion, motion_names = motion_features(frames, triples, dt, causal=causal)
        frames = pd.concat([frames, motion], axis=1)
        notes["motion_features"] = (
            f"{len(motion_names)} features — fingertip and wrist speed in "
            "hand-widths per second, rolling mean over "
            f"{MOTION_MEAN_WINDOWS} and sd over {MOTION_SD_WINDOWS} frames, "
            + ("trailing windows (causal — matches what a live app can "
               "compute)" if causal else
               "centred windows (looks forward; a live app CANNOT reproduce "
               "these — see causal_motion)"))
    else:
        notes["motion_features"] = "off"

    pose_triples = (_landmark_triples(WORLD_COLUMNS, _WORLD_COL)
                    if pose_source == "world" else triples)
    if pose_source == "world":
        # Undetected rows are sentinels like the image columns: all zero
        frames[WORLD_COLUMNS] = frames[WORLD_COLUMNS].fillna(0.0)
        notes["pose_source"] = "world landmarks (metric, hand-centred)"
    else:
        notes["pose_source"] = "image landmarks"

    if cfg.get("normalize") == "wrist" and pose_triples:
        xs = frames[[t[0] for t in pose_triples]].to_numpy()
        ys = frames[[t[1] for t in pose_triples]].to_numpy()
        zs = frames[[t[2] for t in pose_triples]].to_numpy()

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

        frames[[t[0] for t in pose_triples]] = xs
        frames[[t[1] for t in pose_triples]] = ys
        frames[[t[2] for t in pose_triples]] = zs
        notes["normalization"] = "wrist-origin, hand-span scaled"
    else:
        notes["normalization"] = "none"

    if cfg.get("mirror_left_hands", True) and "handedness" in frames.columns and pose_triples:
        left = (frames["handedness"].astype(str) == "Left").to_numpy()
        if left.any():
            x_cols = [t[0] for t in pose_triples]
            xs = frames[x_cols].to_numpy()
            if cfg.get("normalize") == "wrist" or pose_source == "world":
                xs[left] = -xs[left]            # wrist-relative: mirror about the wrist
            else:
                # Absolute image coordinates in image-height units: mirror
                # about the frame's vertical centre line
                width = frames["aspect"].fillna(1.0).to_numpy() if "aspect" in frames else 1.0
                xs[left] = (np.broadcast_to(width, len(xs))[left, None] - xs[left])
            frames[x_cols] = xs
        notes["mirror_left_hands"] = f"{int(left.sum())} Left-labelled frames mirrored to right-hand form"
    elif cfg.get("mirror_left_hands", True):
        notes["mirror_left_hands"] = "no handedness column — nothing mirrored"

    if cfg.get("pose_features", True):
        pose_columns = (["hand_detected"] + WORLD_COLUMNS if pose_source == "world"
                        else list(feature_columns))
    else:
        # Motion-only ablation: keep the detection flag, drop the 63 pose
        # columns that wrist normalisation has already emptied of trajectory
        pose_columns = [c for c in feature_columns if not _LM_COL.match(c)]
        notes["pose_features"] = "off — motion features only"
    feature_columns = pose_columns + motion_names

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


WINDOW_LABEL_MODES = ("last", "majority")


def make_windows(
    frames: pd.DataFrame,
    feature_columns: list[str],
    size: int,
    stride: int,
    label_mode: str = "last",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding windows within each group (never across session bounds).

    Returns (X [n, size, features], y [n] with 1=writing, groups [n],
    end_index [n]). `end_index` is the row position of each window's last
    frame, which is what lets window predictions be laid back onto the frame
    timeline for event-level scoring.

    `label_mode` "last" (default) labels a window by its final frame: the
    question a live app asks is "is the user writing *now*", and the score is
    laid onto the timeline at the window's end. "majority" — the old
    behaviour, kept as an ablation — labels by the window's middle in effect,
    which trains the model to report a state half a window old and makes any
    pause shorter than half the window impossible to target.

    Windows whose target is an unscored (`unsure`) frame are left out. In
    majority mode only scored frames vote.
    """
    if label_mode not in WINDOW_LABEL_MODES:
        raise ValueError(f"label_mode must be one of {WINDOW_LABEL_MODES}")
    xs, ys, gs, ends = [], [], [], []
    positions = np.arange(len(frames))
    scored_all = (frames["scored"].to_numpy(dtype=bool) if "scored" in frames
                  else np.ones(len(frames), bool))
    for group, gdf in frames.groupby("group", sort=False):
        feats = gdf[feature_columns].to_numpy(dtype=np.float32)
        labels = (gdf["label"] == "writing").to_numpy(dtype=np.int8)
        in_group = frames["group"].to_numpy() == group
        rows = positions[in_group]
        scored = scored_all[in_group]
        for start in range(0, len(gdf) - size + 1, stride):
            last = start + size - 1
            if not scored[last]:
                continue
            if label_mode == "last":
                y = int(labels[last])
            else:
                votes = scored[start:last + 1]
                y = int(labels[start:last + 1][votes].sum() * 2 > votes.sum())
            xs.append(feats[start:start + size])
            ys.append(y)
            gs.append(group)
            ends.append(rows[last])
    if not xs:
        return (np.empty((0, size, len(feature_columns)), dtype=np.float32),
                np.empty(0, dtype=np.int8), np.empty(0, dtype=object),
                np.empty(0, dtype=np.int64))
    return (np.stack(xs), np.array(ys, dtype=np.int8),
            np.array(gs, dtype=object), np.array(ends, dtype=np.int64))
