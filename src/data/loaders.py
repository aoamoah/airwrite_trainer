"""Load dataset folders into a common form.

Every source (own recordings, WITA, IPN, and any new `dataset_<Source>`) is
extracted through the same MediaPipe pipeline in the collector, so every
folder uses the AirWrite Capture export layout:
<folder>/P###/S###/{landmarks.csv, labels.csv, metadata.json}.

A loader returns a `LoadedDataset`:
- frames: DataFrame with the feature columns, plus `label` ("writing" /
  "not_writing" / "unsure"), `scored` (False for unsure frames), `group`
  (session, the splitting unit), `participant`, `source`, and per-session
  `aspect` (frame width / height, NaN when unknown).
- feature_columns: ordered list of feature column names.
- meta: dict of dataset facts recorded in the report.

`unsure` frames — the annotator could not judge them — stay in the timeline
so rolling features and windows are computed across them exactly as a live
app would, but they are never a training target and never scored.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.label_stats import label_granularity
from src.data.timing import resample_session

LANDMARK_COLUMNS = [f"l{i}_{axis}" for i in range(21) for axis in ("x", "y", "z")]
WORLD_COLUMNS = [f"wl{i}_{axis}" for i in range(21) for axis in ("x", "y", "z")]
VALID_LABELS = {"writing", "not_writing", "unsure"}
UNSCORED_LABELS = {"unsure"}
RESOLUTIONS_FILE = Path(__file__).parents[2] / "session_resolutions.json"


@dataclass
class LoadedDataset:
    name: str
    frames: pd.DataFrame
    feature_columns: list[str]
    meta: dict = field(default_factory=dict)


def _read_metadata(session_dir: Path) -> dict:
    path = session_dir / "metadata.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _legacy_resolutions() -> dict:
    """Resolutions scanned off the videos by scan_resolutions.py, for exports
    made before metadata.json carried them (collector schema_version < 2)."""
    if not RESOLUTIONS_FILE.exists():
        return {}
    return json.loads(RESOLUTIONS_FILE.read_text()).get("sessions", {})


def _session_aspect(meta: dict, legacy: dict, key: str) -> tuple[float | None, str]:
    video = meta.get("video") or {}
    if video.get("width") and video.get("height"):
        return video["width"] / video["height"], "metadata.json"
    if video.get("aspect"):
        return float(video["aspect"]), "metadata.json"
    if key in legacy and legacy[key].get("aspect"):
        return float(legacy[key]["aspect"]), "session_resolutions.json"
    return None, "unknown"


def load_dataset_folder(dataset_dir: str | Path, name: str,
                        exclude_participants: set[str] | None = None,
                        target_fps: float | None = None,
                        ) -> LoadedDataset:
    """Load one export folder.

    `exclude_participants` drops whole participants before anything is
    computed. Excluding data is a claim that has to survive a reader, so the
    exclusion and its reason live in `meta` and are printed verbatim in the
    report — an omitted participant that goes unmentioned is the difference
    between a cleaned corpus and a cherry-picked one.

    `target_fps` resamples every session onto one frame rate (see
    `src.data.timing`); None keeps each session's native rate.
    """
    root = Path(dataset_dir)
    excluded = set(exclude_participants or ())
    session_dirs = sorted(
        d for d in root.glob("P*/S*")
        if (d / "landmarks.csv").exists() and (d / "labels.csv").exists()
        and d.parent.name not in excluded
    )
    dropped = sorted({d.parent.name for d in root.glob("P*/S*")} & excluded)
    if not session_dirs:
        raise FileNotFoundError(
            f"No sessions with landmarks.csv + labels.csv found under {root}"
        )

    legacy = _legacy_resolutions()
    parts = []
    skipped = []
    aspect_sources: dict[str, int] = {}
    resolutions: dict[str, int] = {}
    timing_notes: dict[str, dict] = {}
    schema_versions: dict[str, int] = {}
    for d in session_dirs:
        landmarks = pd.read_csv(d / "landmarks.csv")
        labels = pd.read_csv(d / "labels.csv")
        merged = landmarks.merge(labels, on="frame_index", how="inner")
        unknown = set(merged["label"].dropna()) - VALID_LABELS
        if merged.empty or unknown:
            skipped.append(f"{d}" + (f" (unknown labels: {sorted(unknown)})" if unknown else ""))
            continue
        merged = merged.sort_values("frame_index", kind="stable").reset_index(drop=True)
        group = f"{d.parent.name}/{d.name}"

        meta = _read_metadata(d)
        schema = str(meta.get("schema_version", 1))
        schema_versions[schema] = schema_versions.get(schema, 0) + 1
        aspect, how = _session_aspect(meta, legacy, f"{root.name}/{group}")
        aspect_sources[how] = aspect_sources.get(how, 0) + 1
        video = meta.get("video") or legacy.get(f"{root.name}/{group}") or {}
        if video.get("width") and video.get("height"):
            shape = f"{video['width']}x{video['height']}"
            resolutions[shape] = resolutions.get(shape, 0) + 1

        if target_fps and "timestamp_ms" in merged:
            merged, note = resample_session(merged, float(target_fps))
            timing_notes[group] = note

        merged = merged.copy()          # one block before adding columns
        merged["group"] = group
        merged["participant"] = d.parent.name
        merged["source"] = name
        merged["aspect"] = aspect if aspect is not None else np.nan
        merged["hand_detected"] = merged["hand_detected"].astype(int)
        merged["scored"] = ~merged["label"].isin(UNSCORED_LABELS)
        if "handedness" in merged:
            merged["handedness"] = merged["handedness"].fillna("").astype(str)
        parts.append(merged)

    if not parts:
        raise FileNotFoundError(f"No usable sessions under {root}")

    frames = pd.concat(parts, ignore_index=True)
    feature_columns = ["hand_detected"] + LANDMARK_COLUMNS
    world_sessions = sum(1 for p in parts if set(WORLD_COLUMNS) <= set(p.columns))

    resampled = [n for n in timing_notes.values() if n.get("resampled")]
    fps_seen = sorted({n["measured_fps"] for n in timing_notes.values()
                       if n.get("measured_fps")})
    meta = {
        "type": "airwrite export (landmarks.csv + labels.csv per session)",
        "path": str(root),
        "excluded_participants": (
            f"{', '.join(dropped)} — off-spec capture: portrait or non-4:3 "
            "aspect and/or 60 fps, and the lowest hand-detection rates in the "
            "corpus" if dropped else "none"),
        "sessions": len(parts),
        "sessions_skipped": skipped,
        "participants": sorted(frames["participant"].unique().tolist()),
        "total_frames": len(frames),
        "scored_frames": int(frames["scored"].sum()),
        "frames_without_hand": int((frames["hand_detected"] == 0).sum()),
        "label_counts": frames["label"].value_counts().to_dict(),
        "export_schema_versions": schema_versions,
        "resolutions_seen": resolutions,
        "aspect_source": aspect_sources,
        "sessions_with_world_landmarks": world_sessions,
        "sessions_with_handedness": sum(1 for p in parts if "handedness" in p),
        "timing": ({"target_fps": target_fps,
                    "sessions_resampled": len(resampled),
                    "native_fps_seen": fps_seen}
                   if target_fps else {"target_fps": None,
                                       "note": "native frame rates kept"}),
        "label_granularity": label_granularity(frames),
    }
    return LoadedDataset(name, frames, feature_columns, meta)


def combine_datasets(datasets: dict[str, LoadedDataset],
                     name: str = "combined") -> LoadedDataset:
    """Pool several corpora into one training corpus (RQ4).

    Folds are still cut by participant, so a person held out of training is
    held out of every corpus they appear in. Participant and session IDs are
    namespaced by source wherever two corpora reuse an ID — without that, an
    identically-named `P001` in two corpora would be treated as one person
    and land on both sides of a fold, which is the leakage this whole design
    exists to prevent.
    """
    if len(datasets) < 2:
        raise ValueError("combine_datasets needs at least two datasets")

    features = [tuple(ds.feature_columns) for ds in datasets.values()]
    if len(set(features)) != 1:
        raise ValueError("cannot combine datasets with different feature columns")

    seen: dict[str, str] = {}
    collisions = set()
    for src, ds in datasets.items():
        for p in ds.frames["participant"].unique():
            if seen.setdefault(p, src) != src:
                collisions.add(p)

    parts = []
    for src, ds in datasets.items():
        f = ds.frames.copy()
        if collisions:  # namespace everything, so IDs stay comparable
            f["participant"] = src + ":" + f["participant"].astype(str)
            f["group"] = src + ":" + f["group"].astype(str)
        parts.append(f)
    frames = pd.concat(parts, ignore_index=True)

    per_source = {
        src: {
            "frames": int((frames["source"] == src).sum()),
            "participants": int(
                frames.loc[frames["source"] == src, "participant"].nunique()),
            "writing_share": round(float(
                (frames.loc[frames["source"] == src, "label"] == "writing").mean()), 4),
            "label_granularity": ds.meta.get("label_granularity"),
        }
        for src, ds in datasets.items()
    }
    meta = {
        "type": f"combined corpus ({', '.join(datasets)})",
        "sources": list(datasets),
        "per_source": per_source,
        "sessions": int(frames["group"].nunique()),
        "participants": sorted(frames["participant"].unique().tolist()),
        "total_frames": len(frames),
        "frames_without_hand": int((frames["hand_detected"] == 0).sum()),
        "label_counts": frames["label"].value_counts().to_dict(),
        "id_namespacing": (f"applied — IDs shared across corpora: "
                           f"{sorted(collisions)}" if collisions
                           else "not needed — participant IDs are unique across corpora"),
    }
    first = next(iter(datasets.values()))
    return LoadedDataset(name, frames, list(first.feature_columns), meta)
