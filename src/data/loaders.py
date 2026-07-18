"""Load dataset folders into a common form.

All three sources (own recordings, WITA, IPN) are extracted through the
same MediaPipe pipeline in the collector, so every folder uses the AirWrite
Capture export layout: <folder>/P###/S###/{landmarks.csv, labels.csv}.

A loader returns a `LoadedDataset`:
- frames: DataFrame with the feature columns, plus `label` ("writing" /
  "not_writing"), `group` (session, the splitting unit) and `participant`.
- feature_columns: ordered list of feature column names.
- meta: dict of dataset facts recorded in the report.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

LANDMARK_COLUMNS = [f"l{i}_{axis}" for i in range(21) for axis in ("x", "y", "z")]
VALID_LABELS = {"writing", "not_writing"}


@dataclass
class LoadedDataset:
    name: str
    frames: pd.DataFrame
    feature_columns: list[str]
    meta: dict = field(default_factory=dict)


def load_dataset_folder(dataset_dir: str | Path, name: str) -> LoadedDataset:
    root = Path(dataset_dir)
    session_dirs = sorted(
        d for d in root.glob("P*/S*")
        if (d / "landmarks.csv").exists() and (d / "labels.csv").exists()
    )
    if not session_dirs:
        raise FileNotFoundError(
            f"No sessions with landmarks.csv + labels.csv found under {root}"
        )

    parts = []
    skipped = []
    for d in session_dirs:
        landmarks = pd.read_csv(d / "landmarks.csv")
        labels = pd.read_csv(d / "labels.csv")
        merged = landmarks.merge(labels, on="frame_index", how="inner")
        if merged.empty or not set(merged["label"]).issubset(VALID_LABELS):
            skipped.append(str(d))
            continue
        merged["group"] = f"{d.parent.name}/{d.name}"
        merged["participant"] = d.parent.name
        merged["hand_detected"] = merged["hand_detected"].astype(int)
        parts.append(merged)

    if not parts:
        raise FileNotFoundError(f"No usable sessions under {root}")

    frames = pd.concat(parts, ignore_index=True)
    feature_columns = ["hand_detected"] + LANDMARK_COLUMNS

    meta = {
        "type": "airwrite export (landmarks.csv + labels.csv per session)",
        "path": str(root),
        "sessions": len(parts),
        "sessions_skipped": skipped,
        "participants": sorted(frames["participant"].unique().tolist()),
        "total_frames": len(frames),
        "frames_without_hand": int((frames["hand_detected"] == 0).sum()),
        "label_counts": frames["label"].value_counts().to_dict(),
    }
    return LoadedDataset(name, frames, feature_columns, meta)
