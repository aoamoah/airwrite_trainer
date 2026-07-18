# AirWrite Trainer

Trains and compares **Random Forest**, **LSTM** and **GRU** models for
writing-state detection (`writing` vs `not_writing`) from MediaPipe hand
landmarks, across up to three datasets, and exports a detailed Markdown
report for academic use.

The three datasets are folders produced by the AirWrite Capture collector's
export — own recordings plus WITA and IPN videos, all extracted through the
same MediaPipe pipeline, so features are identical everywhere.

---

## 1. Setup (once)

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

GPU: on WSL with an NVIDIA card nothing else is needed — `requirements.txt`
installs `tensorflow[and-cuda]` and `train.py` loads the CUDA libraries
itself. The run header in the report records whether a GPU was visible.

## 2. Data layout

Copy the exported dataset folders from the collector into this directory:

```
airwrite_trainer/
  dataset/            <- own recordings
  dataset_WITA/       <- WITA sessions
  dataset_IPN/        <- IPN sessions
    P001/S002/landmarks.csv
    P001/S002/labels.csv
    ...
  reports/            <- created per run
```

Only `landmarks.csv` and `labels.csv` are read (video.mp4 / metadata.json
may be present; they are ignored). Folders that don't exist or contain no
labeled sessions are skipped with a note.

## 3. Training — command reference

Everything is `venv/bin/python train.py` plus optional flags:

| Flag | Meaning | Default |
|---|---|---|
| `--datasets NAME [NAME ...]` | Which dataset folders to train on | all of `dataset dataset_WITA dataset_IPN` that exist |
| `--models NAME [NAME ...]` | Which models: `random_forest`, `lstm`, `gru` | all three |
| `--quick` | Tiny smoke run (3 epochs, 50 trees) to check the pipeline | off |
| `--config PATH` | Alternative config file | `config.yaml` |

### All three datasets, all three models (the full comparison)

```bash
venv/bin/python train.py
```

### One specific dataset

```bash
venv/bin/python train.py --datasets dataset            # own recordings only
venv/bin/python train.py --datasets dataset_WITA       # WITA only
venv/bin/python train.py --datasets dataset_IPN        # IPN only
```

### One specific model

```bash
venv/bin/python train.py --models random_forest
venv/bin/python train.py --models lstm
venv/bin/python train.py --models gru
```

### A specific dataset with a specific model

```bash
venv/bin/python train.py --datasets dataset_WITA --models lstm
```

### Two datasets, two models

```bash
venv/bin/python train.py --datasets dataset dataset_IPN --models lstm gru
```

### Fast pipeline check before a long run

```bash
venv/bin/python train.py --quick
```

Every run — whatever the filters — produces one complete report comparing
everything that was trained in that run.

## 4. Tuning a run — `config.yaml`

The config is copied verbatim into every report, so results are always
traceable to their settings. The knobs:

- `seed` — global seed (Python / NumPy / TensorFlow).
- `preprocessing.normalize` — `wrist` (default: translate to wrist origin,
  scale by hand span) or `none`.
- `preprocessing.drop_undetected` — drop frames with no hand instead of
  keeping them as all-zero rows (default: keep).
- `preprocessing.add_velocity` — append frame-to-frame landmark deltas
  (gives the frame-level model temporal information; off by default so the
  recurrent models' temporal advantage is measured fairly).
- `split.method` — `session` (default), `participant`
  (leave-participants-out), or `random` (leaky; baseline only).
- `split.ratios` — train/val/test, default `[0.70, 0.15, 0.15]`.
- `windows` — the sequence configurations LSTM/GRU are trained on; each
  entry is trained and reported separately. Default: 15/30/60-frame windows.
- `models.*` — hyperparameters per model (units, dropout, epochs, batch
  size, early-stopping patience, learning rate; forest size and depth).

Example: to train GRU only on 30-frame windows, reduce `windows:` to
`- {size: 30, stride: 5}` and run `--models gru`.

## 5. What a run produces

`reports/run_<timestamp>/`:

| File | Contents |
|---|---|
| `report.md` | Environment and library versions; per-dataset statistics (sessions, participants, frames, class balance figure and tables); preprocessing applied; **exact split composition** (which sessions landed in train/val/test); the verbatim config; per-run sections with hyperparameters, sample counts, training time, epochs, train/val/test metric tables, confusion matrix, training curves, feature importances (RF), full classification report; final ranked comparison table and figure; the **best configuration** named with its scores |
| `figures/` | Every figure as PNG (150 dpi) — drop straight into a thesis |
| `models/` | Trained models: `<dataset>_random_forest.joblib`, `<dataset>_<rnn>_w<size>.keras` — these are what an inference app loads |
| `results.json` | Every metric, history and setting, machine-readable |

## 6. Methodology notes (what a reviewer will ask)

- **No temporal leakage:** whole sessions go to one split; random frame
  splits are available only as an explicitly-labeled leaky baseline. The
  splitter balances splits by frame count, guarantees a non-empty test
  split, and repairs splits that would contain a single class.
- **Windowing:** RNN windows never cross session boundaries; a window's
  label is the majority frame label.
- **Imbalance:** class weights for LSTM/GRU, `class_weight=balanced` for
  the forest.
- **Fair comparison:** identical features, preprocessing, splits and
  metrics for every model; early stopping on validation loss with best
  weights restored.
- **Ranking metric:** F1 of the `writing` class on the held-out test split
  (accuracy alone would reward always predicting the majority class).
