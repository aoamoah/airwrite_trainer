# AirWrite Trainer

Trains and compares **Random Forest**, **LSTM** and **GRU** models for
writing-state detection (`writing` vs `not_writing`) from MediaPipe hand
landmarks, across multiple datasets, and exports a detailed Markdown report
for academic use.

## Setup

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
```

## Data layout

```
airwrite_trainer/
  dataset/            <- own recordings (AirWrite Capture export)
  dataset_WITA/       <- WITA videos, extracted + annotated in the collector
  dataset_IPN/        <- IPN videos, extracted + annotated in the collector
    P001/S002/landmarks.csv
    P001/S002/labels.csv
    ...
  reports/            <- one folder per run: report.md, figures/, models/, results.json
```

All three folders use the AirWrite Capture export layout — every source is
run through the same MediaPipe extraction in the collector, so features are
identical everywhere. Only `landmarks.csv` and `labels.csv` are read.
Folders that don't exist yet are skipped.

## Run

```bash
venv/bin/python train.py             # full run: all dataset folders present
venv/bin/python train.py --quick     # fast smoke test of the whole pipeline
venv/bin/python train.py --datasets dataset_WITA dataset_IPN
venv/bin/python train.py --models random_forest lstm
```

## What a run produces

`reports/run_<timestamp>/`:
- `report.md` — environment, dataset statistics, class balance, the verbatim
  configuration, the split composition (which sessions landed in which
  split), per-run hyperparameters / timings / train-val-test metrics /
  confusion matrices / training curves / feature importances, and a final
  ranked comparison naming the best configuration.
- `figures/` — every figure as PNG, ready to drop into a thesis.
- `models/` — trained models (`.joblib` for RF, `.keras` for LSTM/GRU).
- `results.json` — every number in the report, machine-readable.

## Methodology defaults (all configurable in `config.yaml`)

- **Split:** whole sessions go to one of train/val/test (70/15/15) — random
  frame-level splits leak near-identical neighboring frames and inflate
  scores. Downloaded datasets without a session column are split in
  contiguous row chunks for the same reason.
- **Windows:** LSTM/GRU are trained on sliding windows (15/30/60 frames by
  default, all reported); Random Forest is frame-level.
- **Normalization:** landmarks are translated to the wrist origin and scaled
  by hand span, removing dependence on screen position.
- **Imbalance:** class weights for the recurrent models,
  `class_weight=balanced` for the forest.
- **Ranking metric:** F1 of the `writing` class on the held-out test split.
