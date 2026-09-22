# AirWrite Trainer

Trains and compares two **rule-based baselines**, **Random Forest**, **LSTM**
and **GRU** models for writing-state detection (`writing` vs `not_writing`)
from MediaPipe hand landmarks, across up to three datasets, and exports a
detailed Markdown report for academic use.

Evaluation is **participant-level cross-validation** by default: every
configuration is trained once per held-out participant (or participant
group), and reported as mean ± standard deviation over folds. No participant
contributes frames to more than one split of a fold, so a score measures
generalisation to an unseen hand rather than to another session of a hand
the model already trained on.

The three datasets are folders produced by the AirWrite Capture collector's
export — own recordings plus WITA and IPN videos, all extracted through the
same MediaPipe pipeline, so features are identical everywhere. Identical
features do **not** make the corpora interchangeable, though: the report
computes the rank agreement between them (Kendall tau) and
`--transfer A:B` measures cross-corpus generalisation directly, so
"the source doesn't matter" is a result rather than an assumption.

Three corrections are applied before any model sees the data, because
without them the features do not describe writing:

- **Aspect correction.** MediaPipe x is a fraction of frame width and y a
  fraction of frame height, so the axes carry different scales on any
  non-square video — and nine sessions are off the 640x480 spec, from
  362x640 portrait to 832x464. x is rescaled into image-height units.
- **Time normalisation.** Speeds are divided by the real `dt` from
  `timestamp_ms` rather than counting frames. Three sessions are 60 fps and
  the rest 30, so frame-indexed speeds put them on different scales.
- **Motion features.** Fingertip and wrist speed in hand-widths per second,
  with rolling means and standard deviations. Air-writing is mostly
  whole-hand translation, which wrist normalisation deletes by construction;
  these are computed before it is applied.

Scores are reported at **frame level and episode level**. Frame F1 asks what
share of frames were labelled right; a user experiences whether the detector
noticed they started writing and held on until they stopped. A model can
score a decent frame F1 while shattering every episode into fragments, which
is what "it keeps cutting out" means in the live app.

Every model's decision threshold is fitted on validation data. Previously
only the rule baselines had a fitted operating point while the learned models
were scored at a flat 0.5 — on a task whose writing prior runs from 10% to
84% across participants, that handicapped one side of the very comparison
the study exists to make.

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
  dataset_<Source>/   <- any other source, one folder each (add it to
    P001/S002/           dataset_dirs in config.yaml)
      landmarks.csv
      labels.csv      <- writing / not_writing / unsure
      metadata.json   <- resolution, fps, extraction settings
    ...
  reports/            <- created per run
```

Folders that don't exist or contain no labelled sessions are skipped with a
note. A session containing a label outside writing / not_writing / unsure is
skipped and listed in the report.

**What is read from each session:**
- `landmarks.csv` — image landmarks `l*`, plus (collector schema v2) world
  landmarks `wl*` and the per-frame `handedness`.
- `labels.csv` — `unsure` frames stay on the timeline, so features and
  windows are computed across them as a live app would, but they are never a
  training target and never scored.
- `metadata.json` — the video resolution, used for aspect correction. A
  session with no known resolution stops the run
  (`preprocessing.require_resolution`), because leaving a portrait or 16:9
  clip uncorrected silently puts it on a different scale from everything
  else. Exports made before collector schema v2 fall back to
  `session_resolutions.json`, built with
  `../data_collector/venv/bin/python scan_resolutions.py`.

**Every session is resampled to `preprocessing.target_fps` (30).** Rolling
windows, recurrent windows, smoothing durations and event tolerances are all
counted in frames, so without this a 30-frame window is one second from one
device and half a second from another. The resampler is sample-and-hold on
real timestamps (causal, never invents a pose); sessions already within 5% of
the target are untouched.

**Check a new source's label granularity before pooling it.** The report's
dataset section tabulates, per corpus, the median writing run and median
pause between writing runs. Own recordings: median pause ~0.5 s, 27% under
167 ms — the labels mark pauses inside letters. IPN: median gap 11 s — the
labels mark whole gestures. Pooling those trains on two definitions of the
task.

## 3. Training — command reference

Everything is `venv/bin/python train.py` plus optional flags:

| Flag | Meaning | Default |
|---|---|---|
| `--datasets NAME [NAME ...]` | Which dataset folders to train on | all of `dataset dataset_WITA dataset_IPN` that exist |
| `--models NAME [NAME ...]` | Which models: `velocity_threshold`, `extension_threshold`, `random_forest`, `lstm`, `gru` | all five |
| `--scheme NAME` | `auto` / `lopo` / `group_kfold` / `holdout` | `auto` |
| `--max-folds N` | Cap the number of folds (debugging) | all folds |
| `--velocity` / `--no-velocity` | Add or drop frame-to-frame landmark deltas | config value |
| `--quick` | Tiny smoke run (3 epochs, 50 trees) to check the pipeline | off |
| `--config PATH` | Alternative config file | `config.yaml` |

### All three datasets, all models (the full comparison)

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

### Feature ablations

```bash
venv/bin/python train.py --no-motion     # wrist-normalised pose only
venv/bin/python train.py --motion-only   # motion features, no pose columns
venv/bin/python train.py --no-aspect     # skip the width/height correction
venv/bin/python train.py --no-fit-threshold   # score everything at 0.5
venv/bin/python train.py --pose-source world   # MediaPipe world landmarks as the pose
venv/bin/python train.py --no-mirror           # keep left hands un-mirrored
venv/bin/python train.py --window-label majority   # the old window labelling
venv/bin/python train.py --target-fps 0        # native frame rates
```

`--pose-source world` is the resolution-independence arm: world landmarks
are metric and hand-centred, so no aspect correction touches the pose at all
(motion features still come from image landmarks, which carry position).

`--no-motion` is the arm to keep for the write-up: a speed rule beating the
learned models under pose-only preprocessing is a genuine reportable result
about feature design, and it only means something next to the pose + motion
arm.

### Cross-corpus transfer

```bash
venv/bin/python train.py --transfer dataset_IPN:dataset
```

Trains on every participant of the first corpus and evaluates on every
participant of the second. Validation — and therefore the threshold fit —
comes from the training corpus only, so the target corpus is untouched
during model selection. This is the experiment that decides whether the data
source matters: if it did not, transfer would land close to within-corpus
cross-validation.

### Fast pipeline check before a long run

```bash
venv/bin/python train.py --quick --max-folds 2
```

### The combined corpus (RQ1 — across data sources)

```bash
venv/bin/python train.py --datasets dataset dataset_IPN --combine \
    --scheme group_kfold
```

Pools the corpora into one training set instead of training each in
isolation. Folds still hold out whole participants, and participant/session
IDs are namespaced by corpus if two corpora reuse an ID — otherwise an
identically-named `P001` in both would be treated as one person and land on
both sides of a fold. The report adds a **per-source breakdown**: how the
pooled model scored on each corpus's held-out participants separately, which
is what tells you whether pooling helped your data or only lifted the
average. Add `--also-separate` to train the individual corpora in the same
run, so all three arms land in one report.

### Merging runs into one table

```bash
venv/bin/python merge_reports.py                    # every run in reports/
venv/bin/python merge_reports.py --csv table.csv    # also as CSV
venv/bin/python merge_reports.py reports/run_A reports/run_B --out ch4.md
```

A run only reports what was trained inside it, so a study split across runs
(one model per run, one corpus per run) has no single ranked table.
`merge_reports.py` reads each run's `summary.json` and writes
`reports/merged_comparison.md`: one ranked row per configuration, the
learned-vs-rule-baseline margin per corpus, the combined-corpus breakdown,
and a "superseded" list when a configuration was re-run.

### The participant-leakage ablation

```bash
venv/bin/python train.py --scheme holdout      # session-level split
venv/bin/python train.py                       # participant-level folds
```

Identical data, features and models; the only difference is whether a
participant may appear on both sides of the split. The gap between the two
reports is the size of the leakage effect, and it belongs in the write-up as
a result rather than being quietly discarded.

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
  recurrent models' temporal advantage is measured fairly). Note this is
  *not* a substitute for `add_motion`: the deltas are taken after wrist
  normalisation, so they are still translation-free and still frame-indexed.
- `preprocessing.aspect_correct` — rescale x by width/height into
  image-height units (default on; needs `session_resolutions.json`).
- `preprocessing.time_normalize` — divide speeds by the real `dt` from
  `timestamp_ms` instead of counting frames (default on). Applies to the
  rule baselines' speed signal as well as the motion features, so both sides
  of the comparison get the same correction or neither does.
- `preprocessing.add_motion` — fingertip and wrist speed in hand-widths per
  second plus rolling means and standard deviations (default on).
- `preprocessing.pose_features` — keep the 63 landmark columns (default on;
  `false` is the motion-only ablation).
- `evaluation.fit_threshold` — fit every model's decision threshold on the
  validation split (default on). `evaluation.threshold_beta` picks the
  F-beta it maximises; `2.0` weights recall, which is what a live detector
  wants.
- `evaluation.event_metrics` and `evaluation.events.*` — episode-level
  scoring. `iou_threshold` (default 0.5) is how much of an episode a
  prediction must cover; `tolerance_frames` (default 12, about 400 ms at
  30 fps) is the onset-proximity criterion and matches the tolerance the
  closest published video work reports; `min_event_frames` (default 3)
  drops annotation noise — 10.7% of the label runs in `dataset` are one or
  two frames long.
- `evaluation.scheme` — `auto` (default: leave-one-participant-out up to
  `lopo_max_participants`, grouped k-fold beyond), `lopo`, `group_kfold`, or
  `holdout` (the single split under `split:`, kept for the leakage ablation).
- `evaluation.val_participants` / `val_fraction` — size and target frame
  share of the early-stopping validation set. It is always drawn from the
  **training** participants, and among the candidates the subset whose
  writing prior is closest to the training pool wins — early stopping
  against an unrepresentative validation prior is what makes a model learn
  to answer "writing" almost unconditionally.
- `split.method` — only consulted when the scheme is `holdout`: `session`,
  `participant`, or `random` (leaky; baseline only).
- `split.ratios` — train/val/test for `holdout`, default `[0.70, 0.15, 0.15]`.
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
| `report.md` | Environment and library versions; per-dataset statistics (sessions, participants, frames, class balance figure and tables); preprocessing applied; **the fold table** (who is in train/val/test and the writing prior of each split, per fold); the verbatim config; per-configuration sections with hyperparameters, fold count, training time, epochs, mean ± sd metric tables, pooled confusion matrix, per-fold breakdown, training curves, feature importances (RF), classification report; **model-major comparison** (one row per model, one column per corpus) with the Kendall-tau rank agreement between corpora; the per-corpus best and the **learned-vs-rule-baseline** margin; the **episode-level** table (IoU and onset F1, onset lag, fragments per episode); the **hand-detected vs no-hand** breakdown; and an **external benchmarks** section |
| `figures/` | Every figure as PNG (150 dpi) — drop straight into a thesis, including the per-fold spread plot |
| `models/` | Trained models from the first fold (`report.save_models`): `<dataset>_random_forest.joblib`, `<dataset>_<rnn>_w<size>.keras` — these are what an inference app loads |
| `results.json` | Every metric, history and setting for every fold, machine-readable |
| `summary.json` | Per-configuration aggregates (mean/sd/min/max, per-fold values, pooled confusion) |
| `folds.json` | Fold composition and per-split class priors |

## 6. Using the models in a Qt / C++ inference app

`.keras` / `.joblib` files are Python-only. Every run therefore also
exports **ONNX** copies of all models (`models/*.onnx`, parity-checked
against the originals at export time) plus `models/inference_spec.json`
describing exactly how to prepare inputs. One C++ runtime — [ONNX
Runtime](https://onnxruntime.ai) — then serves all three model types.

To re-export an older run manually:

```bash
CUDA_VISIBLE_DEVICES= venv/bin/python export_onnx.py reports/run_<timestamp>
```

### The contract (from `inference_spec.json`)

1. Per frame, build the feature vector in `feature_order` (default 64
   values: `hand_detected`, then `l0_x…l20_z` from MediaPipe).
2. Apply the same normalization: subtract wrist (l0) x/y/z from every
   landmark, divide by the max distance of any landmark from the wrist;
   all-zero frames stay zero.
3. **LSTM/GRU**: keep a rolling buffer of the last *window* frames (oldest
   first) and run input shape `[1, window, n_features]` — output is the
   sigmoid probability of `writing`. **Random Forest**: single frame,
   `[1, n_features]` — output 1 is `probabilities[not_writing, writing]`.
4. Compare against `decision_threshold` (0.5).

### Minimal Qt/C++ example

```cpp
#include <onnxruntime_cxx_api.h>

Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "airwrite");
Ort::Session session(env, "dataset_gru_w30.onnx", Ort::SessionOptions{});

// rolling window buffer filled from MediaPipe, preprocessed per the spec
std::vector<float> input(1 * 30 * 64);          // [1, window, features]
std::array<int64_t, 3> shape{1, 30, 64};

auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
Ort::Value tensor = Ort::Value::CreateTensor<float>(
    mem, input.data(), input.size(), shape.data(), shape.size());

const char* in_names[]  = {"input"};
const char* out_names[] = {"output_0"};
auto out = session.Run(Ort::RunOptions{}, in_names, &tensor, 1, out_names, 1);
float p_writing = out[0].GetTensorData<float>()[0];
bool writing = p_writing >= 0.5f;
```

CMake: link against onnxruntime (`find_package(onnxruntime)` with the
prebuilt release, or point `target_include_directories` /
`target_link_libraries` at the extracted archive). MediaPipe's C++ hand
landmarker provides the per-frame landmarks on the Qt side. Check the
actual input/output names with `Ort::Session::GetInputNameAllocated` — the
RF models use `input` → outputs `[label, probabilities]`.

## 7. Methodology notes (what a reviewer will ask)

- **No participant leakage:** the default scheme holds out whole
  participants, so no hand appears on both sides of a fold. Session-level
  splitting is still available (`--scheme holdout`) and is reported as an
  ablation, not as the headline result — a session split leaves the same
  participant in train and test and inflates every metric.
- **No temporal leakage:** whole sessions go to one split within the
  holdout scheme too; random frame splits exist only as an
  explicitly-labeled leaky baseline.
- **Validation is never the test fold, and never an odd distribution:** the
  early-stopping set is drawn from training participants and chosen so its
  writing prior tracks the training pool's. Every fold's priors are printed
  in the report, so a mismatch is visible rather than silent.
- **Windowing:** RNN windows never cross session boundaries; a window is
  labelled by its **last** frame — the "writing now?" question a live app
  answers, and the frame its score is placed on. The older majority label
  trained models to report a state half a window old and could never target
  a pause shorter than half the window (`--window-label majority` keeps it
  as an ablation). Windows whose last frame is `unsure` are dropped.
- **Pauses are the benchmark.** Mid-letter detection means detecting the
  pauses inside writing, so the report scores them with the protocol of the
  closest published work (Pen-In-Air States from Video, arXiv 2606.02342):
  a pause bounded by writing is matched when both its boundaries are within
  5 / 10 / 12 frames, one-to-one, F2. Their published F2 sits beside ours in
  the pause table. Segmental Edit and F1@{10,25,50} — the temporal action
  segmentation standard — are reported for the over-segmentation that frame
  accuracy hides, along with signed onset delay.
- **Canonical hand:** poses MediaPipe labels Left are mirrored into
  right-hand form (`mirror_left_hands`), so left-handed participants and
  mirrored front-camera sources share one pose space. Only the
  wrist-relative pose is mirrored; speeds are unchanged by a mirror.
- **Imbalance:** class weights for LSTM/GRU, `class_weight=balanced` for
  the forest, and average precision reported alongside ROC AUC — its
  baseline is the class prior rather than 0.5, which is the honest reference
  when the prior moves between 10% and 84% across participants.
- **One operating point rule for everybody:** every model's threshold is
  fitted on validation and applied to test. This used to be true only of the
  rule baselines while the learned models were scored at a flat 0.5, which
  quietly handicapped them in the comparison the study turns on. Turn it off
  with `--no-fit-threshold` to reproduce the older numbers.
- **Episode-level scoring:** predictions are also matched to writing
  episodes one-to-one, under both an IoU criterion and an onset-tolerance
  criterion. One-to-one matters — with overlap matching, a single prediction
  spanning the whole session would "detect" every episode in it. Fragments
  per episode and median onset lag are reported next to precision and
  recall, because a detector that chops one stroke into six is experienced
  as broken however good its frame F1 looks.
- **Detection-state breakdown:** scores are split by whether MediaPipe found
  a hand. 12.1% of `dataset` frames have none, and only 3.0% of those are
  writing against 38.1% where a hand is visible, so "no hand" is very nearly
  a free correct answer. The collector extracts with inter-frame tracking
  (`RunningMode.VIDEO`) that a live app cannot match, so the app sees a
  lower detection rate than training did and any score propped up by those
  frames will not survive deployment.
- **The corpus is a reported factor, not a nuisance variable:** the report
  computes Kendall tau between corpora's model rankings. Where the rankings
  disagree, a pooled leaderboard ranks models by which corpus they were
  evaluated on rather than by how well they detect writing. `--transfer A:B`
  measures the gap directly.
- **External benchmarks:** the report carries a section of the closest
  published work with the reasons none of it is like-for-like. There is no
  established benchmark for this exact task — binary writing-state detection
  from RGB hand pose under subject-independent evaluation — and saying so is
  more useful than forcing a comparison.
- **Baselines:** two non-learned detectors (fingertip speed, finger
  extension) are trained and scored alongside the models, so "the learned
  model helps" is a measured margin rather than an assumption. Their
  signals are computed from raw landmarks, before wrist normalisation
  removes the absolute motion the speed rule depends on.
- **Fair comparison:** identical features, preprocessing, folds and metrics
  for every model; early stopping on validation loss with best weights
  restored. Frame-level and window-level scores are labelled as such.
- **Reported statistic:** mean ± standard deviation of F1 (`writing`) over
  held-out participants, with every fold listed (accuracy alone would
  reward always predicting the majority class, and a single split hides how
  much the score moves between participants).
