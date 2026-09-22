"""Convert a run's trained models to ONNX for C++ (e.g. Qt) inference apps.

    venv/bin/python export_onnx.py reports/run_<timestamp>

Writes, next to each model in <run>/models/:
- <name>.onnx — the converted model, numerically parity-checked here
- inference_spec.json — everything the inference app must reproduce:
  feature order, normalization, window sizes, threshold, label mapping.

Run with the GPU hidden (train.py does this automatically): tracing on GPU
bakes CudnnRNN ops into the graph, which ONNX cannot represent.
"""

import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from src.config import load_config
from src.data.loaders import LANDMARK_COLUMNS, WORLD_COLUMNS
from src.data.preprocess import MOTION_MEAN_WINDOWS, MOTION_SD_WINDOWS

RTOL = 1e-4


def convert_keras(path: Path) -> Path:
    import tensorflow as tf
    import tf2onnx
    import onnx
    from tensorflow import keras

    model = keras.models.load_model(path)
    _, window, n_feat = model.input_shape
    spec = (tf.TensorSpec((None, window, n_feat), tf.float32, name="input"),)
    onnx_model, _ = tf2onnx.convert.from_function(
        tf.function(lambda x: model(x)), input_signature=spec, opset=17)
    out = path.with_suffix(".onnx")
    onnx.save(onnx_model, str(out))

    x = np.random.default_rng(0).random((4, window, n_feat), dtype=np.float32)
    expected = model.predict(x, verbose=0).ravel()
    _assert_parity(out, {"input": x}, expected, path.name)
    return out


def convert_forest(path: Path) -> Path:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    model = joblib.load(path)
    n_feat = model.n_features_in_
    onnx_model = convert_sklearn(
        model,
        initial_types=[("input", FloatTensorType([None, n_feat]))],
        options={id(model): {"zipmap": False}},  # plain tensors for C++
        target_opset=17,
    )
    out = path.with_suffix(".onnx")
    out.write_bytes(onnx_model.SerializeToString())

    x = np.random.default_rng(0).random((16, n_feat), dtype=np.float32)
    expected = model.predict_proba(x)[:, 1]
    _assert_parity(out, {"input": x}, expected, path.name, output_index=1,
                   column=1)
    return out


def _assert_parity(onnx_path, feeds, expected, name, output_index=0, column=None):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, feeds)[output_index]
    got = np.asarray(got)
    if column is not None:
        got = got[:, column]
    got = got.ravel()
    diff = float(np.abs(got - expected).max())
    if diff > RTOL:
        raise AssertionError(f"{name}: ONNX output diverges (max diff {diff:.2e})")
    print(f"  {name} -> {onnx_path.name}  (parity max diff {diff:.2e})")


def _effective_config(run_dir: Path) -> dict:
    """Load the config the run was actually trained with.

    train.py writes config_effective.json into the run directory after applying
    its CLI flags. Falling back to config.yaml is what made every ablation
    export a spec for the default feature set regardless of what was trained.
    """
    path = run_dir / "config_effective.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    print(f"  [warn] {run_dir.name} has no config_effective.json — falling "
          "back to config.yaml; the spec may not match the models")
    return load_config()


def feature_order(prep: dict) -> list[str]:
    """Rebuild the training feature order exactly as `preprocess.py` does.

    This must stay in lockstep with `src/data/preprocess.py`. A mismatch here
    is the single most dangerous failure mode in the whole project: the app
    would feed correctly-shaped nonsense into the model and get plausible
    numbers out, with no error anywhere.
    """
    columns = ["hand_detected"]
    if prep.get("pose_features", True):
        columns += (WORLD_COLUMNS if prep.get("pose_source", "image") == "world"
                    else LANDMARK_COLUMNS)
    if prep.get("add_motion", True):
        for label in ("tip", "wrist"):
            columns.append(f"mot_{label}_speed")
            columns += [f"mot_{label}_speed_m{w}" for w in MOTION_MEAN_WINDOWS]
            columns += [f"mot_{label}_speed_sd{w}" for w in MOTION_SD_WINDOWS]
    if prep.get("add_velocity"):
        columns += [f"d_{c}" for c in LANDMARK_COLUMNS]
    return columns


def _thresholds(run_dir: Path) -> dict:
    """Per-model decision thresholds fitted during the run.

    The exported weights are one fold's, but that fold's threshold is one
    draw from a distribution that measurably swings across participants — so
    the median across folds is published as the recommended operating point,
    with the exported fold's own value alongside for traceability.
    """
    summary = run_dir / "summary.json"
    if not summary.exists():
        return {}
    import statistics
    out = {}
    for agg in json.loads(summary.read_text()):
        window = agg.get("window")
        key = (f"{agg['dataset']}_{agg['model']}"
               + (f"_w{window['size']}" if window else ""))
        values = [f["threshold"] for f in agg.get("per_fold", [])
                  if f.get("threshold") is not None]
        if not values:
            continue
        out[key] = {
            "recommended": round(statistics.median(values), 4),
            "exported_fold": round(values[0], 4),
            "across_folds": {"min": round(min(values), 4),
                             "max": round(max(values), 4),
                             "n": len(values)},
            "f1_writing_mean": round(
                (agg.get("metrics", {}).get("f1_writing") or {}).get("mean", 0), 4),
        }
    return out


def write_spec(models_dir: Path, cfg: dict, converted: list[str], run_dir: Path):
    prep = cfg["preprocessing"]
    features = feature_order(prep)
    ev = cfg.get("evaluation", {})
    smoothing = dict(ev.get("smoothing") or {})

    steps = []
    causal = bool(prep.get("causal_motion", False))
    pose_source = prep.get("pose_source", "image")
    if prep.get("target_fps"):
        steps.append({
            "step": "0. resample to a fixed frame rate",
            "what": (f"run the pipeline on a {prep['target_fps']:g} Hz clock: at "
                     "each tick take the most recent MediaPipe result (sample "
                     "and hold) and use the tick time as its timestamp"),
            "why": ("training resampled every session onto this rate, so every "
                    "window length, rolling window and smoothing duration is a "
                    "fixed duration. A live stream at another rate (MediaPipe "
                    "alone caps near 25 fps) must be held onto the same clock, "
                    "or a 30-frame window stops meaning one second."),
        })
    if prep.get("aspect_correct", True):
        steps.append({
            "step": "1. aspect correction",
            "what": "multiply every landmark x by (frame_width / frame_height)",
            "why": ("MediaPipe x is a fraction of frame width and y a fraction "
                    "of frame height, so on any non-square frame the two axes "
                    "carry different scales. Training rescaled x into "
                    "image-height units; the app MUST do the same using its "
                    "own live capture resolution."),
        })
    if prep.get("time_normalize", True):
        steps.append({
            "step": "2. elapsed time per frame",
            "what": ("dt = seconds since the previous frame, from the capture "
                     "clock; clamp to [0.005, 0.200] s and substitute the "
                     "running median when outside that range"),
            "why": ("every speed below is per second, not per frame, so the "
                    "model behaves identically at 30 and 60 fps. Do NOT "
                    "assume a fixed frame interval."),
        })
    if prep.get("add_motion", True):
        steps.append({
            "step": "3. motion features",
            "what": (f"hand_scale = ||l9 - l0|| (wrist to middle-finger MCP) "
                     "on the aspect-corrected frame. For the index fingertip "
                     "(l8, prefix mot_tip) and the wrist (l0, prefix "
                     "mot_wrist): speed = ||p[t]/scale - p[t-1]/scale|| / dt, "
                     "in hand-widths per second. Then a "
                     + ("trailing" if causal else "centred")
                     + f" rolling mean over {list(MOTION_MEAN_WINDOWS)} frames "
                     "and a " + ("trailing" if causal else "centred")
                     + f" rolling standard deviation (ddof=1) over "
                     f"{list(MOTION_SD_WINDOWS)} frames, min_periods 1 for the "
                     "means and 2 for the deviations."),
            "why": ("air-writing is mostly whole-hand translation, which wrist "
                    "normalisation deletes. These are computed BEFORE it, on "
                    "un-normalised coordinates. A frame with no hand breaks "
                    "the difference (it does not span the gap) and contributes "
                    "speed 0."),
            "live_note": ("trailing windows, as in training — computable live."
                          if causal else
                          "the rolling windows are centred in training. A live "
                          "app cannot see the future, so it must either delay "
                          "output by half a window or use trailing windows and "
                          "accept the mismatch — measure which, and say which "
                          "in any reported result."),
        })
    if prep.get("normalize") == "wrist":
        pose = ("the WORLD landmarks (wl0..wl20, MediaPipe hand_world_landmarks)"
                if pose_source == "world" else "the image landmarks (l0..l20)")
        steps.append({
            "step": "4. wrist normalisation of the pose",
            "what": (f"on {pose}: subtract landmark 0 x/y/z from every "
                     "landmark, then divide all coordinates by the largest "
                     "distance of any landmark from the wrist (hand span). "
                     "All-zero frames stay zero."),
            "why": "removes where the hand is on screen and how large it is.",
        })
    if prep.get("mirror_left_hands", True):
        steps.append({
            "step": "5. canonical hand",
            "what": ("if MediaPipe's handedness label for the tracked hand is "
                     "'Left', negate the x of every wrist-normalised pose "
                     "coordinate. Motion features are NOT mirrored."),
            "why": ("left hands and mirrored front-camera sources were mapped "
                    "into right-hand form in training; speeds are unchanged "
                    "by a mirror, so only the pose is flipped."),
        })

    spec = {
        # v3: resampling, pose_source, canonical hand, window labelling. An
        # app built for v2 would compute different features — it must refuse
        "spec_version": 3,
        "generated_from_run": run_dir.name,
        "labels": {"0": "not_writing", "1": "writing"},
        "n_features": len(features),
        "feature_order": features,
        "feature_groups": {
            "pose": sum(1 for c in features
                        if c in LANDMARK_COLUMNS or c in WORLD_COLUMNS),
            "motion": sum(1 for c in features if c.startswith("mot_")),
            "velocity": sum(1 for c in features if c.startswith("d_")),
            "flags": ["hand_detected"],
        },
        "no_hand_frame": ("every feature is 0.0, hand_detected included. In "
                          "training P(writing | no hand) = 0.03 against 0.38 "
                          "when a hand is present, so a no-hand frame is very "
                          "nearly a free 'not writing' — do not treat a run of "
                          "them as confident evidence."),
        "preprocessing": {
            "pipeline": steps,
            "order_matters": ("apply the steps in the listed order; each "
                              "depends on the previous one"),
            "drop_undetected": prep.get("drop_undetected", False),
            "aspect_correct": prep.get("aspect_correct", True),
            "time_normalize": prep.get("time_normalize", True),
            "add_motion": prep.get("add_motion", True),
            "add_velocity": prep.get("add_velocity", False),
            "normalize": prep.get("normalize", "none"),
            "causal_motion": causal,
            "target_fps": prep.get("target_fps"),
            "pose_source": pose_source,
            "mirror_left_hands": prep.get("mirror_left_hands", True),
        },
        "decision_threshold": 0.5,
        "decision_threshold_note": (
            "0.5 is a placeholder and performed measurably worse than a fitted "
            "point. Use per_model_threshold[<model>].recommended below."),
        "per_model_threshold": _thresholds(run_dir),
        "smoothing": {
            **smoothing,
            "description": (
                "hysteresis then minimum durations: enter writing when p >= "
                "threshold + band/2, leave when p < threshold - band/2; then "
                "delete predicted runs shorter than min_on frames and fill "
                "gaps shorter than min_off. Measured to roughly halve "
                "fragmentation for frame-level models."),
            "applies_to": ("frame-level models (random_forest and the rule "
                           "baselines). Measured to HURT the recurrent models, "
                           "whose overlapping windows already smooth the "
                           "output — leave it off for lstm/gru."),
        },
        "recurrent_input": {
            "shape": "[1, window, n_features] float32, oldest frame first",
            "windows_trained": cfg["windows"],
            "output": "sigmoid probability of 'writing'",
            "window_label": ev.get("window_label", "last"),
            "window_label_note": (
                "'last': the output is the state of the NEWEST frame in the "
                "window — apply it to that frame, not to the window's middle."
                if ev.get("window_label", "last") == "last" else
                "'majority': the output describes the window as a whole, "
                "roughly its middle frame — half a window behind real time."),
        },
        "random_forest_input": {
            "shape": "[1, n_features] float32 (single frame)",
            "output": "outputs: [label, probabilities[not_writing, writing]]",
        },
        "models": converted,
    }
    out = models_dir / "inference_spec.json"
    out.write_text(json.dumps(spec, indent=2))
    print(f"  inference spec -> {out.name} "
          f"({len(features)} features, {len(spec['per_model_threshold'])} "
          "model thresholds)")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    models_dir = run_dir / "models"
    if not models_dir.is_dir():
        print(f"No models/ folder under {run_dir}")
        sys.exit(1)

    cfg = _effective_config(run_dir)
    converted, failed = [], []
    for path in sorted(models_dir.iterdir()):
        try:
            if path.suffix == ".keras":
                converted.append(convert_keras(path).name)
            elif path.suffix == ".joblib":
                converted.append(convert_forest(path).name)
        except Exception as e:
            failed.append(f"{path.name}: {e}")

    if converted:
        write_spec(models_dir, cfg, converted, run_dir)
    for f in failed:
        print(f"  [failed] {f}")
    print(f"ONNX export: {len(converted)} converted, {len(failed)} failed")
    sys.exit(1 if failed and not converted else 0)


if __name__ == "__main__":
    main()
