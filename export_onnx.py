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
from src.data.loaders import LANDMARK_COLUMNS

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


def write_spec(models_dir: Path, cfg: dict, converted: list[str]):
    prep = cfg["preprocessing"]
    features = ["hand_detected"] + LANDMARK_COLUMNS
    if prep.get("add_velocity"):
        features = features + [f"d_{c}" for c in LANDMARK_COLUMNS]
    spec = {
        "labels": {"0": "not_writing", "1": "writing"},
        "decision_threshold": 0.5,
        "feature_order": features,
        "preprocessing": {
            "normalize": prep.get("normalize", "none"),
            "normalize_description": (
                "Per frame: subtract wrist (l0) x/y/z from every landmark, "
                "then divide all coordinates by the maximum distance of any "
                "landmark from the wrist (hand span). All-zero frames (no "
                "hand) stay zero." if prep.get("normalize") == "wrist"
                else "none"),
            "drop_undetected": prep.get("drop_undetected", False),
            "add_velocity": prep.get("add_velocity", False),
        },
        "recurrent_input": {
            "shape": "[1, window, n_features] float32, oldest frame first",
            "windows_trained": cfg["windows"],
            "output": "sigmoid probability of 'writing'",
        },
        "random_forest_input": {
            "shape": "[1, n_features] float32 (single frame)",
            "output": "outputs: [label, probabilities[not_writing, writing]]",
        },
        "models": converted,
    }
    out = models_dir / "inference_spec.json"
    out.write_text(json.dumps(spec, indent=2))
    print(f"  inference spec -> {out.name}")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    models_dir = run_dir / "models"
    if not models_dir.is_dir():
        print(f"No models/ folder under {run_dir}")
        sys.exit(1)

    cfg = load_config()
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
        write_spec(models_dir, cfg, converted)
    for f in failed:
        print(f"  [failed] {f}")
    print(f"ONNX export: {len(converted)} converted, {len(failed)} failed")
    sys.exit(1 if failed and not converted else 0)


if __name__ == "__main__":
    main()
