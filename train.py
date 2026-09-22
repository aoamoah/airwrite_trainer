"""AirWrite trainer — train and compare Random Forest / LSTM / GRU on the
collected dataset and any downloaded datasets, then export a full report.

Usage:
    python train.py                     # everything in config.yaml
    python train.py --quick             # smoke run (tiny epochs/forest)
    python train.py --datasets ours     # limit to named datasets
    python train.py --models random_forest lstm
    python train.py --scheme holdout    # the session-split leakage ablation
    python train.py --velocity          # add landmark deltas to the features
    python train.py --no-motion         # wrist-normalised pose only (ablation)
    python train.py --motion-only       # motion features without the pose
    python train.py --transfer dataset_IPN:dataset   # train on one, test on another
    python train.py --pose-source world  # MediaPipe world landmarks as the pose
    python train.py --window-label majority          # the old window labelling (ablation)
    python train.py --target-fps 0       # keep every session's native frame rate
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from src.config import PROJECT_ROOT, load_config, preload_cuda_libs, seed_everything

preload_cuda_libs()  # must run before anything imports TensorFlow
from src.data.loaders import (LoadedDataset, combine_datasets,
                              load_dataset_folder)
from src.training.aggregate import aggregate_results
from src.training.runner import run_dataset, run_transfer
from src.report.builder import build_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--quick", action="store_true",
                        help="tiny run to check the pipeline end to end")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="only these dataset folders "
                             "(dataset / dataset_<Source>)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="only these models (velocity_threshold/"
                             "extension_threshold/random_forest/lstm/gru)")
    parser.add_argument("--scheme", default=None,
                        choices=["auto", "lopo", "group_kfold", "holdout"],
                        help="override evaluation.scheme; holdout runs the "
                             "single session-level split (leakage ablation)")
    parser.add_argument("--max-folds", type=int, default=None,
                        help="cap the number of folds (debugging)")
    parser.add_argument("--combine", action="store_true",
                        help="pool the selected datasets into one training "
                             "corpus (folds still hold out whole participants)")
    parser.add_argument("--also-separate", action="store_true",
                        help="with --combine, also train each corpus on its own")
    parser.add_argument("--velocity", dest="velocity", action="store_true",
                        default=None, help="add frame-to-frame landmark deltas")
    parser.add_argument("--no-velocity", dest="velocity", action="store_false",
                        help="drop the landmark deltas")
    parser.add_argument("--no-motion", dest="motion", action="store_false",
                        default=None,
                        help="drop the motion features — the "
                             "'wrist-normalised pose only' ablation")
    parser.add_argument("--motion-only", action="store_true",
                        help="motion features without the 63 pose columns")
    parser.add_argument("--no-aspect", dest="aspect", action="store_false",
                        default=None,
                        help="skip the width/height correction of x")
    parser.add_argument("--no-fit-threshold", dest="fit_threshold",
                        action="store_false", default=None,
                        help="score every model at 0.5 instead of fitting the "
                             "operating point on validation")
    parser.add_argument("--causal-motion", dest="causal", action="store_true",
                        default=None,
                        help="use trailing instead of centred rolling windows "
                             "for the motion features, so training matches "
                             "what a live app can compute")
    parser.add_argument("--pose-source", choices=["image", "world"], default=None,
                        help="pose from image landmarks (default) or MediaPipe "
                             "world landmarks, which are metric and independent "
                             "of resolution and aspect")
    parser.add_argument("--no-mirror", dest="mirror", action="store_false",
                        default=None,
                        help="do not mirror Left-labelled poses into right-hand form")
    parser.add_argument("--window-label", choices=["last", "majority"], default=None,
                        help="label a window by its last frame (default) or by "
                             "majority vote (the older behaviour)")
    parser.add_argument("--target-fps", type=float, default=None,
                        help="resample every session to this frame rate "
                             "(0 keeps native rates)")
    parser.add_argument("--exclude", nargs="*", default=None, metavar="P###",
                        help="drop these participants before training; the "
                             "exclusion is recorded in the report")
    parser.add_argument("--transfer", metavar="TRAIN:TEST", default=None,
                        help="cross-corpus transfer — train on the first "
                             "corpus, evaluate on the second "
                             "(e.g. dataset_IPN:dataset)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.models:
        cfg["models"] = {k: v for k, v in cfg["models"].items() if k in args.models}
    cfg.setdefault("evaluation", {})
    if args.scheme:
        cfg["evaluation"]["scheme"] = args.scheme
    if args.max_folds is not None:
        cfg["evaluation"]["max_folds"] = args.max_folds
    if args.velocity is not None:
        cfg["preprocessing"]["add_velocity"] = args.velocity
    if args.motion is not None:
        cfg["preprocessing"]["add_motion"] = args.motion
    if args.motion_only:
        cfg["preprocessing"]["add_motion"] = True
        cfg["preprocessing"]["pose_features"] = False
    if args.causal is not None:
        cfg["preprocessing"]["causal_motion"] = args.causal
    if args.aspect is not None:
        cfg["preprocessing"]["aspect_correct"] = args.aspect
    if args.fit_threshold is not None:
        cfg["evaluation"]["fit_threshold"] = args.fit_threshold
    if args.pose_source:
        cfg["preprocessing"]["pose_source"] = args.pose_source
    if args.mirror is not None:
        cfg["preprocessing"]["mirror_left_hands"] = args.mirror
    if args.window_label:
        cfg["evaluation"]["window_label"] = args.window_label
    if args.target_fps is not None:
        cfg["preprocessing"]["target_fps"] = args.target_fps or None
    seed_everything(cfg["seed"])

    # ---- load datasets ----
    loaded = {}
    wanted = args.datasets or cfg["dataset_dirs"]
    for name in wanted:
        folder = PROJECT_ROOT / name
        if not folder.is_dir():
            print(f"[skip] {name}: folder not present")
            continue
        try:
            excluded = set(args.exclude or cfg.get("exclude_participants") or ())
            # A corpus with participants removed is a different corpus, so it
            # gets a different name — otherwise the merged table would silently
            # supersede the full-cohort numbers instead of sitting beside them
            ds_name = f"{name}_excl{len(excluded)}p" if excluded else name
            ds = load_dataset_folder(folder, name=ds_name,
                                     exclude_participants=excluded,
                                     target_fps=cfg["preprocessing"].get("target_fps"))
            loaded[ds_name] = ds
            if excluded:
                print(f"[{ds_name}] excluded {', '.join(sorted(excluded))} — "
                      f"{len(ds.meta['participants'])} participants remain")
        except FileNotFoundError as e:
            print(f"[skip] {name}: {e}")

    if not loaded:
        print("No datasets found. Copy exported folders here: "
              + ", ".join(cfg["dataset_dirs"]))
        sys.exit(1)

    transfer_pair = None
    if args.transfer:
        if ":" not in args.transfer:
            print("--transfer expects TRAIN:TEST, e.g. dataset_IPN:dataset")
            sys.exit(1)
        train_name, test_name = args.transfer.split(":", 1)
        missing = [n for n in (train_name, test_name) if n not in loaded]
        if missing:
            print(f"--transfer: corpus not loaded: {', '.join(missing)} "
                  f"(loaded: {', '.join(loaded) or 'none'})")
            sys.exit(1)
        if train_name == test_name:
            print("--transfer: train and test corpora must differ")
            sys.exit(1)
        transfer_pair = (loaded[train_name], loaded[test_name])

    if args.combine:
        if len(loaded) < 2:
            print("--combine needs at least two datasets present; "
                  f"only found: {', '.join(loaded) or 'none'}")
            sys.exit(1)
        pooled = combine_datasets(loaded)
        print(f"Combined corpus: {pooled.meta['total_frames']:,} frames, "
              f"{len(pooled.meta['participants'])} participants "
              f"({pooled.meta['id_namespacing']})")
        loaded = ({**loaded, pooled.name: pooled} if args.also_separate
                  else {pooled.name: pooled})

    print(f"Datasets: {', '.join(loaded)}")
    print(f"Models: {', '.join(cfg['models'])}")

    # ---- train ----
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / cfg["report"]["out_dir"] / f"run_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # The ONNX/spec export runs in a subprocess and reloads config.yaml, which
    # loses every CLI flag applied above — that is how the ablation runs came
    # to ship specs describing the default 76-feature set regardless of what
    # was trained. Persist the effective config so the run is self-describing.
    (out_dir / "config_effective.json").write_text(
        json.dumps(cfg, indent=2, default=str), encoding="utf-8")

    all_results, all_details = [], {}

    if transfer_pair is not None:
        train_ds, test_ds = transfer_pair
        name = f"{train_ds.name}->{test_ds.name}"
        print(f"\n=== transfer: train {train_ds.name} "
              f"({train_ds.meta['total_frames']:,} frames) -> "
              f"test {test_ds.name} ({test_ds.meta['total_frames']:,}) ===")
        results, details = run_transfer(train_ds, test_ds, cfg, out_dir,
                                        quick=args.quick)
        for agg in aggregate_results(results):
            win = (f" w{agg.window['size']}/s{agg.window['stride']}"
                   if agg.window else " frames")
            f1 = agg.metrics.get("f1_writing")
            print(f"  {agg.model}{win}: test F1(writing)="
                  + (f"{f1['mean']:.4f}" if f1 else "n/a"))
        all_results.extend(results)
        all_details[name] = details
        # The report's dataset section reads `meta` only, so the transfer arm
        # is described rather than carrying a second copy of both corpora
        loaded = {name: LoadedDataset(
            name, train_ds.frames.head(0), train_ds.feature_columns,
            {"type": f"cross-corpus transfer — trained on {train_ds.name}, "
                     f"evaluated on {test_ds.name}",
             "train_corpus": train_ds.name,
             "test_corpus": test_ds.name,
             "train_frames": train_ds.meta["total_frames"],
             "test_frames": test_ds.meta["total_frames"],
             "train_participants": len(train_ds.meta["participants"]),
             "test_participants": len(test_ds.meta["participants"]),
             "total_frames": (train_ds.meta["total_frames"]
                              + test_ds.meta["total_frames"]),
             "label_counts": {
                 k: train_ds.meta["label_counts"].get(k, 0)
                    + test_ds.meta["label_counts"].get(k, 0)
                 for k in set(train_ds.meta["label_counts"])
                          | set(test_ds.meta["label_counts"])}})}

    for name, ds in ({} if transfer_pair is not None else loaded).items():
        print(f"\n=== {name}: {ds.meta['total_frames']:,} frames ===")
        results, details = run_dataset(ds, cfg, out_dir, quick=args.quick)
        design = details.get("evaluation", {})
        print(f"  evaluation: {design.get('scheme')} "
              f"({design.get('folds')} fold(s))")
        # One line per configuration, not per fold — with LOPO the per-fold
        # log is hundreds of lines and the fold spread is what matters
        for agg in aggregate_results(results):
            win = (f" w{agg.window['size']}/s{agg.window['stride']}"
                   if agg.window else " frames")
            f1, acc = agg.metrics.get("f1_writing"), agg.metrics.get("accuracy")
            fmt = lambda s: (f"{s['mean']:.4f}±{s['std']:.4f}" if s else "n/a")
            print(f"  {agg.model}{win}: test F1(writing)={fmt(f1)} "
                  f"acc={fmt(acc)} "
                  f"({agg.n_folds} folds, {agg.train_seconds_total:.1f}s)")
        all_results.extend(results)
        all_details[name] = details

    # ---- report ----
    report_path = build_report(cfg, loaded, all_details, all_results, out_dir)
    print(f"\nReport: {report_path}")
    print(f"Models + results.json in: {out_dir}")

    # ---- ONNX export for C++ inference apps ----
    # Runs in a subprocess with the GPU hidden: tracing on GPU bakes
    # CudnnRNN ops into the graph, which ONNX cannot represent.
    import os
    import subprocess
    print("\nExporting ONNX models…")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "export_onnx.py"), str(out_dir)],
        env=env, capture_output=True, text=True,
    )
    for line in proc.stdout.splitlines():
        if line.strip():
            print(line)
    if proc.returncode != 0:
        print("[warn] ONNX export failed — run manually: "
              f"CUDA_VISIBLE_DEVICES= python export_onnx.py {out_dir}")


if __name__ == "__main__":
    main()
