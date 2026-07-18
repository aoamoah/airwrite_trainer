"""AirWrite trainer — train and compare Random Forest / LSTM / GRU on the
collected dataset and any downloaded datasets, then export a full report.

Usage:
    python train.py                     # everything in config.yaml
    python train.py --quick             # smoke run (tiny epochs/forest)
    python train.py --datasets ours     # limit to named datasets
    python train.py --models random_forest lstm
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from src.config import PROJECT_ROOT, load_config, preload_cuda_libs, seed_everything

preload_cuda_libs()  # must run before anything imports TensorFlow
from src.data.loaders import load_dataset_folder
from src.training.runner import run_dataset
from src.report.builder import build_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--quick", action="store_true",
                        help="tiny run to check the pipeline end to end")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="only these dataset folders "
                             "(dataset / dataset_WITA / dataset_IPN)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="only these models (random_forest/lstm/gru)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.models:
        cfg["models"] = {k: v for k, v in cfg["models"].items() if k in args.models}
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
            ds = load_dataset_folder(folder, name=name)
            loaded[name] = ds
        except FileNotFoundError as e:
            print(f"[skip] {name}: {e}")

    if not loaded:
        print("No datasets found. Copy exported folders here: "
              + ", ".join(cfg["dataset_dirs"]))
        sys.exit(1)

    print(f"Datasets: {', '.join(loaded)}")
    print(f"Models: {', '.join(cfg['models'])}")

    # ---- train ----
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / cfg["report"]["out_dir"] / f"run_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results, all_details = [], {}
    for name, ds in loaded.items():
        print(f"\n=== {name}: {ds.meta['total_frames']:,} frames ===")
        results, details = run_dataset(ds, cfg, out_dir, quick=args.quick)
        fmt = lambda v: f"{v:.4f}" if v is not None else "n/a"
        for r in results:
            win = f" {r.window}" if r.window else ""
            print(f"  {r.model}{win}: test F1(writing)="
                  f"{fmt(r.test_metrics.get('f1_writing'))} "
                  f"acc={fmt(r.test_metrics.get('accuracy'))} "
                  f"({r.train_seconds:.1f}s)")
        all_results.extend(results)
        all_details[name] = details

    # ---- report ----
    report_path = build_report(cfg, loaded, all_details, all_results, out_dir)
    print(f"\nReport: {report_path}")
    print(f"Models + results.json in: {out_dir}")


if __name__ == "__main__":
    main()
