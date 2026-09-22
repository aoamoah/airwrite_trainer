"""Compare feature-set ablations that all share one corpus.

`merge_reports.py` keys a configuration on (dataset, model, window) and lets
the newest run win. Every ablation trains the same models on the same corpus
under the same name, so merging them there makes the newest ablation silently
supersede the baseline. This keeps the run as the column instead.

    venv/bin/python ablation_table.py                    # default set, below
    venv/bin/python ablation_table.py reports/run_A reports/run_B
"""

import argparse
import json
from pathlib import Path

from src.config import PROJECT_ROOT, load_config

METRICS = (("f1_writing", "F1(writing)"), ("average_precision", "AP"),
           ("roc_auc", "ROC AUC"), ("f2_writing", "F2(writing)"))
RULE_MODELS = ("velocity_threshold", "extension_threshold")
DEFAULT_RUNS = ("run_20260919_211002", "run_20260919_235812",
                "run_20260920_005512", "run_20260920_053128",
                "run_20260920_062153")


def label(run_dir: Path) -> str:
    """Name a run by how its effective preprocessing differs from config.yaml."""
    path = run_dir / "config_effective.json"
    if not path.is_file():
        return run_dir.name
    prep = json.loads(path.read_text())["preprocessing"]
    base = load_config()["preprocessing"]
    diff = [f"{k}={prep[k]}" for k in sorted(prep)
            if prep.get(k) != base.get(k)]
    return ", ".join(diff) if diff else "baseline"


def key(entry: dict) -> str:
    w = entry.get("window")
    if isinstance(w, dict):
        return f"{entry['model']} (w{w.get('size')}/s{w.get('stride')})"
    return entry["model"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", type=Path)
    p.add_argument("--out", type=Path,
                   default=PROJECT_ROOT / "reports" / "ablation_table.md")
    args = p.parse_args()

    runs = args.runs or [PROJECT_ROOT / "reports" / r for r in DEFAULT_RUNS]
    runs = [r for r in runs if (r / "summary.json").is_file()]
    if not runs:
        print("No runs with summary.json found.")
        return

    cols, data, widths = [], {}, {}
    for r in runs:
        name = label(r)
        cols.append(name)
        S = json.loads((r / "summary.json").read_text())
        data[name] = {key(e): e["metrics"] for e in S}
        spec = r / "models" / "inference_spec.json"
        widths[name] = (json.loads(spec.read_text())["n_features"]
                        if spec.is_file() else None)

    models = list(dict.fromkeys(m for c in cols for m in data[c]))
    learned = [m for m in models if not m.startswith(RULE_MODELS)]
    rules = [m for m in models if m.startswith(RULE_MODELS)]

    out = ["# Feature-set ablations", "",
           f"All on the `dataset` corpus, same folds, same seed. "
           f"Generated from {len(runs)} runs.", "",
           "| features | " + " | ".join(f"`{c}`" for c in cols) + " |",
           "|---|" + "---|" * len(cols),
           "| vector width | "
           + " | ".join(str(widths[c] or "?") for c in cols) + " |", ""]

    for metric, title in METRICS:
        out += [f"## {title}", "",
                "| estimator | " + " | ".join(f"`{c}`" for c in cols) + " |",
                "|---|" + "---|" * len(cols)]
        for group, members in (("Learned models", learned),
                               ("Rule baselines", rules)):
            if not members:
                continue
            out.append(f"| **{group}** |" + " |" * len(cols))
            for m in members:
                cells = []
                for c in cols:
                    st = data[c].get(m, {}).get(metric)
                    cells.append(f"{st['mean']*100:.2f}% ± {st['std']*100:.2f}"
                                 if st and metric != "roc_auc" else
                                 (f"{st['mean']:.3f}" if st else "—"))
                cells = [f"{c}" for c in cells]
                out.append(f"| {m} | " + " | ".join(cells) + " |")
        out.append("")

    args.out.write_text("\n".join(out))
    print(f"{len(models)} model(s) across {len(cols)} ablation(s) -> {args.out}")


if __name__ == "__main__":
    main()
