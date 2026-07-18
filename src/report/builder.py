"""Build the Markdown report — every configuration value, dataset fact,
split decision, hyperparameter, epoch, metric and timing of the run."""

import json
import platform
import sys
from datetime import datetime
from pathlib import Path

from src.report import figures
from src.training.runner import RunResult


def _pct(x):
    return f"{100 * x:.2f}%" if x is not None else "—"


def _num(x, digits=4):
    return f"{x:.{digits}f}" if x is not None else "—"


def _window_name(r: RunResult) -> str:
    return f"w{r.window['size']}/s{r.window['stride']}" if r.window else "frames"


def _env_section() -> str:
    import numpy, pandas, sklearn, matplotlib
    lines = [
        "## 1. Environment",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Platform: {platform.platform()}",
        f"- Python: {sys.version.split()[0]}",
        f"- numpy {numpy.__version__}, pandas {pandas.__version__}, "
        f"scikit-learn {sklearn.__version__}, matplotlib {matplotlib.__version__}",
    ]
    try:
        import tensorflow as tf
        lines.append(f"- tensorflow {tf.__version__} "
                     f"(GPUs visible: {len(tf.config.list_physical_devices('GPU'))})")
    except ImportError:
        pass
    return "\n".join(lines)


def _dataset_section(loaded: dict, details: dict, fig_dir: Path) -> str:
    lines = ["## 2. Datasets", ""]
    balances = {name: ds.meta.get("label_counts", {}) for name, ds in loaded.items()}
    fig = figures.class_balance_figure(balances, fig_dir / "class_balance.png")
    lines += [f"![Class balance](figures/{fig})", ""]

    for name, ds in loaded.items():
        lines += [f"### 2.{list(loaded).index(name) + 1} `{name}`", ""]
        for k, v in ds.meta.items():
            lines.append(f"- **{k}**: {v}")
        d = details.get(name, {})
        if d:
            lines += ["", "**Preprocessing applied:**"]
            for k, v in d.get("preprocessing", {}).items():
                lines.append(f"- {k}: {v}")
            lines.append(f"- final feature count: {d.get('feature_count')}")
            lines += ["", "**Split:**"]
            split = d.get("split", {})
            for k, v in split.items():
                if k == "units_per_split":
                    for s, units in v.items():
                        shown = ", ".join(units[:12]) + (" …" if len(units) > 12 else "")
                        lines.append(f"  - {s} ({len(units)} units): {shown}")
                else:
                    lines.append(f"- {k}: {v}")
            lines += ["", "**Class balance per split:**", "",
                      "| split | writing | not_writing |", "|---|---|---|"]
            for s, counts in d.get("class_balance_per_split", {}).items():
                lines.append(f"| {s} | {counts.get('writing', 0):,} "
                             f"| {counts.get('not_writing', 0):,} |")
        lines.append("")
    return "\n".join(lines)


def _config_section(cfg: dict) -> str:
    import yaml
    return "\n".join([
        "## 3. Run configuration",
        "",
        "The exact configuration used, verbatim:",
        "",
        "```yaml",
        yaml.safe_dump(cfg, sort_keys=False).strip(),
        "```",
    ])


def _run_section(results: list[RunResult], fig_dir: Path) -> str:
    lines = ["## 4. Training runs", ""]
    for i, r in enumerate(results, 1):
        tag = f"{r.dataset} — {r.model} ({_window_name(r)})"
        slug = f"{r.dataset}_{r.model}_{_window_name(r)}".replace("/", "-")
        lines += [f"### 4.{i} {tag}", ""]
        lines += [
            f"- Hyperparameters: `{json.dumps(r.extra.get('params', {}))}`",
            f"- Granularity: {r.extra.get('granularity')}",
            f"- Samples: train {r.n_train:,} / val {r.n_val:,} / test {r.n_test:,}",
            f"- Training time: {r.train_seconds:.1f}s",
        ]
        if r.epochs_trained is not None:
            lines.append(f"- Epochs trained: {r.epochs_trained} "
                         "(early stopping, best weights restored)")
        if r.extra.get("class_weight"):
            cw = {k: round(v, 3) for k, v in r.extra["class_weight"].items()}
            lines.append(f"- Class weights: {cw}")
        lines += ["", "| metric | train | val | test |", "|---|---|---|---|"]
        for key, label in [
            ("accuracy", "Accuracy"), ("precision_writing", "Precision (writing)"),
            ("recall_writing", "Recall (writing)"), ("f1_writing", "F1 (writing)"),
            ("f1_macro", "F1 (macro)"), ("roc_auc", "ROC AUC"),
        ]:
            row = [label]
            for m in (r.train_metrics, r.val_metrics, r.test_metrics):
                row.append(_num(m.get(key)) if m else "—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

        cm_fig = figures.confusion_matrix_figure(
            r.test_metrics["confusion_matrix"], f"{tag} — test confusion",
            fig_dir / f"cm_{slug}.png")
        lines += [f"![Confusion matrix](figures/{cm_fig})", ""]

        if r.history:
            tc_fig = figures.training_curves_figure(
                r.history, f"{tag} — training curves", fig_dir / f"curves_{slug}.png")
            lines += [f"![Training curves](figures/{tc_fig})", ""]

        if r.feature_importances:
            fi_fig = figures.feature_importance_figure(
                r.feature_importances, f"{tag} — top feature importances",
                fig_dir / f"importance_{slug}.png")
            lines += [f"![Feature importances](figures/{fi_fig})", ""]

        lines += ["<details><summary>Full classification report (test)</summary>", "",
                  "```", r.test_metrics["classification_report"].rstrip(), "```",
                  "</details>", ""]
    return "\n".join(lines)


def _comparison_section(results: list[RunResult], fig_dir: Path) -> str:
    lines = ["## 5. Comparison", ""]
    fig = figures.comparison_figure(results, "f1_writing", fig_dir / "comparison_f1.png")
    lines += [f"![Model comparison](figures/{fig})", ""]

    lines += ["| dataset | model | window | test acc | test F1 (writing) "
              "| test F1 (macro) | ROC AUC | train time |",
              "|---|---|---|---|---|---|---|---|"]
    ranked = sorted(results, key=lambda r: r.test_metrics.get("f1_writing") or 0,
                    reverse=True)
    for r in ranked:
        m = r.test_metrics
        lines.append(
            f"| {r.dataset} | {r.model} | {_window_name(r)} "
            f"| {_pct(m.get('accuracy'))} | {_pct(m.get('f1_writing'))} "
            f"| {_pct(m.get('f1_macro'))} | {_num(m.get('roc_auc'))} "
            f"| {r.train_seconds:.1f}s |"
        )

    best = ranked[0]
    lines += [
        "",
        "### Best performing configuration",
        "",
        f"**{best.model}** on **{best.dataset}** ({_window_name(best)}) — "
        f"test F1 (writing) {_pct(best.test_metrics.get('f1_writing'))}, "
        f"accuracy {_pct(best.test_metrics.get('accuracy'))}, "
        f"ROC AUC {_num(best.test_metrics.get('roc_auc'))}.",
        "",
        "_Ranking metric: F1 of the `writing` class on the held-out test split. "
        "F1 balances precision and recall, which matters here because the classes "
        "are imbalanced; accuracy alone would reward always predicting the "
        "majority class._",
    ]
    return "\n".join(lines)


def build_report(cfg: dict, loaded: dict, details: dict,
                 results: list[RunResult], out_dir: Path) -> Path:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    sections = [
        "# AirWrite — Model Training Report",
        "",
        "Writing-state detection (`writing` vs `not_writing`) from MediaPipe "
        "hand landmarks: Random Forest, LSTM and GRU compared across "
        f"{len(loaded)} dataset(s).",
        "",
        _env_section(), "",
        _dataset_section(loaded, details, fig_dir), "",
        _config_section(cfg), "",
        _run_section(results, fig_dir), "",
        _comparison_section(results, fig_dir), "",
    ]
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(sections))

    # Machine-readable copy of every number in the report
    results_json = [
        {k: v for k, v in vars(r).items() if k != "history"} | {"history": r.history}
        for r in results
    ]
    (out_dir / "results.json").write_text(json.dumps(results_json, indent=2, default=str))
    return report_path
