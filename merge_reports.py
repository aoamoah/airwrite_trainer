"""Merge several run directories into one comparison table.

A run only reports what was trained inside it, so a study split across runs
(one model per run, one corpus per run) has no single ranked table. This
reads the `summary.json` each run writes and produces one — Markdown for the
thesis, CSV for anything else.

    venv/bin/python merge_reports.py                       # every run in reports/
    venv/bin/python merge_reports.py reports/run_A reports/run_B
    venv/bin/python merge_reports.py --out chapter4_table.md --csv table.csv

Runs are matched on (dataset, model, window). If the same configuration
appears in two runs the newer one wins and the older is listed under
"superseded", so a re-run never silently doubles a row.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
METRICS = ("accuracy", "precision_writing", "recall_writing", "f1_writing",
           "f2_writing", "f1_macro", "average_precision", "roc_auc",
           "threshold", "predicted_positive_rate")
RANK_METRIC = "f1_writing"
# See src/report/builder.py — a rule baseline fits one threshold on one
# hand-crafted signal and exists to be a floor, not a competing architecture.
RULE_MODELS = ("velocity_threshold", "extension_threshold")


def _window_name(window) -> str:
    if not window:
        return "frames"
    return f"w{window['size']}/s{window['stride']}"


def _pm(stats: dict, key: str) -> str:
    s = stats.get(key)
    if not s or s.get("mean") is None:
        return "—"
    return f"{100 * s['mean']:.2f}% ± {100 * s.get('std', 0):.2f}"


def load_runs(paths: list[Path]) -> tuple[list[dict], list[dict]]:
    """Return (rows, superseded) — one row per unique configuration."""
    rows: dict[tuple, dict] = {}
    superseded = []
    for path in sorted(paths):
        summary_path = path / "summary.json"
        if not summary_path.exists():
            print(f"[skip] {path.name}: no summary.json "
                  "(run predates the aggregate reporting)")
            continue
        stamp = path.name.replace("run_", "")
        for agg in json.loads(summary_path.read_text()):
            key = (agg["dataset"], agg["model"], _window_name(agg["window"]))
            row = {
                "run": path.name,
                "stamp": stamp,
                "dataset": agg["dataset"],
                "model": agg["model"],
                "window": _window_name(agg["window"]),
                "config": (agg["model"] if not agg["window"]
                           else f"{agg['model']} ({_window_name(agg['window'])})"),
                "folds": agg["n_folds"],
                "metrics": agg.get("metrics", {}),
                "by_source": agg.get("by_source", {}),
                "events": agg.get("events", {}),
                "train_seconds": agg.get("train_seconds_total", 0.0),
            }
            if key in rows:
                older, newer = sorted([rows[key], row], key=lambda r: r["stamp"])
                superseded.append(older)
                rows[key] = newer
            else:
                rows[key] = row
    ordered = sorted(
        rows.values(),
        key=lambda r: (r["dataset"], -(r["metrics"].get("f1_writing", {}) or {}).get("mean", 0)),
    )
    return ordered, superseded


def _num(stats: dict | None) -> str:
    """Metrics added later in the project are simply absent from older runs;
    show that as an em dash rather than a NaN that reads like a failure."""
    value = (stats or {}).get("mean")
    return f"{value:.4f}" if value is not None and value == value else "—"


def _mean(row: dict, metric: str = RANK_METRIC) -> float:
    return (row["metrics"].get(metric) or {}).get("mean") or 0.0


def _rank_agreement(rows: list[dict]) -> list[str]:
    """Kendall tau between each pair of corpora's model rankings.

    The reason the table below is organised by model rather than by corpus,
    and the reason the corpus is still shown rather than averaged away: if
    source were a nuisance variable the rankings would agree, and they can be
    checked rather than assumed.
    """
    try:
        from scipy.stats import kendalltau
    except ImportError:
        return []
    corpora = sorted({r["dataset"] for r in rows})
    if len(corpora) < 2:
        return []
    scores = {c: {r["config"]: _mean(r) for r in rows if r["dataset"] == c}
              for c in corpora}
    lines = []
    for i, a in enumerate(corpora):
        for b in corpora[i + 1:]:
            shared = sorted(set(scores[a]) & set(scores[b]))
            if len(shared) < 3:
                continue
            tau, pval = kendalltau([scores[a][k] for k in shared],
                                   [scores[b][k] for k in shared])
            verdict = ("rankings largely agree" if tau > 0.4 else
                       "rankings are unrelated" if abs(tau) <= 0.4 else
                       "rankings are **inverted**")
            lines.append(f"- `{a}` vs `{b}` over {len(shared)} shared "
                         f"configurations: Kendall tau = **{tau:+.3f}** "
                         f"(p = {pval:.3f}) — {verdict}.")
    if lines:
        lines = [
            "",
            "## Does the corpus matter?",
            "",
            "Every corpus is extracted through the same MediaPipe pipeline "
            "into the same feature columns, so a model trained on one will "
            "run on another. That does not make the corpora interchangeable: "
            "identical features do not imply identical labels or identical "
            "difficulty. If the source were a nuisance variable, the model "
            "rankings would agree across corpora and only the level would "
            "move.",
            "",
        ] + lines + [
            "",
            "Where they disagree, a single pooled leaderboard ranks models by "
            "which corpus they were evaluated on, not by how well they detect "
            "writing — so the per-corpus columns above are the comparison, "
            "and `train.py --transfer A:B` measures the gap directly.",
        ]
    return lines


def markdown(rows: list[dict], superseded: list[dict], paths: list[Path]) -> str:
    corpora = sorted({r["dataset"] for r in rows})
    lines = [
        "# Merged comparison",
        "",
        f"Generated {datetime.now().isoformat(timespec='seconds')} from "
        f"{len(paths)} run director{'y' if len(paths) == 1 else 'ies'}. "
        "Every figure is the mean ± standard deviation over held-out "
        "participant folds.",
        "",
        "## Frame-level F1 (writing), by model",
        "",
        "The unit of comparison is the model; the corpus is a reported factor "
        "rather than a row, so the same model's scores sit side by side.",
        "",
        "| estimator | " + " | ".join(f"`{c}`" for c in corpora) + " |",
        "|" + "---|" * (len(corpora) + 1),
    ]
    cells: dict[str, dict[str, dict]] = {}
    for r in rows:
        cells.setdefault(r["config"], {})[r["dataset"]] = r

    for keep, title in (
            (lambda r: r["model"] not in RULE_MODELS, "**Learned models**"),
            (lambda r: r["model"] in RULE_MODELS,
             "**Rule baselines** — one hand-crafted signal, one fitted threshold")):
        group = [r for r in rows if keep(r)]
        if not group:
            continue
        best = {c: max((_mean(r) for r in group if r["dataset"] == c), default=None)
                for c in corpora}
        configs = sorted({r["config"] for r in group},
                         key=lambda k: -max(_mean(v) for v in cells[k].values()))
        lines.append(f"| {title} |" + " |" * len(corpora))
        for config in configs:
            row = [f"| {config} "]
            for c in corpora:
                r = cells[config].get(c)
                if r is None or best[c] is None:
                    row.append("| — ")
                    continue
                mark = " **←**" if _mean(r) == best[c] else ""
                row.append(f"| {_pm(r['metrics'], 'f1_writing')}{mark} ")
            lines.append("".join(row) + "|")
    lines += ["", "_**←** marks the best within its own group on that corpus. "
                  "The groups are not peers._"]

    lines += _rank_agreement(rows)

    lines += ["", "## Best per corpus", ""]
    for dataset in corpora:
        group = [r for r in rows if r["dataset"] == dataset]
        best_row = max(group, key=_mean)
        lines.append(f"- **`{dataset}`** — {best_row['config']} at "
                     f"{_pm(best_row['metrics'], 'f1_writing')} F1 (writing), "
                     f"{best_row['folds']} folds (`{best_row['run']}`).")
        rules = [r for r in group if r["model"].endswith("_threshold")]
        learned = [r for r in group if not r["model"].endswith("_threshold")]
        if not (rules and learned):
            continue
        best_rule = max(rules, key=_mean)
        best_learned = max(learned, key=_mean)
        delta = 100 * (_mean(best_learned) - _mean(best_rule))
        lines.append(
            f"  - Learned vs rule: best rule `{best_rule['model']}` at "
            f"{_pm(best_rule['metrics'], 'f1_writing')}; best learned "
            f"`{best_learned['config']}` at "
            f"{_pm(best_learned['metrics'], 'f1_writing')} — "
            f"a margin of {delta:+.2f} percentage points.")

    event_rows = [r for r in rows if r["events"]]
    if event_rows:
        lines += [
            "",
            "## Episode level",
            "",
            "Writing episodes matched one-to-one, which is the granularity "
            "the live app is judged on and the one the closest published work "
            "reports.",
            "",
            "| dataset | model | IoU F1 | onset F1 | fragments / episode |",
            "|---|---|---|---|---|",
        ]
        for r in sorted(event_rows, key=lambda r: (r["dataset"], -_mean(r))):
            ev = r["events"]
            iou = next((k for k in ev if k.startswith("iou@")), None)
            onset = next((k for k in ev if k.startswith("onset@")), None)
            cell = lambda k: (f"{100 * ev[k]['f1']['mean']:.1f}%"
                              if k and "f1" in (ev.get(k) or {}) else "—")
            frag = ev.get("mean_fragments_per_true_event")
            lines.append(f"| {r['dataset']} | {r['config']} | {cell(iou)} "
                         f"| {cell(onset)} "
                         f"| {frag['mean']:.2f} |" if frag else
                         f"| {r['dataset']} | {r['config']} | {cell(iou)} "
                         f"| {cell(onset)} | — |")

    by_source_rows = [r for r in rows if r["by_source"]]
    if by_source_rows:
        lines += [
            "",
            "## Combined-corpus breakdown",
            "",
            "How a model trained on the pooled corpus scored on each source "
            "corpus separately.",
            "",
            "| dataset | model | window | source | folds | accuracy "
            "| F1 (writing) |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in by_source_rows:
            for src, stats in r["by_source"].items():
                lines.append(
                    f"| {r['dataset']} | {r['model']} | {r['window']} | {src} "
                    f"| {stats.get('folds', 0)} | {_pm(stats, 'accuracy')} "
                    f"| {_pm(stats, 'f1_writing')} |")

    lines += [
        "",
        "<details><summary>Full table (every configuration, every metric)</summary>",
        "",
        "| dataset | model | window | folds | accuracy | F1 (writing) | F2 "
        "| F1 (macro) | AP | ROC AUC | run |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: (r["dataset"], -_mean(r))):
        auc = r["metrics"].get("roc_auc") or {}
        ap = r["metrics"].get("average_precision") or {}
        lines.append(
            f"| {r['dataset']} | {r['model']} | {r['window']} | {r['folds']} "
            f"| {_pm(r['metrics'], 'accuracy')} | {_pm(r['metrics'], 'f1_writing')} "
            f"| {_pm(r['metrics'], 'f2_writing')} "
            f"| {_pm(r['metrics'], 'f1_macro')} "
            f"| {_num(ap)} | {_num(auc)} | `{r['run']}` |")
    lines += ["", "</details>"]

    if superseded:
        lines += ["", "## Superseded", "",
                  "Older runs of a configuration that also appears in a newer "
                  "run; the newer numbers are the ones tabled above.", ""]
        for r in superseded:
            lines.append(f"- `{r['run']}` — {r['dataset']} / {r['model']} "
                         f"/ {r['window']}, F1 {_pm(r['metrics'], 'f1_writing')}")
    return "\n".join(lines) + "\n"


def write_csv(rows: list[dict], path: Path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "dataset", "model", "window", "folds"]
                   + [f"{m}_{s}" for m in METRICS for s in ("mean", "std")])
        for r in rows:
            out = [r["run"], r["dataset"], r["model"], r["window"], r["folds"]]
            for m in METRICS:
                stats = r["metrics"].get(m) or {}
                out += [stats.get("mean", ""), stats.get("std", "")]
            w.writerow(out)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", type=Path,
                   help="run directories (default: every reports/run_* )")
    p.add_argument("--out", type=Path, default=None,
                   help="write Markdown here (default: reports/merged_comparison.md)")
    p.add_argument("--csv", type=Path, default=None, help="also write a CSV")
    args = p.parse_args()

    paths = args.runs or sorted((PROJECT_ROOT / "reports").glob("run_*"))
    paths = [p for p in paths if p.is_dir()]
    if not paths:
        print("No run directories found.")
        return

    rows, superseded = load_runs(paths)
    if not rows:
        print("No summary.json files found in the given runs.")
        return

    out = args.out or PROJECT_ROOT / "reports" / "merged_comparison.md"
    out.write_text(markdown(rows, superseded, paths))
    print(f"{len(rows)} configuration(s) from {len(paths)} run(s) -> {out}")
    if args.csv:
        write_csv(rows, args.csv)
        print(f"CSV -> {args.csv}")


if __name__ == "__main__":
    main()
