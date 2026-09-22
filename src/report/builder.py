"""Build the Markdown report — every configuration value, dataset fact,
fold decision, hyperparameter, epoch, metric and timing of the run.

Under cross-validation the headline for a configuration is mean ± std over
held-out participants, with the per-fold numbers kept underneath: the spread
across participants is the finding, not noise to be averaged away.
"""

import json
import platform
import statistics
import sys
from datetime import datetime
from pathlib import Path

from src.report import figures
from src.training.aggregate import aggregate_results, Aggregate
from src.training.runner import RunResult

RANK_METRIC = "f1_writing"

# Rule baselines are not peers of the learned models and should not be ranked
# in one undifferentiated list. Each fits a single threshold on one
# hand-crafted signal — a 1-feature model whose job is to be a floor, so that
# "learning helps" is a measured margin rather than an assumption. Keeping
# them visually separate also stops a reader concluding that a forest lost to
# "a rule" when what actually happened is documented below the table.
RULE_MODELS = ("velocity_threshold", "extension_threshold")


def _is_rule(a) -> bool:
    return a.model in RULE_MODELS


def _pct(x):
    return f"{100 * x:.2f}%" if x is not None else "—"


def _num(x, digits=4):
    return f"{x:.{digits}f}" if x is not None else "—"


def _pm(agg: Aggregate, metric: str) -> str:
    """mean ± std, in percent."""
    m, s = agg.mean(metric), agg.std(metric)
    if m is None:
        return "—"
    return f"{100 * m:.2f}% ± {100 * (s or 0):.2f}"


def _always_writing(aggregates: list[Aggregate], corpus: str) -> dict | None:
    """Scores of a detector that answers "writing" on every frame, per fold.

    With the writing share p of a test fold, precision is p and recall 1, so
    F1 = 2p/(1+p) and F2 = 5p/(4p+1); accuracy and average precision are p.
    This is the floor that F1 (writing) has to be read against: on a corpus
    that is mostly writing it is high without any model. Priors come from the
    frame-level results, whose folds and scored frames every model shares.
    """
    priors = next(
        ([f["test_writing_prior"] for f in a.per_fold
          if f.get("test_writing_prior") is not None]
         for a in aggregates if a.dataset == corpus and a.window is None
         and a.per_fold), [])
    if not priors:
        return None

    def stats(values):
        return {"mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0}
    return {
        "f1_writing": stats([2 * p / (1 + p) for p in priors]),
        "f2_writing": stats([5 * p / (4 * p + 1) for p in priors]),
        "accuracy": stats(priors),
        "average_precision": stats(priors),
        "n_folds": len(priors),
    }


def _pm_stats(block: dict | None) -> str:
    if not block:
        return "—"
    return f"{100 * block['mean']:.2f}% ± {100 * block['std']:.2f}"


def _window_name(r) -> str:
    w = r.window
    return f"w{w['size']}/s{w['stride']}" if w else "frames"


def _slug(agg: Aggregate) -> str:
    return f"{agg.dataset}_{agg.model}_{_window_name(agg)}".replace("/", "-")


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
    lines = ["## 2. Datasets and evaluation design", ""]
    balances = {name: ds.meta.get("label_counts", {}) for name, ds in loaded.items()}
    fig = figures.class_balance_figure(balances, fig_dir / "class_balance.png")
    lines += [f"![Class balance](figures/{fig})", ""]

    lines += _granularity_table(loaded)

    for name, ds in loaded.items():
        lines += [f"### 2.{list(loaded).index(name) + 1} `{name}`", ""]
        for k, v in ds.meta.items():
            if k == "label_granularity":
                continue          # tabulated above
            lines.append(f"- **{k}**: {v}")
        d = details.get(name, {})
        if not d:
            lines.append("")
            continue

        lines += ["", "**Preprocessing applied:**"]
        for k, v in d.get("preprocessing", {}).items():
            lines.append(f"- {k}: {v}")
        lines.append(f"- final feature count: {d.get('feature_count')}")

        lines += ["", "**Evaluation design:**"]
        design = d.get("evaluation", {})
        for k, v in design.items():
            lines.append(f"- {k}: {v}")

        folds = d.get("folds", [])
        if design.get("scheme") == "holdout" and folds:
            lines += ["", "**Split:**"]
            for k, v in folds[0].items():
                if isinstance(v, list) and k != "participants_in_multiple_splits":
                    shown = ", ".join(str(x) for x in v[:12]) + (" …" if len(v) > 12 else "")
                    lines.append(f"- {k}: {shown}")
                elif k != "fold":
                    lines.append(f"- {k}: {v}")
            leaked = folds[0].get("participants_in_multiple_splits") or []
            if leaked:
                lines += [
                    "",
                    f"> **Participant leakage:** {len(leaked)} participant(s) "
                    f"appear in more than one split — {', '.join(leaked)} — "
                    f"covering "
                    f"{_pct(folds[0].get('leaked_participant_frame_share'))} of "
                    "frames. Test scores from this split measure recall of a "
                    "hand the model has already trained on, and are reported "
                    "here as a leakage ablation only.",
                ]
        elif folds:
            lines += [
                "",
                "**Folds** — no participant appears in more than one split of "
                "a fold, and the validation participants are drawn from the "
                "training pool, so the test fold is never seen during model "
                "selection.",
                "",
                "| fold | test participant(s) | train / val / test frames "
                "| writing prior train / val / test |",
                "|---|---|---|---|",
            ]
            for f in folds:
                test = ", ".join(f.get("test_participants", []))
                lines.append(
                    f"| {f['fold']} | {test} "
                    f"| {f.get('train_frames', 0):,} / {f.get('val_frames', 0):,} "
                    f"/ {f.get('test_frames', 0):,} "
                    f"| {_pct(f.get('train_writing_prior'))} / "
                    f"{_pct(f.get('val_writing_prior'))} / "
                    f"{_pct(f.get('test_writing_prior'))} |"
                )
            gaps = [abs(f["val_writing_prior"] - f["train_writing_prior"])
                    for f in folds
                    if f.get("val_writing_prior") is not None
                    and f.get("train_writing_prior") is not None]
            if gaps:
                lines += [
                    "",
                    f"_Largest train↔val writing-prior gap across folds: "
                    f"{100 * max(gaps):.2f} percentage points "
                    f"(median {100 * statistics.median(gaps):.2f})._",
                ]
            notes = [f"`{f['fold']}`: {f['val_fallback']}"
                     for f in folds if f.get("val_fallback")]
            if notes:
                lines += ["", "**Validation fallbacks:** " + "; ".join(notes)]
        lines.append("")
    return "\n".join(lines)


def _granularity_table(loaded: dict) -> list[str]:
    """What each corpus's labels mark. Pooling corpora whose pauses differ by
    an order of magnitude trains on two different definitions of the task."""
    rows = [(name, ds.meta.get("label_granularity")) for name, ds in loaded.items()
            if ds.meta.get("label_granularity")]
    if not rows:
        return []
    ms = lambda d, k: f"{d[k]['p50']:,} ms" if d.get(k) else "—"
    lines = [
        "**Label granularity.** The task is writing state *including pauses "
        "inside a letter*, so a corpus only teaches it if its labels mark such "
        "pauses. A pause here is a not_writing run with writing on both sides. "
        "Check a new source against this table before pooling it.",
        "",
        "| corpus | writing runs | median writing run | interior pauses "
        "| median pause | pauses < 167 ms | pauses < 400 ms | unsure frames |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, g in rows:
        lines.append(
            f"| {name} | {g['writing_runs']:,} | {ms(g, 'writing_run_ms')} "
            f"| {g['interior_pauses']:,} | {ms(g, 'interior_pause_ms')} "
            f"| {_pct(g.get('pauses_under_167ms'))} | {_pct(g.get('pauses_under_400ms'))} "
            f"| {g.get('unsure_frames', 0):,} |")
    return lines + [""]


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


def _results_section(aggregates: list[Aggregate], results: list[RunResult],
                     fig_dir: Path) -> str:
    cross_validated = any(a.n_folds > 1 for a in aggregates)
    heading = ("## 4. Results per configuration"
               + (" (cross-validated)" if cross_validated else ""))
    lines = [heading, ""]
    if cross_validated:
        lines += [
            "Each configuration is trained once per fold. Metrics below are "
            "mean ± standard deviation over folds; the confusion matrix pools "
            "every fold's held-out predictions, so it covers the whole cohort "
            "exactly once.",
            "",
        ]

    by_fold = {}
    for r in results:
        by_fold.setdefault((r.dataset, r.model, _window_name(r)), []).append(r)

    for i, agg in enumerate(aggregates, 1):
        tag = f"{agg.dataset} — {agg.model} ({_window_name(agg)})"
        runs = by_fold[(agg.dataset, agg.model, _window_name(agg))]
        first = runs[0]
        lines += [f"### 4.{i} {tag}", ""]
        lines += [
            f"- Hyperparameters: `{json.dumps(first.extra.get('params', {}))}`",
            f"- Granularity: {first.extra.get('granularity')}",
            f"- Folds: {agg.n_folds}",
            f"- Total training time: {agg.train_seconds_total:.1f}s",
        ]
        if first.extra.get("threshold_fitted_on"):
            lines.append(f"- Operating point: {first.extra['threshold_fitted_on']}"
                         + (f" (mean threshold "
                            f"{_num(agg.mean('threshold'), 3)})"
                            if agg.mean("threshold") is not None else ""))
        if first.extra.get("rule"):
            lines.append(f"- Rule: {first.extra['rule']} "
                         f"({first.extra.get('fitted_on')})")
        if agg.epochs_trained:
            lines.append(
                f"- Epochs trained: min {min(agg.epochs_trained)}, "
                f"median {statistics.median(agg.epochs_trained):.0f}, "
                f"max {max(agg.epochs_trained)} (early stopping, best weights "
                "restored)")

        lines += ["", "| metric | test mean ± std | min | max |",
                  "|---|---|---|---|"]
        for key, label in [
            ("accuracy", "Accuracy"), ("precision_writing", "Precision (writing)"),
            ("recall_writing", "Recall (writing)"), ("f1_writing", "F1 (writing)"),
            ("f2_writing", "F2 (writing)"), ("f1_macro", "F1 (macro)"),
            ("average_precision", "Average precision"), ("roc_auc", "ROC AUC"),
            ("positive_rate", "Writing prior (test)"),
            ("predicted_positive_rate", "Predicted writing rate"),
        ]:
            s = agg.metrics.get(key)
            if not s:
                lines.append(f"| {label} | — | — | — |")
                continue
            lines.append(f"| {label} | {_pm(agg, key)} | {_num(s['min'])} "
                         f"| {_num(s['max'])} |")
        lines.append("")

        if agg.events:
            crit = next((k for k in agg.events if k.startswith("iou@")), None)
            onset = next((k for k in agg.events if k.startswith("onset@")), None)
            lines += ["**Episode level** — writing episodes matched one-to-one "
                      "against the ground truth:", "",
                      "| criterion | precision | recall | F1 | F2 |",
                      "|---|---|---|---|---|"]
            for key in (c for c in (crit, onset) if c):
                block = agg.events.get(key, {})
                cell = lambda m: (f"{100 * block[m]['mean']:.1f}% ± "
                                  f"{100 * block[m]['std']:.1f}"
                                  if m in block else "—")
                lines.append(f"| {key} | {cell('precision')} | {cell('recall')} "
                             f"| {cell('f1')} | {cell('f2')} |")
            scalar = lambda k, d=2: (f"{agg.events[k]['mean']:.{d}f}"
                                     if k in agg.events else "—")
            lines += [
                "",
                f"- Median onset lag: {scalar('median_onset_error_frames', 0)} "
                "frames on matched episodes",
                f"- Predicted fragments per true episode: "
                f"{scalar('mean_fragments_per_true_event')} "
                "(1.0 is one clean detection per episode)",
                f"- True episodes missed entirely: "
                f"{scalar('true_events_missed_entirely', 1)} per fold, of "
                f"{scalar('n_true_events', 1)}",
                "",
            ]

        if agg.by_detection:
            lines += ["**By hand detection** — how the score splits between "
                      "frames where MediaPipe found a hand and frames where it "
                      "did not:", "",
                      "| frames | n | writing prior | accuracy | F1 (writing) |",
                      "|---|---|---|---|---|"]
            for state, st in agg.by_detection.items():
                cell = lambda m: (f"{100 * st[m]['mean']:.2f}%"
                                  if m in st else "—")
                lines.append(f"| {state} | {st.get('n_test', 0):,} "
                             f"| {cell('positive_rate')} | {cell('accuracy')} "
                             f"| {cell('f1_writing')} |")
            lines.append("")

        if agg.by_source:
            lines += [
                "**Per source corpus** — the same pooled model, scored on "
                "each corpus's held-out participants separately:",
                "",
                "| source | folds | test n | accuracy | precision | recall "
                "| F1 (writing) |",
                "|---|---|---|---|---|---|---|",
            ]
            for src, s in agg.by_source.items():
                cell = lambda k: (
                    f"{100 * s[k]['mean']:.2f}% ± {100 * s[k]['std']:.2f}"
                    if k in s else "—")
                lines.append(
                    f"| {src} | {s.get('folds', 0)} | {s.get('n_test', 0):,} "
                    f"| {cell('accuracy')} | {cell('precision_writing')} "
                    f"| {cell('recall_writing')} | {cell('f1_writing')} |")
            lines.append("")

        cm_fig = figures.confusion_matrix_figure(
            agg.pooled_confusion,
            f"{tag} — pooled held-out confusion" if agg.n_folds > 1
            else f"{tag} — test confusion",
            fig_dir / f"cm_{_slug(agg)}.png")
        lines += [f"![Confusion matrix](figures/{cm_fig})", ""]

        if agg.n_folds > 1:
            lines += ["<details><summary>Per-fold results</summary>", "",
                      "| fold | test | n | writing prior | acc | precision "
                      "| recall | F1 (writing) | ROC AUC |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for f in agg.per_fold:
                lines.append(
                    f"| {f['fold']} | {', '.join(f['test_participants'])} "
                    f"| {f['n_test']:,} | {_pct(f['test_writing_prior'])} "
                    f"| {_num(f['accuracy'])} | {_num(f['precision_writing'])} "
                    f"| {_num(f['recall_writing'])} | {_num(f['f1_writing'])} "
                    f"| {_num(f['roc_auc'])} |")
            lines += ["", "</details>", ""]

        curve_run = next((r for r in runs if r.history), None)
        if curve_run:
            tc_fig = figures.training_curves_figure(
                curve_run.history,
                f"{tag} — training curves ({curve_run.fold})",
                fig_dir / f"curves_{_slug(agg)}.png")
            lines += [f"![Training curves](figures/{tc_fig})", ""]

        imp_run = next((r for r in runs if r.feature_importances), None)
        if imp_run:
            fi_fig = figures.feature_importance_figure(
                imp_run.feature_importances,
                f"{tag} — top feature importances ({imp_run.fold})",
                fig_dir / f"importance_{_slug(agg)}.png")
            lines += [f"![Feature importances](figures/{fi_fig})", ""]

        lines += ["<details><summary>Classification report "
                  f"({first.fold})</summary>", "",
                  "```", first.test_metrics["classification_report"].rstrip(),
                  "```", "</details>", ""]
    return "\n".join(lines)


def _config_label(a: Aggregate) -> str:
    """Model plus window — the unit being compared, without the corpus."""
    return a.model if a.window is None else f"{a.model} ({_window_name(a)})"


def _rank_agreement(aggregates: list[Aggregate]) -> tuple[str | None, list[str]]:
    """Kendall tau between two corpora's model rankings.

    This is the number that settles whether the data source can be treated as
    a nuisance variable. Extracting identical features from every corpus does
    make the *pipeline* source-agnostic, but identical features do not imply
    identical labels or identical difficulty. If source were irrelevant the
    ranking of models would be stable across corpora and only the level would
    move. A negative tau means the best model on one corpus is among the worst
    on the other, and no single pooled leaderboard can be read as "which model
    detects writing best" — it would rank models by which corpus they happened
    to be evaluated on.
    """
    from scipy.stats import kendalltau

    corpora = sorted({a.dataset for a in aggregates})
    if len(corpora) < 2:
        return None, []
    scores = {c: {_config_label(a): a.mean(RANK_METRIC)
                  for a in aggregates
                  if a.dataset == c and a.mean(RANK_METRIC) is not None}
              for c in corpora}
    lines = []
    reported = None
    for i, a_name in enumerate(corpora):
        for b_name in corpora[i + 1:]:
            shared = sorted(set(scores[a_name]) & set(scores[b_name]))
            if len(shared) < 3:
                continue
            tau, pval = kendalltau([scores[a_name][k] for k in shared],
                                   [scores[b_name][k] for k in shared])
            reported = reported or f"{tau:+.2f}"
            verdict = ("rankings largely agree" if tau > 0.4 else
                       "rankings are unrelated" if abs(tau) <= 0.4 else
                       "rankings are **inverted**")
            lines.append(
                f"- `{a_name}` vs `{b_name}` over {len(shared)} shared "
                f"configurations: Kendall tau = **{tau:+.3f}** "
                f"(p = {pval:.3f}) — {verdict}.")
    return reported, lines


def _corpus_matrix(aggregates: list[Aggregate], metric: str) -> list[str]:
    """One row per model configuration, one column per corpus."""
    corpora = sorted({a.dataset for a in aggregates})
    cells: dict[str, dict[str, Aggregate]] = {}
    for a in aggregates:
        cells.setdefault(_config_label(a), {})[a.dataset] = a

    header = "| estimator | " + " | ".join(f"`{c}`" for c in corpora) + " |"
    lines = [header, "|" + "---|" * (len(corpora) + 1)]

    # Best within each kind, so the marker means "best learned model" and
    # "best baseline" rather than mixing the two
    def best_of(rows):
        return {c: max((a.mean(metric) or 0)
                       for a in rows if a.dataset == c) if
                any(a.dataset == c for a in rows) else None
                for c in corpora}

    for kind, keep, title in (
            ("learned", lambda a: not _is_rule(a), "**Learned models**"),
            ("rule", _is_rule, "**Rule baselines** — one hand-crafted signal "
                               "plus one fitted threshold")):
        rows = [a for a in aggregates if keep(a)]
        if not rows:
            continue
        best = best_of(rows)
        labels = sorted({_config_label(a) for a in rows},
                        key=lambda k: -max((v.mean(metric) or 0)
                                           for v in cells[k].values()))
        lines.append(f"| {title} |" + " |" * len(corpora))
        for label in labels:
            row = [f"| {label} "]
            for c in corpora:
                a = cells[label].get(c)
                if a is None or best[c] is None:
                    row.append("| — ")
                    continue
                mark = " **←**" if (a.mean(metric) or 0) == best[c] else ""
                row.append(f"| {_pm(a, metric)}{mark} ")
            lines.append("".join(row) + "|")

    trivial = {c: _always_writing(aggregates, c) for c in corpora}
    if any(t and metric in t for t in trivial.values()):
        lines.append("| **No model** — the reference every score is read "
                     "against |" + " |" * len(corpora))
        lines.append("| always writing " + "".join(
            f"| {_pm_stats((trivial[c] or {}).get(metric))} " for c in corpora) + "|")
    return lines


def _events_table(aggregates: list[Aggregate]) -> list[str]:
    """Episode-level scores, which is what the live app is judged on."""
    with_events = [a for a in aggregates if a.events]
    if not with_events:
        return []
    crit = next((k for k in with_events[0].events if k.startswith("iou@")), None)
    onset = next((k for k in with_events[0].events if k.startswith("onset@")), None)
    if not crit:
        return []

    def cell(a, key, metric):
        stats = (a.events.get(key) or {}).get(metric)
        return f"{100 * stats['mean']:.1f}%" if stats else "—"

    def scalar(a, key, digits=1):
        stats = a.events.get(key)
        return f"{stats['mean']:.{digits}f}" if stats else "—"

    lines = [
        "",
        "### Episode-level results",
        "",
        "Frame F1 asks what share of frames were labelled correctly. A user "
        "does not experience frames — they experience whether the detector "
        "noticed that they started writing, and whether it held on until they "
        "stopped. A model can score a respectable frame F1 while shattering "
        "every episode into flickering fragments, and that is what a live app "
        "feels like when it 'keeps cutting out'. Episodes are matched "
        "one-to-one, so a prediction spanning a whole session cannot detect "
        "everything at once.",
        "",
        f"| dataset | model | {crit} F1 | {crit} recall | "
        + (f"{onset} F1 | " if onset else "")
        + "onset lag (frames) | fragments / episode | episodes missed |",
        "|---|---|---|---|" + ("---|" if onset else "") + "---|---|---|",
    ]
    for a in sorted(with_events,
                    key=lambda a: -((a.events.get(crit, {}).get("f1") or {}).get("mean", 0))):
        lines.append(
            f"| {a.dataset} | {_config_label(a)} | {cell(a, crit, 'f1')} "
            f"| {cell(a, crit, 'recall')} | "
            + (f"{cell(a, onset, 'f1')} | " if onset else "")
            + f"{scalar(a, 'median_onset_error_frames', 0)} "
            f"| {scalar(a, 'mean_fragments_per_true_event', 2)} "
            f"| {scalar(a, 'true_events_missed_entirely', 1)} |")
    return lines


# Pen-In-Air States from Video (arXiv 2606.02342), Table III: pause (pen-up)
# event F2 under leave-one-video-out, at tolerances of 5 / 10 / 12 frames
PEN_IN_AIR_F2 = {
    "LightGBM (their best)": (0.757, 0.792, 0.805),
    "Random Forest": (0.721, 0.753, 0.760),
}


def _count(stats: dict | None) -> str:
    return f"{stats['mean']:.0f}" if stats else "—"


def _pause_table(aggregates: list[Aggregate]) -> list[str]:
    """Pause detection under the pen-in-air protocol, beside its numbers."""
    rows = [a for a in aggregates if any(k.startswith("pause@") for k in a.events)]
    if not rows:
        return []
    tols = sorted({k for a in rows for k in a.events if k.startswith("pause@")},
                  key=lambda k: int(k[6:-1]))

    def cell(block, key, metric):
        stats = (block.get(key) or {}).get(metric)
        return f"{stats['mean']:.3f}" if stats else "—"

    lines = [
        "",
        "### Pause detection (pen-in-air protocol)",
        "",
        "Detecting writing state *mid-letter* means detecting the pauses "
        "inside writing. This table scores exactly that, with the protocol of "
        "the closest published work (arXiv 2606.02342): an event is a pause "
        "with writing on both sides, a prediction matches it when both "
        "boundaries fall within the tolerance, matching is one-to-one, and F2 "
        "weights recall. Their rows are theirs — pen on paper, five videos, "
        "leave-one-video-out — so they locate this work rather than rank it.",
        "",
        "| dataset | model | " + " | ".join(f"F2 {t[6:]}" for t in tols)
        + " | recall " + tols[-1][6:] + " | precision " + tols[-1][6:]
        + " | pauses / fold | F2 " + tols[-1][6:] + " smoothed |",
        "|---|---|" + "---|" * (len(tols) + 4),
    ]
    for a in sorted(rows, key=lambda a: -((a.events.get(tols[-1]) or {})
                                          .get("f2") or {}).get("mean", 0)):
        n = a.events.get("n_true_pauses")
        lines.append(
            f"| {a.dataset} | {_config_label(a)} | "
            + " | ".join(cell(a.events, t, "f2") for t in tols)
            + f" | {cell(a.events, tols[-1], 'recall')} "
            f"| {cell(a.events, tols[-1], 'precision')} "
            f"| {_count(n)} "
            f"| {cell(a.smoothed_events or {}, tols[-1], 'f2')} |")
    for name, f2 in PEN_IN_AIR_F2.items():
        if len(tols) == 3:
            lines.append(f"| *pen-in-air (published)* | *{name}* | "
                         + " | ".join(f"*{v:.3f}*" for v in f2)
                         + " | — | — | — | — |")
    return lines


def _segmental_table(aggregates: list[Aggregate]) -> list[str]:
    """Edit and segmental F1@k — the temporal action segmentation standard."""
    rows = [a for a in aggregates if "edit_score" in a.events]
    if not rows:
        return []

    def cell(block, key):
        stats = (block or {}).get(key)
        return f"{100 * stats['mean']:.1f}" if stats else "—"

    lines = [
        "",
        "### Segmentation quality (action-segmentation metrics)",
        "",
        "Edit score and segmental F1@{10,25,50}, as reported throughout "
        "temporal action segmentation (MS-TCN; online: OnlineTAS, NeurIPS "
        "2024). Frame accuracy can stay high while these collapse — OnlineTAS "
        "reports 56.7% accuracy at 9.3% F1@50 on Breakfast — and that gap is "
        "the over-segmentation a live user sees as flicker.",
        "",
        "| dataset | model | frame acc | Edit | F1@10 | F1@25 | F1@50 "
        "| Edit smoothed | F1@50 smoothed |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for a in sorted(rows, key=lambda a: -(a.events.get("seg_f1@50") or {}).get("mean", 0)):
        acc = a.metrics.get("accuracy", {}).get("mean")
        lines.append(
            f"| {a.dataset} | {_config_label(a)} "
            f"| {f'{100 * acc:.1f}' if acc is not None else '—'} "
            f"| {cell(a.events, 'edit_score')} | {cell(a.events, 'seg_f1@10')} "
            f"| {cell(a.events, 'seg_f1@25')} | {cell(a.events, 'seg_f1@50')} "
            f"| {cell(a.smoothed_events, 'edit_score')} "
            f"| {cell(a.smoothed_events, 'seg_f1@50')} |")
    return lines


def _smoothing_table(aggregates: list[Aggregate]) -> list[str]:
    """What temporal post-processing buys, per model.

    The models emit an independent decision per frame, so nothing stops a
    score hovering near the threshold from flipping every few frames. The
    difference between these two columns is the share of the live "it keeps
    cutting out" problem that is temporal consistency rather than a failure
    to discriminate — and it is bought with a state machine, not a better
    model.
    """
    rows = [a for a in aggregates if a.smoothed_events]
    if not rows:
        return []
    lines = [
        "",
        "### What temporal smoothing buys",
        "",
        "Hysteresis (a band around the fitted threshold instead of a single "
        "point) followed by minimum episode and gap durations. Parameters are "
        "set on training data and travel to the inference app in "
        "`inference_spec.json`, so the app applies a measured configuration "
        "rather than hand-tuned constants.",
        "",
        "| dataset | model | frame F1 raw → smoothed | episode F1 raw → "
        "smoothed | fragments raw → smoothed |",
        "|---|---|---|---|---|",
    ]
    for a in rows:
        iou = next((k for k in a.events if k.startswith("iou@")), None)
        iou_s = next((k for k in a.smoothed_events if k.startswith("iou@")), None)
        raw_f1 = a.mean("f1_writing")
        sm_f1 = (a.smoothed.get("f1_writing") or {}).get("mean")
        raw_ep = ((a.events.get(iou) or {}).get("f1") or {}).get("mean")
        sm_ep = ((a.smoothed_events.get(iou_s) or {}).get("f1") or {}).get("mean")
        raw_fr = (a.events.get("mean_fragments_per_true_event") or {}).get("mean")
        sm_fr = (a.smoothed_events.get("mean_fragments_per_true_event") or {}).get("mean")
        pair = lambda x, y, d=1, scale=100: (
            f"{scale * x:.{d}f} → {scale * y:.{d}f}"
            if x is not None and y is not None else "—")
        lines.append(
            f"| {a.dataset} | {_config_label(a)} "
            f"| {pair(raw_f1, sm_f1)}% | {pair(raw_ep, sm_ep)}% "
            f"| {pair(raw_fr, sm_fr, 2, 1)} |")
    return lines


def _detection_table(aggregates: list[Aggregate]) -> list[str]:
    """How much of each score comes from frames with no hand in them."""
    with_split = [a for a in aggregates if a.by_detection]
    if not with_split:
        return []
    lines = [
        "",
        "### Where the score comes from: hand detected versus no hand",
        "",
        "12.1% of frames in `dataset` have no hand detected, and only 3.0% of "
        "those are labelled writing, against 38.1% of the frames where a hand "
        "is visible. \"No hand\" is therefore very nearly a free correct "
        "answer, and a model can lean on it instead of learning writing. This "
        "matters beyond bookkeeping: the collector extracts landmarks with "
        "inter-frame tracking (`RunningMode.VIDEO`) that the live app cannot "
        "match, so the app sees a *lower* detection rate than training did. "
        "Any score propped up by these frames will not survive deployment.",
        "",
        "| dataset | model | frames w/ hand — F1 | frames w/o hand — F1 "
        "| writing prior w/o hand |",
        "|---|---|---|---|---|",
    ]
    def cell(a, state, metric):
        stats = (a.by_detection.get(state) or {}).get(metric)
        return f"{100 * stats['mean']:.1f}%" if stats else "—"
    for a in with_split:
        lines.append(
            f"| {a.dataset} | {_config_label(a)} "
            f"| {cell(a, 'hand_detected', 'f1_writing')} "
            f"| {cell(a, 'no_hand', 'f1_writing')} "
            f"| {cell(a, 'no_hand', 'positive_rate')} |")
    return lines


def _benchmark_section() -> str:
    """External work these numbers can be read against.

    There is no established benchmark for this exact task — writing versus
    not-writing, from RGB hand pose, evaluated subject-independently. That
    absence is worth stating plainly rather than papering over with a loosely
    related figure, and it is part of what makes the task worth reporting.
    """
    return "\n".join([
        "## 6. External benchmarks",
        "",
        "No published benchmark covers this exact task: binary writing-state "
        "detection from RGB hand pose under subject-independent evaluation. "
        "The four comparisons below are the closest available, and none is "
        "like-for-like — the caveat column says why. They are context for the "
        "numbers above, not a scoreboard the numbers can be slotted into.",
        "",
        "| work | task | protocol | headline | why it is not like-for-like |",
        "|---|---|---|---|---|",
        "| Pen-In-Air States from Video (arXiv 2606.02342, 2026) | pen-up vs "
        "pen-down from video | Leave-One-Video-Out, 5 videos, 13,507 frames, "
        "147 kinematic features | **pause (pen-up) event F2 = 0.805** "
        "(LightGBM, 12-frame tolerance, recall 0.880); 0.792 at 10, 0.757 at "
        "5 | scored with the same protocol in the pause table above; but "
        "folds are videos, not participants; pen-tip tracking rather than "
        "hand pose; handwriting on paper, not air-writing; data not public |",
        "| — its end-to-end deep baselines | same | same | CNN acc 0.630 / F2 "
        "0.250; CNN-LSTM 0.573 / 0.264; 3D-CNN 0.786 / 0.481 | as above — but "
        "note kinematic features beat end-to-end deep learning there too |",
        "| Amma, Georgi & Schultz, Airwriting (2014) | writing vs non-writing "
        "*spotting* | realistic data incl. everyday activity | recall **99%**, "
        "precision **25%**; person-independent recognition error 11.0% | "
        "wrist-worn IMU, not video; spotting feeds a recogniser that filters "
        "false positives downstream |",
        "| OnlineTAS (Zhong et al., NeurIPS 2024) | online temporal action "
        "segmentation | standard splits of GTEA / 50Salads / Breakfast | "
        "causal TCN on 50Salads: acc 75.2, Edit 19.6, F1@50 19.6; with a GRU "
        "and post-processing Edit 69.2, F1@50 62.8 | multi-class kitchen "
        "activities from I3D features; cited for the metric definitions and "
        "the online over-segmentation effect, not for a comparable score |",
        "| Emporio et al., CVIU 2025 (continuous hand gesture survey) | "
        "online gesture detection | 12 benchmarks reviewed | SHREC'22 best: "
        "detection rate 92%, FPR 9%, JI 85%, 8-frame delay; a binary GRU "
        "localiser (Two-Model) 85% / 9% / 78% / 5 frames | gesture, not "
        "writing; its finding that no two benchmarks share metrics is why "
        "this report gives frame, episode, pause, segmental and delay "
        "figures side by side |",
        "| IPN Hand (Benitez-Garcia et al., ICPR 2021) | continuous gesture "
        "recognition | official split | isolated 86.32% (ResNeXt-101, "
        "RGB-Flow); continuous Levenshtein accuracy 42.47% | Levenshtein "
        "accuracy over gesture sequences is not frame or episode F1; 13 "
        "gesture classes, not writing |",
        "",
        "Two things follow for the write-up. First, the fingertip-speed rule "
        "is not a strawman — velocity-thresholded \"virtual pen-up/pen-down\" "
        "segmentation is standard published prior art in air-writing, so a "
        "learned model losing to it is a real result about feature design "
        "rather than an artifact of a weak baseline. Second, Amma et al. "
        "reaching 25% precision on the spotting subtask is the clearest "
        "evidence that this subtask is genuinely hard, and it is the right "
        "context for reading the frame F1 figures above.",
    ])


def _comparison_section(aggregates: list[Aggregate], fig_dir: Path) -> str:
    """Model-major comparison.

    The unit of comparison is the model. The corpus is kept as a reported
    factor rather than collapsed away, because the rank-agreement statistic
    below decides whether collapsing it would be sound — and on this data it
    is not.
    """
    lines = ["## 5. Comparison", ""]
    fig = figures.comparison_figure(aggregates, RANK_METRIC,
                                    fig_dir / "comparison_f1.png")
    lines += [f"![Model comparison](figures/{fig})", ""]

    spread_fig = figures.fold_spread_figure(
        aggregates, RANK_METRIC, fig_dir / "fold_spread_f1.png")
    if spread_fig:
        lines += [f"![Per-fold spread](figures/{spread_fig})", ""]

    corpora = sorted({a.dataset for a in aggregates})
    lines += ["### Frame-level F1 (writing) by model", ""]
    lines += _corpus_matrix(aggregates, RANK_METRIC)
    lines += [
        "",
        "_**←** marks the best in its own group on that corpus. The two "
        "groups are not peers: a rule baseline fits one threshold on one "
        "hand-crafted signal, and exists to be a floor the learned models "
        "have to clear, not a competing architecture._",
        "",
        "_**always writing** is no model at all: it answers writing on every "
        "frame, and its F1 is 2p/(1+p) for a fold whose writing share is p. "
        "F1 of the writing class rewards it because recall is perfect, so on a "
        "corpus that is mostly writing it sits close to the models. Read every "
        "F1 above as a margin over this row. Average precision (whose floor is "
        "p) and the episode and segmental tables below separate the models "
        "far better. Window-level models are scored per window, whose writing "
        "share differs slightly from the per-frame one used here._",
        "",
        "> **Read the margin with care.** The rule baselines take their "
        "signal from *raw* landmarks, because wrist normalisation removes the "
        "very translation fingertip speed measures. For most of this "
        "project's history the learned models saw only wrist-normalised pose, "
        "so a rule beating them was not a finding about rules versus "
        "learning — it was the only estimator with access to the signal "
        "winning. The motion features close that gap, and the margin is only "
        "interpretable as \"learning helps\" now that both sides can see "
        "hand movement.",
        "",
    ]

    # ---- does the corpus matter? ----
    tau, tau_lines = _rank_agreement(aggregates)
    if tau_lines:
        lines += [
            "### Does the data source matter?",
            "",
            "Every corpus goes through the same MediaPipe extraction and "
            "arrives with the same feature columns, so the pipeline is "
            "genuinely source-agnostic and a model trained on one corpus will "
            "*run* on another. It does not follow that the corpora are "
            "interchangeable: identical features do not imply identical "
            "labels or identical difficulty. If source were a nuisance "
            "variable, model rankings would agree across corpora and only the "
            "level would shift.",
            "",
            *tau_lines,
            "",
            "Where the rankings disagree, a single pooled leaderboard cannot "
            "be read as \"which model detects writing best\" — it ranks "
            "models by which corpus they were evaluated on. The per-corpus "
            "winners are reported separately below for that reason, and the "
            "cross-corpus transfer arm (`--transfer A:B`) measures the gap "
            "directly rather than assuming it away.",
            "",
        ]

    # ---- per-corpus winners and the rule-vs-learned margin ----
    lines += ["### Best configuration per corpus", ""]
    for corpus in corpora:
        group = [a for a in aggregates if a.dataset == corpus]
        ranked = sorted(group, key=lambda a: a.mean(RANK_METRIC) or 0, reverse=True)
        if not ranked:
            continue
        best = ranked[0]
        lines.append(
            f"- **`{corpus}`** — best is **{_config_label(best)}** at "
            f"{_pm(best, 'f1_writing')} F1 (writing), "
            f"accuracy {_pm(best, 'accuracy')}, "
            f"AP {_num(best.mean('average_precision'))}"
            + (f", over {best.n_folds} held-out folds." if best.n_folds > 1 else "."))
        rules = [a for a in ranked if _is_rule(a)]
        learned = [a for a in ranked if not _is_rule(a)]
        if rules and learned:
            delta = ((learned[0].mean(RANK_METRIC) or 0)
                     - (rules[0].mean(RANK_METRIC) or 0))
            lines.append(
                f"  - Learned vs rule (Objective 2): best rule "
                f"`{rules[0].model}` {_pm(rules[0], 'f1_writing')}; best "
                f"learned `{_config_label(learned[0])}` "
                f"{_pm(learned[0], 'f1_writing')} — margin "
                f"**{100 * delta:+.2f}** points.")
        trivial = _always_writing(aggregates, corpus)
        if trivial:
            floor = trivial["f1_writing"]["mean"]
            parts = [f"`{_config_label(a)}` **{100 * ((a.mean(RANK_METRIC) or 0) - floor):+.2f}**"
                     for a in (rules[:1] + learned[:1])]
            lines.append(
                f"  - Against always writing ({_pm_stats(trivial['f1_writing'])} "
                f"F1, average precision {trivial['average_precision']['mean']:.4f}): "
                + ", ".join(parts) + " points of F1.")
    lines.append("")

    lines += _events_table(aggregates)
    lines += _pause_table(aggregates)
    lines += _segmental_table(aggregates)
    lines += _smoothing_table(aggregates)
    lines += _detection_table(aggregates)

    # ---- the full table, kept for reference under the model-major view ----
    lines += [
        "",
        "<details><summary>Full ranked table (every configuration)</summary>",
        "",
        "| dataset | model | window | folds | acc | F1 (writing) | F2 "
        "| F1 (macro) | AP | ROC AUC | threshold | pred. positive rate "
        "| train time |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for a in sorted(aggregates, key=lambda a: a.mean(RANK_METRIC) or 0,
                    reverse=True):
        lines.append(
            f"| {a.dataset} | {a.model} | {_window_name(a)} | {a.n_folds} "
            f"| {_pm(a, 'accuracy')} | {_pm(a, 'f1_writing')} "
            f"| {_pm(a, 'f2_writing')} | {_pm(a, 'f1_macro')} "
            f"| {_num(a.mean('average_precision'))} "
            f"| {_num(a.mean('roc_auc'))} "
            f"| {_num(a.mean('threshold'), 3)} "
            f"| {_pct(a.mean('predicted_positive_rate'))} "
            f"| {a.train_seconds_total:.1f}s |")
    lines += ["", "</details>", ""]

    lines += [
        "_Ranking metric: F1 of the `writing` class on held-out participants. "
        "Every model's decision threshold is fitted on validation data and "
        "applied to test, so the rule baselines and the learned models choose "
        "their operating point on the same terms — previously only the rules "
        "had a fitted threshold while the learned models were scored at a flat "
        "0.5. Average precision is reported alongside ROC AUC because its "
        "baseline is the class prior rather than 0.5, which is the honest "
        "reference on an imbalanced problem. Frame-level models are scored per "
        "frame and window-level models per window, so the two granularities "
        "are not strictly interchangeable — the episode-level table above is "
        "the comparison that holds across both._",
    ]
    return "\n".join(lines)


def build_report(cfg: dict, loaded: dict, details: dict,
                 results: list[RunResult], out_dir: Path) -> Path:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    aggregates = aggregate_results(results)

    sections = [
        "# AirWrite — Model Training Report",
        "",
        "Writing-state detection (`writing` vs `not_writing`) from MediaPipe "
        "hand landmarks: rule baselines, Random Forest, LSTM and GRU compared "
        f"across {len(loaded)} dataset(s).",
        "",
        _env_section(), "",
        _dataset_section(loaded, details, fig_dir), "",
        _config_section(cfg), "",
        _results_section(aggregates, results, fig_dir), "",
        _comparison_section(aggregates, fig_dir), "",
        _benchmark_section(), "",
    ]
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(sections))

    # Machine-readable copy of every number in the report
    results_json = [
        {k: v for k, v in vars(r).items() if k != "history"} | {"history": r.history}
        for r in results
    ]
    (out_dir / "results.json").write_text(json.dumps(results_json, indent=2, default=str))
    (out_dir / "summary.json").write_text(json.dumps(
        [vars(a) for a in aggregates], indent=2, default=str))
    (out_dir / "folds.json").write_text(json.dumps(
        {name: d.get("folds", []) for name, d in details.items()},
        indent=2, default=str))
    return report_path
