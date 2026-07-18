"""Report figures. Palette and mark conventions follow the validated
reference palette (fixed categorical order, sequential single-hue ramp,
recessive chrome); every figure has a companion table in the report, so
color never carries information alone."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Categorical slots in fixed order — model identity keeps its color everywhere
MODEL_COLORS = {
    "random_forest": "#2a78d6",   # blue
    "lstm": "#008300",            # green
    "gru": "#e87ba4",             # magenta
}
TRAIN_COLOR, VAL_COLOR = "#2a78d6", "#008300"

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "font.size": 10,
})


def _save(fig, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path.name


def class_balance_figure(balances: dict[str, dict], path: Path) -> str:
    """balances: {dataset: {"writing": n, "not_writing": n}}"""
    datasets = list(balances)
    writing = [balances[d].get("writing", 0) for d in datasets]
    not_writing = [balances[d].get("not_writing", 0) for d in datasets]

    x = np.arange(len(datasets))
    width = 0.32
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for offset, vals, label, color in [
        (-width / 2, writing, "writing", "#2a78d6"),
        (width / 2, not_writing, "not_writing", "#008300"),
    ]:
        bars = ax.bar(x + offset, vals, width * 0.94, label=label, color=color)
        ax.bar_label(bars, fmt="{:,.0f}", fontsize=8, color=INK, padding=2)
    ax.set_xticks(x, datasets)
    ax.set_ylabel("frames")
    ax.set_title("Class balance per dataset", color=INK)
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    return _save(fig, path)


def training_curves_figure(history: dict, title: str, path: Path) -> str:
    epochs = np.arange(1, len(history.get("loss", [])) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

    for ax, metric, label in [(axes[0], "loss", "loss"),
                              (axes[1], "accuracy", "accuracy")]:
        ax.plot(epochs, history.get(metric, []), color=TRAIN_COLOR,
                linewidth=2, label="train")
        val = history.get(f"val_{metric}")
        if val:
            ax.plot(epochs, val, color=VAL_COLOR, linewidth=2, label="validation")
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        ax.legend(frameon=False)
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    return _save(fig, path)


def confusion_matrix_figure(cm: list, title: str, path: Path) -> str:
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap=SEQ_BLUE, vmin=0)
    labels = ["not_writing", "writing"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title, color=INK, fontsize=10)
    ax.grid(visible=False)
    vmax = cm.max() or 1
    for i in range(2):
        for j in range(2):
            ink = "#ffffff" if cm[i, j] > 0.55 * vmax else INK
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color=ink, fontsize=11)
    return _save(fig, path)


def feature_importance_figure(importances: list, title: str, path: Path) -> str:
    names = [n for n, _ in importances][::-1]
    vals = [v for _, v in importances][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(names) + 1.2))
    ax.barh(np.arange(len(names)), vals, height=0.62, color="#2a78d6")
    ax.set_yticks(np.arange(len(names)), names, fontsize=8)
    ax.set_xlabel("importance")
    ax.set_title(title, color=INK)
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def comparison_figure(results: list, metric: str, path: Path) -> str:
    """Grouped bars: best test <metric> per model, grouped by dataset.
    For recurrent models the best window config is shown."""
    datasets = sorted({r.dataset for r in results})
    models = [m for m in MODEL_COLORS if any(r.model == m for r in results)]

    best = {}
    for r in results:
        v = r.test_metrics.get(metric)
        if v is None:
            continue
        key = (r.dataset, r.model)
        if key not in best or v > best[key][0]:
            best[key] = (v, r.window)

    x = np.arange(len(datasets))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(1.8 + 1.9 * len(datasets), 3.6))
    for mi, model in enumerate(models):
        vals = [best.get((d, model), (np.nan, None))[0] for d in datasets]
        offset = (mi - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width * 0.92, label=model,
                      color=MODEL_COLORS[model])
        ax.bar_label(bars, fmt="%.3f", fontsize=8, color=INK, padding=2)
    ax.set_xticks(x, datasets)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(f"test {metric}")
    ax.set_title(f"Best test {metric} — model × dataset "
                 "(best window per recurrent model)", color=INK)
    ax.legend(frameon=False)
    ax.grid(axis="x", visible=False)
    return _save(fig, path)
