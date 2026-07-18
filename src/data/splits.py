"""Train/val/test splitting without temporal leakage.

Whole units (sessions / participants / row-chunks) are assigned to one
split. Assignment is greedy by frame count — each unit goes to the split
furthest below its target share — which keeps the ratios honest when units
have very different sizes and guarantees every split receives at least one
unit. A repair pass then swaps units so no split is left with only one
class (a real risk with few sessions: a session can be 100% not_writing).
"""

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def assign_groups(
    frames: pd.DataFrame,
    method: str,
    ratios: tuple[float, float, float],
    seed: int,
    chunk_rows: int = 500,
) -> tuple[pd.Series, dict]:
    """Return a per-row split assignment ("train"/"val"/"test") and split info."""
    frames = frames.copy()
    if frames["group"].nunique() == 1:
        frames["group"] = [
            f"chunk_{i // chunk_rows:04d}" for i in range(len(frames))
        ]
        regrouped = True
    else:
        regrouped = False

    rng = np.random.default_rng(seed)

    if method == "random":
        idx = rng.permutation(len(frames))
        n_train = round(len(idx) * ratios[0])
        n_val = round(len(idx) * ratios[1])
        assignment = pd.Series("test", index=frames.index)
        assignment.iloc[idx[:n_train]] = "train"
        assignment.iloc[idx[n_train:n_train + n_val]] = "val"
        info = {"method": "random (leaky baseline)", "regrouped_chunks": regrouped}
        return assignment, info

    unit_col = "participant" if method == "participant" else "group"
    unit_to_split, notes = _greedy_stratified(frames, unit_col, ratios, rng)
    assignment = frames[unit_col].map(unit_to_split)

    split_units = {s: sorted(u for u, sp in unit_to_split.items() if sp == s)
                   for s in SPLITS}
    info = {
        "method": method,
        "unit": unit_col,
        "regrouped_chunks": regrouped,
        "assignment": "greedy by frame count, class-repair pass",
        "units_per_split": split_units,
        "rows_per_split": assignment.value_counts().to_dict(),
        **notes,
    }
    return assignment, info


def _greedy_stratified(frames, unit_col, ratios, rng) -> tuple[dict, dict]:
    sizes = frames.groupby(unit_col).size()
    unit_classes = frames.groupby(unit_col)["label"].agg(set)
    units = list(sizes.index)
    rng.shuffle(units)
    units.sort(key=lambda u: -sizes[u])  # big units placed first, ties by shuffle

    total = len(frames)
    targets = {s: r * total for s, r in zip(SPLITS, ratios)}
    filled = {s: 0 for s in SPLITS}
    assign: dict = {}

    active = [s for s in SPLITS if targets[s] > 0]
    for u in units:
        # Prefer splits that are still empty (guarantees each gets a unit
        # when there are enough units), then the one furthest below target
        empty = [s for s in active if filled[s] == 0]
        pool = empty if empty and len(units) >= len(active) else active
        s = max(pool, key=lambda s: (targets[s] - filled[s]) / max(targets[s], 1))
        assign[u] = s
        filled[s] += sizes[u]

    # Test may never end up empty while other splits hold spare units —
    # evaluation is mandatory, val is not (the runner falls back for val)
    if not any(sp == "test" for sp in assign.values()):
        for donor in ("val", "train"):
            units_in = [u for u, sp in assign.items() if sp == donor]
            min_keep = 1 if donor == "train" else 0
            if len(units_in) > min_keep:
                assign[min(units_in, key=lambda u: sizes[u])] = "test"
                break

    notes = {}
    both = set(frames["label"].unique())
    if len(both) > 1:
        swaps = _repair_single_class(assign, unit_classes, sizes, both)
        if swaps:
            notes["class_repair_swaps"] = swaps
        still_bad = [
            s for s in SPLITS
            if any(sp == s for sp in assign.values())
            and set().union(*(unit_classes[u] for u, sp in assign.items() if sp == s)) != both
        ]
        if still_bad:
            notes["warning_single_class_splits"] = still_bad
    return assign, notes


def _repair_single_class(assign, unit_classes, sizes, both) -> list[str]:
    """Swap units between splits until every split sees both classes."""
    swaps = []
    for s in SPLITS:
        units_in = [u for u, sp in assign.items() if sp == s]
        if not units_in:
            continue
        classes = set().union(*(unit_classes[u] for u in units_in))
        if classes == both:
            continue
        missing = both - classes
        # find a donor unit elsewhere covering the missing class(es), whose
        # removal keeps its own split two-class; swap with our closest-size unit
        for donor, donor_split in sorted(assign.items(), key=lambda kv: sizes[kv[0]]):
            if donor_split == s or not missing.issubset(unit_classes[donor]):
                continue
            donor_rest = [u for u, sp in assign.items()
                          if sp == donor_split and u != donor]
            if donor_rest and set().union(*(unit_classes[u] for u in donor_rest)) != both:
                continue
            here = min(units_in, key=lambda u: abs(sizes[u] - sizes[donor]))
            assign[donor], assign[here] = s, donor_split
            swaps.append(f"{donor} -> {s}, {here} -> {donor_split}")
            break
    return swaps
