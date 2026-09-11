"""Train/val/test splitting without temporal leakage.

Two designs live here.

`make_folds` is the primary one: participant-level cross-validation
(leave-one-participant-out, or grouped k-fold when there are many
participants). No participant contributes frames to more than one split of
a fold, so test scores measure generalisation to an unseen hand rather than
to an unseen session of a hand already trained on. The early-stopping
validation set is drawn *from the training participants only*, and is
chosen so its class prior matches the pool it came from — otherwise early
stopping selects weights against a distribution that appears nowhere else
(the failure that produced a 56%-writing val set against 29%-writing train).

`assign_groups` is the older single holdout, kept because session-level
versus participant-level splits on identical data quantify how much
participant leakage inflates results. Whole units (sessions / participants /
row-chunks) are assigned to one split, greedily by frame count — each unit
goes to the split furthest below its target share — which keeps the ratios
honest when units have very different sizes and guarantees every split
receives at least one unit. A repair pass then swaps units so no split is
left with only one class (a real risk with few sessions: a session can be
100% not_writing).
"""

import itertools
from dataclasses import dataclass, field

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
    # `unsure` is not a class: stratify on the frames that are scored
    labelled = frames[frames["label"] != "unsure"]
    by_unit = labelled.groupby(unit_col)["label"].agg(set).to_dict()
    unit_classes = {u: by_unit.get(u, set()) for u in frames[unit_col].unique()}
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
    both = set(labelled["label"].unique())
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


# --------------------------------------------------------------------------
# Participant-level cross-validation
# --------------------------------------------------------------------------


@dataclass
class Fold:
    """One evaluation fold. `assignment` maps participant -> split name."""

    name: str
    assignment: dict[str, str]
    info: dict = field(default_factory=dict)
    # holdout folds are assigned per row (by session/chunk), not per participant
    precomputed_rows: pd.Series | None = None

    def participants(self, split: str) -> list[str]:
        return sorted(p for p, s in self.assignment.items() if s == split)

    def row_split(self, frames: pd.DataFrame) -> pd.Series:
        if self.precomputed_rows is not None:
            return self.precomputed_rows
        return frames["participant"].map(self.assignment)


def make_folds(frames: pd.DataFrame, eval_cfg: dict, split_cfg: dict,
               seed: int) -> tuple[list[Fold], dict]:
    """Build the evaluation folds. Returns (folds, design info)."""
    scheme = eval_cfg.get("scheme", "auto")
    people = sorted(frames["participant"].unique())
    rng = np.random.default_rng(seed)

    if scheme == "auto":
        limit = eval_cfg.get("lopo_max_participants", 16)
        scheme = "lopo" if len(people) <= limit else "group_kfold"
        auto_note = f"auto -> {scheme} ({len(people)} participants)"
    else:
        auto_note = None

    # A dataset that arrives as one undifferentiated CSV has no participant
    # column worth splitting on; fall back rather than fake it
    if scheme != "holdout" and len(people) < 3:
        scheme = "holdout"
        auto_note = (f"only {len(people)} participant(s) — participant folds "
                     "impossible, fell back to a single holdout split")

    if scheme == "holdout":
        assignment, info = assign_groups(
            frames, split_cfg["method"], tuple(split_cfg["ratios"]), seed,
            chunk_rows=split_cfg.get("chunk_rows", 500),
        )
        fold = Fold(name="holdout", assignment={}, info=info,
                    precomputed_rows=assignment)
        design = {"scheme": "holdout", "folds": 1,
                  "split_method": split_cfg["method"]}
        if auto_note:
            design["note"] = auto_note
        return [fold], design

    if scheme == "lopo":
        test_groups = [[p] for p in people]
    elif scheme == "group_kfold":
        test_groups = _balanced_participant_groups(
            frames, people, int(eval_cfg.get("k", 5)), rng)
    else:
        raise ValueError(f"unknown evaluation scheme: {scheme!r}")

    max_folds = eval_cfg.get("max_folds")
    truncated = None
    if max_folds and len(test_groups) > int(max_folds):
        truncated = f"{len(test_groups)} folds available, {max_folds} run"
        test_groups = test_groups[: int(max_folds)]

    n_val = int(eval_cfg.get("val_participants", 2))
    val_fraction = float(eval_cfg.get("val_fraction", 0.15))

    folds = []
    for i, test in enumerate(test_groups, 1):
        pool = [p for p in people if p not in test]
        val, val_info = _choose_val_participants(
            frames, pool, n_val, val_fraction, rng)
        assignment = {p: "train" for p in pool}
        assignment.update({p: "val" for p in val})
        assignment.update({p: "test" for p in test})
        folds.append(Fold(
            name=f"fold{i:02d}_" + "+".join(test),
            assignment=assignment,
            info={"test": list(test), "val": val,
                  "train": [p for p in pool if p not in val], **val_info},
        ))

    design = {
        "scheme": scheme,
        "folds": len(folds),
        "unit": "participant",
        "participants": len(people),
        "val_selection": ("drawn from training participants only; subset whose "
                          "writing prior is closest to the training pool"),
    }
    if scheme == "group_kfold":
        design["k"] = int(eval_cfg.get("k", 5))
    if auto_note:
        design["note"] = auto_note
    if truncated:
        design["truncated"] = truncated
    return folds, design


def _balanced_participant_groups(frames, people, k, rng) -> list[list[str]]:
    """Partition participants into k groups of roughly equal frame count."""
    k = max(2, min(k, len(people)))
    sizes = frames.groupby("participant").size()
    order = list(people)
    rng.shuffle(order)
    order.sort(key=lambda p: -sizes[p])
    groups: list[list[str]] = [[] for _ in range(k)]
    filled = [0] * k
    for p in order:
        j = int(np.argmin(filled))
        groups[j].append(p)
        filled[j] += int(sizes[p])
    return [sorted(g) for g in groups if g]


def _choose_val_participants(frames, pool, n_val, val_fraction, rng
                             ) -> tuple[list[str], dict]:
    """Pick validation participants out of `pool` (the training participants).

    Scored on two things: how close the validation writing prior is to the
    pool's, and how close the validation share of frames is to
    `val_fraction`. Prior match dominates — an unrepresentative val prior is
    what makes early stopping select the wrong weights.
    """
    if len(pool) < 2:
        return [], {"val_note": "training pool too small for a val split"}

    n_val = max(1, min(n_val, len(pool) - 1))
    sizes = frames.groupby("participant").size()
    writing = frames.assign(w=(frames["label"] == "writing").astype(int)) \
                    .groupby("participant")["w"].sum()
    pool_frames = int(sizes[pool].sum())
    pool_prior = float(writing[pool].sum() / max(pool_frames, 1))

    combos = list(itertools.combinations(pool, n_val))
    if len(combos) > 2000:  # large cohorts: sample rather than enumerate
        idx = rng.choice(len(combos), size=2000, replace=False)
        combos = [combos[i] for i in idx]

    best, best_score, best_stats = None, np.inf, None
    for combo in combos:
        n = int(sizes[list(combo)].sum())
        w = int(writing[list(combo)].sum())
        if n == 0 or w == 0 or w == n:      # val must see both classes
            continue
        rest_n, rest_w = pool_frames - n, int(writing[pool].sum()) - w
        if rest_n == 0 or rest_w == 0 or rest_w == rest_n:
            continue
        val_prior = w / n
        score = abs(val_prior - pool_prior) + 0.5 * abs(n / pool_frames - val_fraction)
        if score < best_score:
            best, best_score = list(combo), score
            best_stats = {"val_prior": round(val_prior, 4),
                          "train_prior": round(rest_w / rest_n, 4),
                          "pool_prior": round(pool_prior, 4),
                          "val_frame_share": round(n / pool_frames, 4)}

    if best is None:  # every candidate was single-class — leave val empty
        return [], {"val_note": "no two-class validation subset available"}
    return best, best_stats
