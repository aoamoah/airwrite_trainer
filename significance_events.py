"""Friedman + Wilcoxon (Holm) on per-fold episode F1 and pause F2."""
import json, sys, collections
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

run = sys.argv[1]
R = json.load(open(f"{run}/results.json"))
lab = lambda r: r["model"] + ("" if not r["window"] else f" w{r['window']['size']}")

def pick(ev, key, sub):
    v = (ev or {}).get(key)
    return None if not isinstance(v, dict) else v.get(sub)

for metric, key, sub in (("episode F1 (IoU 0.5)", "iou@0.5", "f1"),
                         ("pause F2 @12 frames", "pause@12f", "f2"),
                         ("Edit score", "edit_score", None)):
    scores = collections.defaultdict(dict)
    for r in R:
        ev = r.get("event_metrics") or {}
        val = ev.get("edit_score") if sub is None else pick(ev, key, sub)
        if isinstance(val, dict):
            val = val.get("mean")
        if val is not None:
            scores[lab(r)][r["fold"]] = val
    if not scores:
        print(metric, "— not available"); continue
    common = sorted(set.intersection(*(set(v) for v in scores.values())))
    names = sorted(scores, key=lambda k: -np.mean([scores[k][f] for f in common]))
    X = {k: np.array([scores[k][f] for f in common]) for k in names}
    stat, p = friedmanchisquare(*[X[k] for k in names])
    print(f"\n=== {metric}: {len(common)} folds, {len(names)} configurations")
    for k in names[:4]:
        print(f"   {k:22s} mean {100*X[k].mean():5.1f}  sd {100*X[k].std(ddof=1):5.1f}")
    print(f"   Friedman chi2={stat:.3f}, p={p:.4f}" + ("  → differences detected" if p < 0.05 else "  → none detected"))
    res = []
    for b in names[1:]:
        d = X[names[0]] - X[b]
        if np.allclose(d, 0): continue
        res.append((b, 100*d.mean(), wilcoxon(X[names[0]], X[b]).pvalue))
    res.sort(key=lambda r: r[2]); prev = 0
    for i, (b, diff, p2) in enumerate(res):
        adj = max(prev, min(1.0, (len(res) - i) * p2)); prev = adj
        print(f"     {names[0]} vs {b:22s} {diff:+6.1f}  p={p2:.4f}  Holm={adj:.4f}{'  *' if adj<0.05 else ''}")
