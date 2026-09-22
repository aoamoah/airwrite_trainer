"""Friedman + Wilcoxon (Holm) on per-fold F1 of the writing class."""
import json, sys, itertools
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

run = sys.argv[1]
S = json.load(open(f"{run}/summary.json"))
lab = lambda a: a["model"] + ("" if not a["window"] else f" w{a['window']['size']}")
scores = {}
for a in S:
    folds = {f["fold"]: f["f1_writing"] for f in a["per_fold"] if f.get("f1_writing") is not None}
    scores[lab(a)] = folds
common = sorted(set.intersection(*(set(v) for v in scores.values())))
names = sorted(scores, key=lambda k: -np.mean([scores[k][f] for f in common]))
X = {k: np.array([scores[k][f] for f in common]) for k in names}
print(f"{run}: {len(common)} folds, {len(names)} configurations")
for k in names:
    print(f"  {k:24s} mean {100*X[k].mean():5.2f}  sd {100*X[k].std(ddof=1):5.2f}")
stat, p = friedmanchisquare(*[X[k] for k in names])
print(f"\nFriedman chi2 = {stat:.3f}, p = {p:.4f}  ({'differences detected' if p < 0.05 else 'no detectable difference'})")

pairs = [(names[0], k) for k in names[1:]]
res = []
for a, b in pairs:
    d = X[a] - X[b]
    if np.allclose(d, 0):
        continue
    s, p = wilcoxon(X[a], X[b])
    res.append((a, b, 100*d.mean(), p))
res.sort(key=lambda r: r[3])
m = len(res)
print(f"\nWilcoxon signed-rank vs the best configuration ({names[0]}), Holm-corrected:")
prev = 0
for i, (a, b, diff, p) in enumerate(res):
    adj = max(prev, min(1.0, (m - i) * p)); prev = adj
    print(f"  vs {b:24s} mean diff {diff:+6.2f}  p = {p:.4f}  p_holm = {adj:.4f}{'  *' if adj < 0.05 else ''}")
