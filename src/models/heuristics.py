"""Rule-based baselines (Objective 4).

Two non-learned detectors that the learned models have to beat for the
comparison to mean anything:

- `velocity_threshold` — the index fingertip is moving faster than a fitted
  speed threshold. This is the obvious "writing is motion" rule.
- `extension_threshold` — the index finger is extended further from the
  wrist than the other fingertips, i.e. the hand is in a pointing pose.

Both read a single precomputed signal column (see
`src.data.preprocess.heuristic_signals`) computed from *raw* landmarks:
wrist-origin normalisation removes the very translation the speed rule
depends on, so these signals are taken before it is applied.

The only thing fitted is the decision threshold, chosen on the validation
split by maximising F-beta of the `writing` class — the same split and the
same beta the learned models' operating point uses, so neither side of the
comparison gets a better-fitted threshold than the other.
"""

import numpy as np
from sklearn.metrics import fbeta_score


class ThresholdBaseline:
    """Predict `writing` where a scalar signal exceeds a fitted threshold."""

    signal_column: str = ""
    description: str = ""

    def __init__(self, params: dict | None = None):
        self.params = dict(params or {})
        self.threshold: float = 0.0
        self.fit_f1: float | None = None

    def fit(self, signal: np.ndarray, y: np.ndarray,
            beta: float = 1.0) -> "ThresholdBaseline":
        """Fit the threshold. `beta` must match the one the learned models'
        operating point is fitted for, or the two sides of the comparison are
        optimising different objectives."""
        signal = np.asarray(signal, dtype=np.float64)
        if len(signal) == 0 or len(np.unique(y)) < 2:
            self.threshold = float(np.median(signal)) if len(signal) else 0.0
            return self
        # Candidate thresholds: quantiles of the observed signal, which
        # concentrates the search where the data actually is
        qs = np.linspace(0.01, 0.99, self.params.get("search_steps", 199))
        candidates = np.unique(np.quantile(signal, qs))
        scores = [fbeta_score(y, (signal >= t).astype(int), beta=beta,
                              zero_division=0)
                  for t in candidates]
        best = int(np.argmax(scores))
        self.threshold = float(candidates[best])
        self.fit_f1 = float(scores[best])
        self.fit_beta = beta
        return self

    def decision_scores(self, signal: np.ndarray) -> np.ndarray:
        """Raw signal — monotone in P(writing), so ROC AUC is meaningful."""
        return np.asarray(signal, dtype=np.float64)


class VelocityThreshold(ThresholdBaseline):
    name = "velocity_threshold"
    signal_column = "heur_speed"
    description = ("index fingertip speed (smoothed) above a fitted "
                   "threshold -> writing")


class ExtensionThreshold(ThresholdBaseline):
    name = "extension_threshold"
    signal_column = "heur_extension"
    description = ("index fingertip extended beyond the other fingertips "
                   "(pointing pose) above a fitted threshold -> writing")


BASELINES = {
    VelocityThreshold.name: VelocityThreshold,
    ExtensionThreshold.name: ExtensionThreshold,
}
