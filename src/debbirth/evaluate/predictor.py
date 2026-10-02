"""Original-parameter inference semantics shared by NN and GP predictors (T06A).

Subclasses bundle a model with its feature schema, preprocessing and
temperature and implement `_outputs(features, feature_names, offset)` on
unscaled prepared features. It returns NumPy vectors of shape (N,):
learned_output, logit, probability, prediction and, for boundary models,
margin. Boundary predictions are the strict margin > 0. A probability
threshold is only an explicit operating point. This module imports no
model backend.
"""
import numpy as np
import pandas as pd

from ..data.prepare import prepare_features
from ..data.schema import FULL_PAR_COLS


class ParameterPredictor:
    spec = None  # DatasetSpec of the bundled model

    def _outputs(self, features, feature_names, offset):
        raise NotImplementedError

    def _frame(self, parameters):
        if isinstance(parameters, pd.DataFrame):
            return parameters
        values = np.asarray(parameters, dtype=float)
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValueError("Original parameter arrays must have shape (N, 4): g, k, v_Hb, f.")
        return pd.DataFrame(values, columns=FULL_PAR_COLS)

    def predict_details(self, parameters):
        """Outputs from original parameters (DataFrame or (N, 4) array ordered g, k, v_Hb, f)."""
        features, offset = prepare_features(self._frame(parameters), self.spec)
        return self._outputs(features.to_numpy(), features.columns, offset)

    def predict_prepared(self, split):
        """Return details from an unscaled PreparedSplit; never normalize again."""
        return self._outputs(split.features, split.feature_names, split.log_nu_b)

    def evaluate_prepared(self, split, *, probability_threshold=None):
        """Metrics from one prediction pass over an unscaled PreparedSplit."""
        from .predict import metrics_from_predictions
        details = self.predict_prepared(split)
        return metrics_from_predictions(split.labels, details["probability"],
                                        y_pred=self._decide(details, probability_threshold))

    def predict_proba(self, parameters):
        return self.predict_details(parameters)["probability"]

    @staticmethod
    def _decide(details, probability_threshold):
        if probability_threshold is None:
            return details["prediction"]
        if not 0 < probability_threshold < 1:
            raise ValueError("probability_threshold must be in (0, 1).")
        # An explicit operating point, separate from canonical strict feasibility.
        return (details["probability"] >= probability_threshold).astype(int)

    def predict(self, parameters, *, probability_threshold=None):
        return self._decide(self.predict_details(parameters), probability_threshold)

    def _require_boundary(self):
        if self.spec.formulation != "boundary":
            raise ValueError("Critical maturity and margin require a boundary model.")

    def predict_margin(self, parameters):
        self._require_boundary()
        return self.predict_details(parameters)["margin"]

    def critical_maturity(self, parameters, *, normalized=True, log=False):
        """Return log(Psi)/Psi or log(f^3 Psi)/f^3 Psi.

        Use log=True for extreme thresholds. Explicit exponentiation raises
        on overflow/underflow; probabilities need no exponentiated threshold.
        """
        self._require_boundary()
        frame = self._frame(parameters)
        log_critical = self.predict_details(frame)["learned_output"].astype(float)
        if not normalized:
            log_critical = log_critical + 3 * np.log(frame["f"].to_numpy(dtype=float))
        if log:
            return log_critical
        with np.errstate(over="ignore", under="ignore"):
            critical = np.exp(log_critical)
        if not np.isfinite(critical).all() or (critical == 0).any():
            raise FloatingPointError("Critical maturity is not representable; request log=True.")
        return critical
