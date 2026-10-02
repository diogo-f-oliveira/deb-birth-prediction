"""GP inference from original parameters, with explicit unscaled prepared paths."""
import numpy as np
from scipy.special import expit

from .algorithm import DEBBirthSymbolicClassifier
from .boundary import GPBoundaryModel
from ...evaluate.predictor import ParameterPredictor
from ...formulations import boundary_margin


class GPPredictor(ParameterPredictor):
    """Bundle a GP model with its feature schema (GP inputs are unscaled).

    Same interface as NNPredictor. Classifier scores reproduce gplearn's own
    predict_proba/predict (sigmoid; argmax, so probability 0.5 is infeasible)
    from one program execution; scores may be infinite, but not NaN.
    Boundary models use the strict margin > 0 and their saved temperature.
    """

    def __init__(self, model, data_spec):
        boundary = isinstance(model, GPBoundaryModel)
        if not boundary and not isinstance(model, DEBBirthSymbolicClassifier):
            raise TypeError("Expected a DEBBirthSymbolicClassifier or GPBoundaryModel.")
        if boundary != (data_spec.formulation == "boundary"):
            raise ValueError("Model type does not match the data_spec formulation.")
        names = model.feature_names if boundary else model.feature_names_no_constants
        if tuple(names) != tuple(data_spec.feature_cols):
            raise ValueError("Model feature order differs from the data_spec feature order.")
        self.model, self.spec = model, data_spec
        self.temperature = model.temperature if boundary else 1.0

    def _outputs(self, features, feature_names, offset):
        if tuple(feature_names) != tuple(self.spec.feature_cols):
            raise ValueError("Prepared feature order differs from the saved schema.")
        X = np.asarray(features, dtype=float)
        if isinstance(self.model, GPBoundaryModel):
            if offset is None:
                raise ValueError("Boundary inference requires the separately prepared log_nu_b.")
            offset = np.asarray(offset, dtype=float)
            if not np.isfinite(offset).all():
                raise ValueError("Expected finite log_nu_b.")
            F = self.model.predict_boundary(X)
            margin = boundary_margin(F, offset)
            logit = margin / self.temperature
            return {"learned_output": F, "logit": logit, "probability": expit(logit),
                    "prediction": (margin > 0).astype(int), "margin": margin}
        if offset is not None:
            raise ValueError("Maturity offsets apply only to boundary models.")
        if X.ndim != 2 or X.shape[1] != len(self.spec.feature_cols):
            raise ValueError("Incorrect prepared feature shape.")
        model = self.model
        with np.errstate(over="ignore", invalid="ignore"):
            score = model._program.execute(model._augment_X(X))
            probability = model._transformer(score)
        if np.isnan(probability).any():
            raise FloatingPointError("GP classifier produced NaN scores.")
        proba = np.vstack([1 - probability, probability]).T
        prediction = model.classes_.take(np.argmax(proba, axis=1), axis=0).astype(int)
        return {"learned_output": score, "logit": score, "probability": probability, "prediction": prediction}
