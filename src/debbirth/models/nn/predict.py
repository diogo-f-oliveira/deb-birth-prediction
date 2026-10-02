"""NN inference from original parameters, with explicit unscaled prepared paths."""
import numpy as np
import pandas as pd
import torch

from .data import feature_tensor
from ...data.prepare import prepare_features
from ...data.scalers import TorchLogStandardScaler
from ...data.schema import FULL_PAR_COLS
from ...formulations import output_to_logit, boundary_margin


class NNPredictor:
    """Bundle model, schema, preprocessing and fixed boundary temperature.

    Public methods take physical DataFrames or (N, 4) arrays ordered
    (g, k, v_Hb, f). PreparedSplit methods accept unscaled learned features.
    All outputs are NumPy vectors of shape (N,). Low-level model APIs remain
    available for callers that explicitly manage tensors and scaling.
    """

    def __init__(self, model, config, scaler=None, *, batch_size=1024):
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if config.net_config.input_dim != config.data_spec.n_features:
            raise ValueError("Network input dimension does not match the feature schema.")
        if (config.scaling_type not in (None, "none")) != (scaler is not None):
            raise ValueError("Scaler presence does not match the saved preprocessing.")
        self.model, self.config, self.scaler = model.eval(), config, scaler
        self.spec = config.data_spec
        self.batch_size = batch_size

    def _frame(self, parameters):
        if isinstance(parameters, pd.DataFrame):
            return parameters
        values = np.asarray(parameters, dtype=float)
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValueError("Original parameter arrays must have shape (N, 4): g, k, v_Hb, f.")
        return pd.DataFrame(values, columns=FULL_PAR_COLS)

    @torch.inference_mode()
    def _outputs(self, features, feature_names, offset):
        if tuple(feature_names) != tuple(self.spec.feature_cols):
            raise ValueError("Prepared feature order differs from the saved schema.")
        device = next(self.model.parameters()).device
        self.model.eval()
        if self.scaler is not None:
            self.scaler.to(device)
        x = feature_tensor(features, feature_names, device=device,
                           needs_log=isinstance(self.scaler, TorchLogStandardScaler))
        if self.scaler is not None:
            x = self.scaler.transform(x)
        if not torch.isfinite(x).all():
            raise ValueError("Nonfinite scaled inference features.")
        o = None if offset is None else torch.as_tensor(offset, dtype=x.dtype, device=device)
        if o is not None and (o.shape != (len(x),) or not torch.isfinite(o).all()):
            raise ValueError("Offsets must be finite and aligned with features.")
        outputs = [self.model(part) for part in x.split(self.batch_size) if len(part)]
        learned = torch.cat(outputs) if outputs else x.new_empty(0)
        logits = output_to_logit(learned, formulation=self.spec.formulation, log_nu_b=o,
                                  temperature=self.config.boundary_temperature)
        if not torch.isfinite(learned).all() or not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite network output/logit.")
        p = torch.sigmoid(logits)
        boundary = self.spec.formulation == "boundary"
        margin = boundary_margin(learned, o) if boundary else None
        result = {"learned_output": learned.cpu().numpy(), "logit": logits.cpu().numpy(),
                  "probability": p.cpu().numpy(),
                  "prediction": ((margin > 0) if boundary else (p >= self.model.threshold)).cpu().numpy().astype(int)}
        if boundary:
            result["margin"] = margin.cpu().numpy()
        return result

    def predict_details(self, parameters):
        features, offset = prepare_features(self._frame(parameters), self.spec)
        return self._outputs(features.to_numpy(), features.columns, offset)

    def predict_prepared(self, split):
        """Return details from an unscaled PreparedSplit; never normalize again."""
        return self._outputs(split.features, split.feature_names, split.log_nu_b)

    def predict_proba(self, parameters):
        return self.predict_details(parameters)["probability"]

    def predict(self, parameters, *, probability_threshold=None):
        details = self.predict_details(parameters)
        if probability_threshold is None:
            return details["prediction"]
        if not 0 < probability_threshold < 1:
            raise ValueError("probability_threshold must be in (0, 1).")
        # An explicit operating point, separate from canonical strict feasibility.
        return (details["probability"] >= probability_threshold).astype(int)

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
