"""NN inference from original parameters, with explicit unscaled prepared paths."""
import torch

from .data import feature_tensor
from ...data.scalers import TorchLogStandardScaler
from ...evaluate.predictor import ParameterPredictor
from ...formulations import output_to_logit, boundary_margin


class NNPredictor(ParameterPredictor):
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
