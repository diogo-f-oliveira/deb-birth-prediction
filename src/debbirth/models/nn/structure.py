from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .config import DEBBirthNetConfig
from ...formulations import output_to_logit, boundary_margin, validate_temperature


class DEBBirthNet(nn.Module):

    def __init__(self, cfg: DEBBirthNetConfig):
        super().__init__()
        layers: List[nn.Module] = []

        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            if cfg.dropout and cfg.dropout > 0:
                layers.append(nn.Dropout(p=cfg.dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, 1))  # unrestricted score, or F = log(Psi)
        self.net = nn.Sequential(*layers)

        # store threshold from config
        self.threshold = cfg.threshold

        # Optional: mild init
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D]
        logits = self.net(x).squeeze(-1)  # [B]
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return probabilities in [0, 1] by applying sigmoid to logits."""
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        return probs

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return binary class predictions (0 or 1) by thresholding probabilities using config.threshold."""
        probs = self.predict_proba(x)
        return (probs >= self.threshold).to(torch.int64)


class DEBBirthBoundaryNet(DEBBirthNet):
    """forward(x) returns unrestricted F; maturity never enters the network.

    x is already prepared/scaled. Probability requires an explicit, unscaled
    log_nu_b offset. The inherited state-dict layout preserves the NN format.
    """

    def __init__(self, cfg: DEBBirthNetConfig, temperature=1.0):
        super().__init__(cfg)
        validate_temperature(temperature)
        self.temperature = temperature

    def predict_margin(self, x, log_nu_b):
        return boundary_margin(self(x), log_nu_b)

    def predict_proba(self, x, log_nu_b):
        return torch.sigmoid(output_to_logit(
            self(x), formulation="boundary", log_nu_b=log_nu_b, temperature=self.temperature))

    def predict(self, x, log_nu_b):
        return (self.predict_margin(x, log_nu_b) > 0).to(torch.int64)


def build_net(config):
    if config.data_spec.formulation == "boundary":
        return DEBBirthBoundaryNet(config.net_config, config.boundary_temperature)
    return DEBBirthNet(config.net_config)
