from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn


@dataclass(frozen=True)
class DEBBirthNetConfig:
    input_dim: int
    hidden_dims: List[int] = None
    dropout: float = 0.1
    threshold: float = 0.5  # new hyperparameter: decision threshold for predict

    def __post_init__(self):
        if self.hidden_dims is None:
            object.__setattr__(self, "hidden_dims", [64, 32])
        # validate threshold is in (0,1)
        if not (0.0 < self.threshold < 1.0):
            raise ValueError(f"threshold must be between 0 and 1 (exclusive), got {self.threshold}")


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

        layers.append(nn.Linear(in_dim, 1))  # logits
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
