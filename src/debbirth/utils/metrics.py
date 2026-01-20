from __future__ import annotations

from typing import Any

import numpy as np
import torch


def extract_pos_proba(proba: np.ndarray) -> np.ndarray:
    """Return positive-class probability as shape [N].

    Accepts:
    - [N] (already positive probs)
    - [N, 1] (treated as positive probs)
    - [N, 2] (sklearn convention: column 1 is positive)
    """
    if proba.ndim == 1:
        return proba
    if proba.ndim == 2 and proba.shape[1] == 1:
        return proba[:, 0]
    if proba.ndim == 2 and proba.shape[1] == 2:
        return proba[:, 1]
    raise ValueError(f"Unexpected predict_proba shape: {proba.shape}")


def convert_to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    raise TypeError(f"Unsupported type: {type(x)}")


def compute_pos_weight(y_train: torch.Tensor) -> torch.Tensor:
    # pos_weight = (#neg / #pos) for BCEWithLogitsLoss
    n_pos = torch.sum(y_train == 1).float()
    n_neg = torch.sum(y_train == 0).float()
    if n_pos <= 0:
        return torch.tensor(1.0)
    return n_neg / n_pos
