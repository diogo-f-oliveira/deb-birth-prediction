from __future__ import annotations

from typing import Tuple, Any

import numpy as np
import pandas as pd
import torch

from .numpy import convert_to_numpy


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_str: str) -> torch.device:
    s = (device_str or "auto").lower().strip()
    if s == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def convert_to_tensor(X):
    if isinstance(X, torch.Tensor):
        return X
    if isinstance(X, np.ndarray):
        return torch.from_numpy(np.asarray(X))
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        return torch.from_numpy(X.values)
    raise TypeError("Input must be a torch.Tensor or numpy array")


def collect_xy_from_dataloader(
    dataloader: torch.utils.data.DataLoader,
    to_numpy: bool = False,
) -> Tuple[Any, Any]:
    """Collect X,y from a dataloader.

    Args:
        dataloader: torch DataLoader yielding (x, y) batches.
        to_numpy: if True, return (X: np.ndarray, y: np.ndarray) as before.
                  if False (default), return (X: torch.Tensor, y: torch.Tensor).

    Returns:
        Tuple of concatenated inputs and targets. y is binarized (>=0.5).
    """
    xs = []
    ys = []

    for x, y in dataloader:
        xs.append(x)
        ys.append(y)

    if not xs:
        if to_numpy:
            return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int32)
        else:
            return torch.empty((0, 0), dtype=torch.float32), torch.empty((0,), dtype=torch.int64)

    if to_numpy:
        x_list = [convert_to_numpy(xi) for xi in xs]
        y_list = [convert_to_numpy(yi) for yi in ys]
        X = np.concatenate(x_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        y = (y >= 0.5).astype(np.int32)
        return X, y

    # Return tensors (concatenate as torch tensors). Convert numpy/pandas batches if needed.
    tensor_xs = []
    for xi in xs:
        if torch.is_tensor(xi):
            tensor_xs.append(xi)
        elif isinstance(xi, np.ndarray):
            tensor_xs.append(torch.from_numpy(xi))
        elif isinstance(xi, (pd.DataFrame, pd.Series)):
            tensor_xs.append(torch.from_numpy(xi.values))
        else:
            raise TypeError(f"Unsupported batch type for X: {type(xi)}")

    tensor_ys = []
    for yi in ys:
        if torch.is_tensor(yi):
            # binarize on the tensor
            tensor_ys.append((yi >= 0.5).to(dtype=torch.int64))
        elif isinstance(yi, np.ndarray):
            tensor_ys.append(torch.from_numpy((yi >= 0.5).astype(np.int64)))
        elif isinstance(yi, (pd.Series, pd.DataFrame)):
            vals = yi.values if hasattr(yi, "values") else np.asarray(yi)
            tensor_ys.append(torch.from_numpy((vals >= 0.5).astype(np.int64)))
        else:
            raise TypeError(f"Unsupported batch type for y: {type(yi)}")

    X = torch.cat(tensor_xs, dim=0)
    y = torch.cat(tensor_ys, dim=0)
    return X, y
