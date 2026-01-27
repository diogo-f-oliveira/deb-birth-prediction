import torch
from torch import nn

from ..utils.pytorch import convert_to_tensor, resolve_device
from typing import Union, Any, Dict
from pathlib import Path


class BaseScaler(nn.Module):
    """
    Minimal base scaler: registers common buffers and provides save/load + hooks.
    Does NOT implement any standardization logic — subclasses must implement fit/partial_fit/transform/inverse_transform.
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True, eps: float = 1e-8):
        super().__init__()
        self.with_mean = with_mean
        self.with_std = with_std
        self.eps = float(eps)
        # register buffers so state_dict contains them even if empty
        self.register_buffer("mean_", torch.tensor([]))
        self.register_buffer("var_", torch.tensor([]))
        self.register_buffer("scale_", torch.tensor([]))
        self.register_buffer("n_samples_seen_", torch.tensor(0, dtype=torch.long))

    # preprocessing hooks (identity by default). Subclasses may override.
    def _preprocess(self, X: torch.Tensor) -> torch.Tensor:
        return X

    def _inverse_preprocess(self, X: torch.Tensor) -> torch.Tensor:
        return X

    def _to_tensor(self, X):
        return convert_to_tensor(X).float()

    # Abstract API: subclasses must implement these
    def fit(self, X, sample_dim: int = 0):
        raise NotImplementedError

    def partial_fit(self, X, sample_dim: int = 0):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError

    def inverse_transform(self, X):
        raise NotImplementedError

    def forward(self, X):
        return self.transform(X)

    @property
    def fitted_(self) -> bool:
        return bool(self.n_samples_seen_.numel() and int(self.n_samples_seen_) > 0)

    def save(self, path: Union[str, "Path"]) -> None:
        """Save scaler init args + state_dict to `path` using torch.save."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "class": self.__class__.__name__,
            "init": {"with_mean": self.with_mean, "with_std": self.with_std, "eps": self.eps},
            "state": self.state_dict(),
        }
        torch.save(data, p)

    @classmethod
    def load(cls, path: Union[str, Path], map_location: Any = None) -> "BaseScaler":
        """
        Generic loader that maps tensors to map_location and pre-sets buffers
        present in the saved state to avoid size-mismatch when loading.
        """
        data = torch.load(path, map_location=map_location, weights_only=True)
        init = data.get("init", {})
        state = data.get("state", data if isinstance(data, dict) else {})

        inst = cls(with_mean=init.get("with_mean", True), with_std=init.get("with_std", True),
                   eps=init.get("eps", 1e-8))

        # pre-set buffers from state if available to avoid shape mismatches

        if isinstance(state, dict):
            for name in ("mean_", "var_", "scale_", "n_samples_seen_"):
                if name in state:
                    val = state[name]
                    if name == "n_samples_seen_":
                        try:
                            val = val.to(dtype=torch.long)
                        except Exception:
                            val = torch.tensor(
                                int(val),
                                dtype=torch.long,
                                device=val.device if hasattr(val, "device") else None
                            )
                    inst.register_buffer(name, val)
            inst.load_state_dict(state, strict=False)
        return inst


class TorchStandardScaler(BaseScaler):
    """
    Standardization scaler: implements fit/partial_fit/transform/inverse_transform.
    Uses _preprocess/_inverse_preprocess hook to allow subclasses to change input representation.
    """

    # New helper to set fitted buffers and n_samples_seen_
    def _set_fitted_params(self, mean: torch.Tensor, var: torch.Tensor, n_samples: int, device=None) -> None:
        """
        Centralize setting mean_, var_, scale_ and n_samples_seen_.
        mean/var are expected on the target device or will be moved there.
        """
        if device is None:
            device = mean.device if isinstance(mean,
                                               torch.Tensor) and mean.numel() else self.mean_.device if self.mean_.numel() else None

        if not self.with_mean:
            mean = torch.zeros_like(mean)
        if not self.with_std:
            var = torch.ones_like(var)

        scale = torch.sqrt(var + self.eps)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        if device is not None:
            mean = mean.to(device)
            var = var.to(device)
            scale = scale.to(device)

        # register/update buffers so they appear in state_dict()
        self.register_buffer("mean_", mean.detach())
        self.register_buffer("var_", var.detach())
        self.register_buffer("scale_", scale.detach())

        n_tensor = torch.tensor(
            int(n_samples),
            dtype=torch.long,
            device=mean.device if isinstance(mean, torch.Tensor) and mean.numel() else None,
        )
        self.register_buffer("n_samples_seen_", n_tensor)

    def fit(self, X, sample_dim: int = 0):
        X = self._to_tensor(X)
        if X.numel() == 0:
            raise ValueError("Empty array passed to fit()")

        device = self.mean_.device if self.mean_.numel() else X.device
        X = X.to(device)

        Xp = self._preprocess(X)

        mean = Xp.mean(dim=sample_dim)
        var = Xp.var(dim=sample_dim, unbiased=False)

        n = X.shape[sample_dim]
        # reuse helper to set buffers
        self._set_fitted_params(mean, var, n, device=device)
        return self

    def partial_fit(self, X, sample_dim: int = 0):
        X = self._to_tensor(X)
        if X.numel() == 0:
            return self

        device = self.mean_.device if self.mean_.numel() else X.device
        X = X.to(device)

        n_new = X.shape[sample_dim]
        Xp = self._preprocess(X)
        mean_new = Xp.mean(dim=sample_dim)
        var_new = Xp.var(dim=sample_dim, unbiased=False)

        if self.n_samples_seen_.numel() == 0 or int(self.n_samples_seen_) == 0:
            # set from new batch
            self._set_fitted_params(mean_new, var_new, n_new, device=device)
            return self

        n_old = int(self.n_samples_seen_)
        mean_old = self.mean_.to(device)
        var_old = self.var_.to(device)

        n_total = n_old + n_new
        mean_total = (mean_old * n_old + mean_new * n_new) / n_total

        delta_old = mean_old - mean_total
        delta_new = mean_new - mean_total
        var_total = (n_old * var_old + n_new * var_new + n_old * (delta_old ** 2) + n_new * (delta_new ** 2)) / n_total

        # reuse helper to set updated buffers
        self._set_fitted_params(mean_total, var_total, n_total, device=device)
        return self

    def transform(self, X):
        if self.n_samples_seen_.numel() == 0 or int(self.n_samples_seen_) == 0:
            raise RuntimeError(f"{self.__class__.__name__} not fitted yet. Call fit or partial_fit first.")
        X = self._to_tensor(X)
        device = self.mean_.device
        X = X.to(device)
        Xp = self._preprocess(X)
        return (Xp - self.mean_) / self.scale_

    def inverse_transform(self, X):
        if self.n_samples_seen_.numel() == 0 or int(self.n_samples_seen_) == 0:
            raise RuntimeError(f"{self.__class__.__name__} not fitted yet. Call fit or partial_fit first.")
        X = self._to_tensor(X)
        device = self.mean_.device
        X = X.to(device)
        X_unscaled = X * self.scale_ + self.mean_
        return self._inverse_preprocess(X_unscaled)


class TorchLogStandardScaler(TorchStandardScaler):
    """
    Log + standardize scaler: inherits all standardization behaviour and only
    overrides preprocessing hooks to apply log/exp.
    """

    def _preprocess(self, X: torch.Tensor) -> torch.Tensor:
        # Apply log transform; users must ensure X > 0
        return torch.log(X)

    def _inverse_preprocess(self, X: torch.Tensor) -> torch.Tensor:
        return torch.exp(X)


def fit_and_scale_data_pytorch(data, scaling_type: str, device=None):
    """
    Scale data using Scaler classes.

    Arguments:
      - data: mapping (e.g. dict) of splits -> arrays/tensors, must include a 'train' key
      - scaling_type: 'standardize' to use TorchStandardScaler, 'log_standardize' for log+standardize,
                      'none' or None to skip scaling
      - device: Optional device (str or torch.device). If provided, the scaler will be fitted on this device
        and all transformed outputs will be moved to this device. If None, device is inferred from the data/scaler.

    Returns:
      (scaled_data, scaler) where scaled_data is a dict with same keys as input mapped to torch.Tensor,
      and scaler is the fitted scaler instance (or None if no scaling is applied).
    """
    # no scaling requested
    if scaling_type is None or scaling_type == 'none':
        return data, None

    # require a mapping with 'train'
    if not hasattr(data, "get") or data.get("train", None) is None:
        raise ValueError("scale_data_pytorch requires data to be a mapping containing a 'train' split.")

    # normalize device argument
    if device is not None:
        device = resolve_device(device)

    # create scaler based on requested type
    if scaling_type == 'standardize':
        scaler = TorchStandardScaler()
    elif scaling_type == 'log_standardize':
        scaler = TorchLogStandardScaler()
    else:
        raise ValueError(f"Unknown scaling_type '{scaling_type}'. Supported: 'standardize', 'log_standardize', 'none'.")

    # prepare train split tensor and move to device (if provided) so scaler buffers are created on that device
    train_split = data["train"]
    train_tensor = convert_to_tensor(train_split).float()
    if device is not None:
        train_tensor = train_tensor.to(device)

    # fit on train split (buffers will be created on train_tensor.device)
    scaler.fit(train_tensor)
    # ensure scaler is on target device (safe no-op if device is None)
    if device is not None:
        scaler = scaler.to(device)

    scaled_data = scale_data_pytorch(data=data, scaler=scaler, device=device)

    return scaled_data, scaler


def scale_data_pytorch(data, scaler: BaseScaler, device=None):
    """
    Scale data using a provided scaler instance.

    Arguments:
      - data: mapping (e.g. dict) of splits -> arrays/tensors
      - scaler: fitted scaler instance (e.g. TorchStandardScaler or TorchLogStandardScaler)
      - device: Optional device (str or torch.device). If provided, all transformed outputs will be moved to this device.
        If None, device is inferred from the data/scaler.

    Returns:
      scaled_data: dict with same keys as input mapped to torch.Tensor
    """
    if scaler is None:
        raise ValueError("scale_data_pytorch requires a scaler instance.")

    # ensure scaler is fitted
    if not scaler.fitted_:
        raise RuntimeError("Provided scaler instance is not fitted yet. Call fit or partial_fit first.")

    # normalize device argument
    if device is not None and not isinstance(device, torch.device):
        device = resolve_device(device)

    # transform all splits
    scaled_data = {}
    for split_name, split_array in data.items():
        split_tensor = convert_to_tensor(split_array).float()
        # move to scaler's device (which is the target device when provided)
        target_device = device if device is not None else scaler.mean_.device
        split_tensor = split_tensor.to(target_device)
        scaled_data[split_name] = scaler.transform(split_tensor)

    return scaled_data


def save_scaler(scaler: nn.Module, path: Union[str, "Path"]) -> None:
    """Save a scaler (TorchStandardScaler or TorchLogStandardScaler) to disk."""
    # rely on instance .save for consistent format
    if hasattr(scaler, "save"):
        scaler.save(path)
    else:
        from pathlib import Path
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"class": scaler.__class__.__name__, "state": scaler.state_dict()}, p)


def load_scaler(path: Union[str, Path], map_location: Any = None) -> nn.Module:
    """Load a scaler saved with save_scaler or scaler.save.

    Returns an instance of the correct scaler class.
    """
    data = torch.load(path, map_location=map_location, weights_only=True)
    cls_name = data.get("class", "")
    if cls_name == "TorchStandardScaler":
        return TorchStandardScaler.load(path, map_location=map_location)
    elif cls_name == "TorchLogStandardScaler":
        return TorchLogStandardScaler.load(path, map_location=map_location)
    else:
        raise ValueError(f"Unknown scaler class '{cls_name}' in saved file '{path}'.")
