import torch
from torch import nn

from ..utils.pytorch import convert_to_tensor, resolve_device


class TorchStandardScaler(nn.Module):
    """
    A PyTorch-friendly StandardScaler similar to sklearn.preprocessing.StandardScaler.

    - Registers mean_, var_, scale_ and n_samples_seen_ as buffers so they move with .to(device)
    - Methods: fit, partial_fit, transform, inverse_transform, forward
    - Accepts torch.Tensor or numpy arrays for fit/transform.
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True, eps: float = 1e-8):
        super().__init__()
        self.with_mean = with_mean
        self.with_std = with_std
        self.eps = float(eps)

        # Registered buffers (will be saved/loaded and moved with .to())
        # mean_ and var_ will be set on fit; initialized to empty tensors
        self.register_buffer("mean_", torch.tensor([]))
        self.register_buffer("var_", torch.tensor([]))
        self.register_buffer("scale_", torch.tensor([]))
        # number of samples seen so far (scalar int)
        self.register_buffer("n_samples_seen_", torch.tensor(0, dtype=torch.long))

    def fit(self, X, sample_dim: int = 0):
        """
        Compute mean and std on X.
        sample_dim: dimension that indexes samples (default 0), features are the remaining dims (commonly 1).
        """
        X = convert_to_tensor(X).float()
        if X.numel() == 0:
            raise ValueError("Empty array passed to fit()")

        # move computation to the device of this module's buffers (if they are on some device)
        device = self.mean_.device if self.mean_.numel() else X.device
        X = X.to(device)

        # compute mean and var (population variance, unbiased=False) along sample_dim
        mean = X.mean(dim=sample_dim)
        var = X.var(dim=sample_dim, unbiased=False)

        if not self.with_mean:
            mean = torch.zeros_like(mean)
        if not self.with_std:
            var = torch.ones_like(var)

        scale = torch.sqrt(var + self.eps)
        # avoid zeros
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        self.mean_ = mean.detach()
        self.var_ = var.detach()
        self.scale_ = scale.detach()
        # set sample count
        n = X.shape[sample_dim]
        self.n_samples_seen_ = torch.tensor(int(n), dtype=torch.long, device=device)
        return self

    def partial_fit(self, X, sample_dim: int = 0):
        """
        Incrementally update running mean/var using batch X.
        """
        X = convert_to_tensor(X).float()
        if X.numel() == 0:
            return self

        device = self.mean_.device if self.mean_.numel() else X.device
        X = X.to(device)

        n_new = X.shape[sample_dim]
        mean_new = X.mean(dim=sample_dim)
        var_new = X.var(dim=sample_dim, unbiased=False)

        if not self.with_mean:
            mean_new = torch.zeros_like(mean_new)
        if not self.with_std:
            var_new = torch.ones_like(var_new)

        if self.n_samples_seen_.numel() == 0 or int(self.n_samples_seen_) == 0:
            # nothing seen before
            self.mean_ = mean_new.detach()
            self.var_ = var_new.detach()
            self.scale_ = torch.sqrt(self.var_ + self.eps)
            self.scale_ = torch.where(self.scale_ == 0, torch.ones_like(self.scale_), self.scale_)
            self.n_samples_seen_ = torch.tensor(int(n_new), dtype=torch.long, device=device)
            return self

        # combine existing stats with new batch
        n_old = int(self.n_samples_seen_)
        mean_old = self.mean_.to(device)
        var_old = self.var_.to(device)

        n_total = n_old + n_new
        mean_total = (mean_old * n_old + mean_new * n_new) / n_total

        # combine variances (population variances)
        delta_old = mean_old - mean_total
        delta_new = mean_new - mean_total
        var_total = (n_old * var_old + n_new * var_new + n_old * (delta_old ** 2) + n_new * (delta_new ** 2)) / n_total

        self.mean_ = mean_total.detach()
        self.var_ = var_total.detach()
        self.scale_ = torch.sqrt(self.var_ + self.eps)
        self.scale_ = torch.where(self.scale_ == 0, torch.ones_like(self.scale_), self.scale_)
        self.n_samples_seen_ = torch.tensor(int(n_total), dtype=torch.long, device=device)
        return self

    def transform(self, X):
        """
        Center and scale X. Returns a tensor on same device/dtype as the scaler buffers.
        """
        if self.n_samples_seen_.numel() == 0 or int(self.n_samples_seen_) == 0:
            raise RuntimeError("TorchStandardScaler not fitted yet. Call fit or partial_fit first.")

        X = convert_to_tensor(X).float()
        device = self.mean_.device
        X = X.to(device)

        # support broadcasting for mean/scale across sample dimension
        return (X - self.mean_) / self.scale_

    def inverse_transform(self, X):
        if self.n_samples_seen_.numel() == 0 or int(self.n_samples_seen_) == 0:
            raise RuntimeError("TorchStandardScaler not fitted yet. Call fit or partial_fit first.")
        X = convert_to_tensor(X).float()
        device = self.mean_.device
        X = X.to(device)
        return X * self.scale_ + self.mean_

    # allow use in nn.Sequential / modules
    def forward(self, X):
        return self.transform(X)

    @property
    def fitted_(self) -> bool:
        return bool(self.n_samples_seen_.numel() and int(self.n_samples_seen_) > 0)



# New: log then standardize scaler
class TorchLogStandardScaler(nn.Module):
    """
    Scaler that applies a log transform (log(x)) followed by standardization.

    - Uses log (not log1p) because inputs are positive by default.
    - Mirrors TorchStandardScaler API: fit, partial_fit, transform, inverse_transform, forward, fitted_.
    - All statistics (mean_, var_, scale_, n_samples_seen_) are registered as buffers.
    """
    def __init__(self, with_mean: bool = True, with_std: bool = True, eps: float = 1e-8):
        super().__init__()
        self.with_mean = with_mean
        self.with_std = with_std
        self.eps = float(eps)

        self.register_buffer("mean_", torch.tensor([]))
        self.register_buffer("var_", torch.tensor([]))
        self.register_buffer("scale_", torch.tensor([]))
        self.register_buffer("n_samples_seen_", torch.tensor(0, dtype=torch.long))

    def _to_tensor(self, X):
        return convert_to_tensor(X).float()

    def _log_transform(self, X: torch.Tensor) -> torch.Tensor:
        # use natural log; inputs are expected to be positive
        return torch.log(X)

    def _inv_log_transform(self, X: torch.Tensor) -> torch.Tensor:
        # inverse of log is exp
        return torch.exp(X)

    def fit(self, X, sample_dim: int = 0):
        X = self._to_tensor(X)
        if X.numel() == 0:
            raise ValueError("Empty array passed to fit()")

        device = self.mean_.device if self.mean_.numel() else X.device
        X = X.to(device)

        X_log = self._log_transform(X)

        mean = X_log.mean(dim=sample_dim)
        var = X_log.var(dim=sample_dim, unbiased=False)

        if not self.with_mean:
            mean = torch.zeros_like(mean)
        if not self.with_std:
            var = torch.ones_like(var)

        scale = torch.sqrt(var + self.eps)
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        self.mean_ = mean.detach()
        self.var_ = var.detach()
        self.scale_ = scale.detach()
        n = X.shape[sample_dim]
        self.n_samples_seen_ = torch.tensor(int(n), dtype=torch.long, device=device)
        return self

    def partial_fit(self, X, sample_dim: int = 0):
        X = self._to_tensor(X)
        if X.numel() == 0:
            return self

        device = self.mean_.device if self.mean_.numel() else X.device
        X = X.to(device)

        n_new = X.shape[sample_dim]
        X_log = self._log_transform(X)
        mean_new = X_log.mean(dim=sample_dim)
        var_new = X_log.var(dim=sample_dim, unbiased=False)

        if not self.with_mean:
            mean_new = torch.zeros_like(mean_new)
        if not self.with_std:
            var_new = torch.ones_like(var_new)

        if self.n_samples_seen_.numel() == 0 or int(self.n_samples_seen_) == 0:
            self.mean_ = mean_new.detach()
            self.var_ = var_new.detach()
            self.scale_ = torch.sqrt(self.var_ + self.eps)
            self.scale_ = torch.where(self.scale_ == 0, torch.ones_like(self.scale_), self.scale_)
            self.n_samples_seen_ = torch.tensor(int(n_new), dtype=torch.long, device=device)
            return self

        n_old = int(self.n_samples_seen_)
        mean_old = self.mean_.to(device)
        var_old = self.var_.to(device)

        n_total = n_old + n_new
        mean_total = (mean_old * n_old + mean_new * n_new) / n_total

        delta_old = mean_old - mean_total
        delta_new = mean_new - mean_total
        var_total = (n_old * var_old + n_new * var_new + n_old * (delta_old ** 2) + n_new * (delta_new ** 2)) / n_total

        self.mean_ = mean_total.detach()
        self.var_ = var_total.detach()
        self.scale_ = torch.sqrt(self.var_ + self.eps)
        self.scale_ = torch.where(self.scale_ == 0, torch.ones_like(self.scale_), self.scale_)
        self.n_samples_seen_ = torch.tensor(int(n_total), dtype=torch.long, device=device)
        return self

    def transform(self, X):
        if self.n_samples_seen_.numel() == 0 or int(self.n_samples_seen_) == 0:
            raise RuntimeError("TorchLogStandardScaler not fitted yet. Call fit or partial_fit first.")

        X = self._to_tensor(X)
        device = self.mean_.device
        X = X.to(device)

        X_log = self._log_transform(X)
        return (X_log - self.mean_) / self.scale_

    def inverse_transform(self, X):
        if self.n_samples_seen_.numel() == 0 or int(self.n_samples_seen_) == 0:
            raise RuntimeError("TorchLogStandardScaler not fitted yet. Call fit or partial_fit first.")
        X = self._to_tensor(X)
        device = self.mean_.device
        X = X.to(device)
        X_unscaled = X * self.scale_ + self.mean_
        return self._inv_log_transform(X_unscaled)

    def forward(self, X):
        return self.transform(X)

    @property
    def fitted_(self) -> bool:
        return bool(self.n_samples_seen_.numel() and int(self.n_samples_seen_) > 0)


def scale_data_pytorch(data, scaling_type: str, device=None):
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

    Raises:
      ValueError if 'train' split is missing or if an unknown scaling_type is provided.
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

    # transform all splits
    scaled_data = {}
    for split_name, split_array in data.items():
        split_tensor = convert_to_tensor(split_array).float()
        # move to scaler's device (which is the target device when provided)
        target_device = device if device is not None else scaler.mean_.device
        split_tensor = split_tensor.to(target_device)
        scaled_data[split_name] = scaler.transform(split_tensor)

    return scaled_data, scaler
