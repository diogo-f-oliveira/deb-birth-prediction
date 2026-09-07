"""Torch conversion, scaling, and batching of shared prepared observations."""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from ...data.load import load_prepared_splits
from ...data.prepare import validate_numeric
from ...data.scalers import fit_and_scale_data_pytorch, scale_data_pytorch, TorchLogStandardScaler
from ...utils.pytorch import resolve_device


class PreparedTensorDataset(Dataset):
    """Explicit dictionary batches; row_index indexes this dataset's prepared record.

    Source strings and diagnostics stay in `prepared`; the numerical row index
    travels through the sampler together with x, y and the unscaled offset.
    """
    def __init__(self, prepared, features, targets):
        self.prepared = prepared
        self.features = features
        self.targets = targets
        self.offsets = (None if prepared.log_nu_b is None else
                        torch.tensor(prepared.log_nu_b, dtype=features.dtype, device=features.device))
        if self.offsets is not None and not torch.isfinite(self.offsets).all():
            raise ValueError("log_nu_b is not representable in the tensor dtype.")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        result = {"x": self.features[index], "y": self.targets[index], "row_index": index}
        if self.offsets is not None:
            result["log_nu_b"] = self.offsets[index]
        return result


def tensor_data_from_prepared(prepared, *, scaling_type="standardize", scaler=None,
                              batch_size=128, device="cpu", num_workers=0, seed=42,
                              aligned_batches=True):
    """Fit only on train; never transform offsets. Explicitly choose batch format."""
    device = resolve_device(device) if isinstance(device, str) else torch.device(device)
    if device.type != "cpu" and num_workers:
        raise ValueError("Device-resident tensor datasets require num_workers=0 outside CPU.")
    if not aligned_batches and any(p.log_nu_b is not None for p in prepared.values()):
        raise ValueError("Boundary preparation requires aligned dictionary batches.")
    features = {name: split.features for name, split in prepared.items()}
    needs_log = isinstance(scaler, TorchLogStandardScaler) or (scaler is None and scaling_type == "log_standardize")
    for name, split in prepared.items():
        validate_numeric(split.features, split.feature_names, positive=needs_log)
        # Catch overflow/underflow on float32 conversion before a scaler can hide it.
        with np.errstate(over="ignore", under="ignore"):
            converted = split.features.astype(np.float32)
        validate_numeric(converted, split.feature_names, positive=needs_log)
        underflow = (split.features != 0) & (converted == 0)
        if underflow.any():
            bad = [(int(row), split.feature_names[col]) for row, col in np.argwhere(underflow)[:10]]
            raise ValueError(f"Features underflow in float32 in {name}; invalid (row, column): {bad}")
    if scaler is None:
        scaled, scaler = fit_and_scale_data_pytorch(features, scaling_type, device=device)
    else:
        scaled = scale_data_pytorch(features, scaler, device=device)
    targets = {name: torch.tensor(p.labels, dtype=torch.float32, device=device) for name, p in prepared.items()}
    datasets, dataloaders = {}, {}
    for name, x in scaled.items():
        if not torch.isfinite(x).all():
            raise ValueError(f"Nonfinite scaled tensor features in {name}.")
        datasets[name] = (PreparedTensorDataset(prepared[name], x, targets[name]) if aligned_batches else
                          TensorDataset(x, targets[name]))
        # Legacy datasets still retain the complete record for provenance lookup.
        datasets[name].prepared = prepared[name]
        dataloaders[name] = DataLoader(datasets[name], batch_size=batch_size if name == "train" else 1024,
                                      shuffle=name == "train", num_workers=num_workers,
                                      generator=torch.Generator().manual_seed(seed))
    return scaled, targets, dataloaders, datasets, scaler


def load_data_pytorch(config, scaler=None, *, return_prepared=False):
    prepared, metadata = load_prepared_splits(config.data_dir, config.data_spec, config.data_splits)
    output = tensor_data_from_prepared(
        prepared, scaling_type=config.scaling_type, scaler=scaler, batch_size=config.batch_size,
        device=config.device, num_workers=config.num_workers, seed=config.seed,
        aligned_batches=return_prepared and config.data_spec.formulation == "boundary")
    return (*output, prepared, metadata) if return_prepared else output
