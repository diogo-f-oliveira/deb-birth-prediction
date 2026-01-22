from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

from .schema import DatasetSpec
from .scalers import scale_data_pytorch
from ..models.nn.config import TrainDEBBirthNetConfig
from ..models.gp.config import TrainGPConfig
from ..utils.pytorch import convert_to_tensor

def load_splits(dataset_dir: str, split_type='train_val_test') -> Dict[str, pd.DataFrame]:
    """
    Load pre-made splits from disk.

    Expected: three CSV files with identical schema (features + label column).
    """
    data_splits = {}
    for split in ['train', 'val', 'test']:
        data_splits[split] = pd.read_csv(f"{dataset_dir}/{split}.csv")
    if split_type == 'train_val_test':
        return data_splits
    elif split_type == 'train_test':
        combined_train = pd.concat([data_splits['train'], data_splits['val']], axis=0).reset_index(drop=True)
        return {
            'train': combined_train,
            'test': data_splits['test'],
        }
    return {}

def get_features_targets(data: Dict[str, pd.DataFrame], data_spec: DatasetSpec):
    features = {split: df[data_spec.feature_cols] for split, df in data.items()}
    targets = {split: df[data_spec.target_col] for split, df in data.items()}
    return features, targets


def load_data_pytorch(config: TrainDEBBirthNetConfig):
    # Load dataframes
    data = load_splits(dataset_dir=config.data_dir, split_type=config.data_splits)
    # Extract input features and scale
    features, targets = get_features_targets(data=data, data_spec=config.data_spec)
    scaled_input_data, scalers = scale_data_pytorch(features, scaling_type=config.scaling_type)
    # Extract output and convert to tensors
    targets_tensor = {split: convert_to_tensor(df.astype(float)) for split, df in targets.items()}

    datasets = {}
    dataloaders = {}
    for split in scaled_input_data:
        # Create dataset
        datasets[split] = TensorDataset(
            scaled_input_data[split],
            targets_tensor[split]
        )
        # Create dataloader
        dataloaders[split] = DataLoader(
            datasets[split],
            batch_size=config.batch_size if split == 'train' else 1024,
            shuffle=True if split == 'train' else False,
        )

    return scaled_input_data, targets_tensor, dataloaders, datasets, scalers

def load_data_gp(cfg: TrainGPConfig) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Load splits and return features and targets as numpy arrays for GP training.

    The function accepts a TrainGPConfig and extracts:
      - dataset directory from cfg.dataset_spec.data_dir if present,
        otherwise falls back to cfg.output_dir or current directory.
      - uses default split type 'train_val_test'.

    Returns:
      features_np: dict[split] -> np.ndarray (float)
      targets_np:  dict[split] -> np.ndarray (int)
    """
    # Try to obtain dataset directory from the DatasetSpec, then cfg.output_dir, then cwd
    dataset_dir = cfg.data_dir

    data = load_splits(dataset_dir=dataset_dir, split_type="train_val_test")
    features, targets = get_features_targets(data=data, data_spec=cfg.data_spec)

    features_np: Dict[str, np.ndarray] = {split: df.values.astype(float) for split, df in features.items()}

    targets_np: Dict[str, np.ndarray] = {}
    for split, t in targets.items():
        arr = t.values if hasattr(t, "values") else np.asarray(t)
        targets_np[split] = arr.astype(int).ravel()

    return features_np, targets_np
