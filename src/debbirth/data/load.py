from __future__ import annotations

from typing import Dict

import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

from .schema import DatasetSpec
from .scalers import scale_data_pytorch
from ..models.nn.config import TrainDEBBirthNetConfig
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
    data = load_splits(dataset_dir=config.data_dir, split_type='train_val_test')
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



