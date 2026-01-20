from __future__ import annotations

from dataclasses import dataclass

from src.debbirth.data.schema import DatasetSpec
from src.debbirth.models.nn.structure import DEBBirthNetConfig


@dataclass
class TrainDEBBirthNetConfig:
    # Output
    outdir: str

    # Data
    data_spec: DatasetSpec
    data_splits: str = "train_val_test"  # train_val_test | train_test
    data_dir: str = "data/processed"

    # Training
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-2

    # Model architecture
    net_config: DEBBirthNetConfig = None
    scaling_type: str = "standardize"  # none | standard

    # Imbalance handling
    use_pos_weight: bool = False
    pos_weight: float = None

    # Run settings
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"  # auto | cpu | cuda
