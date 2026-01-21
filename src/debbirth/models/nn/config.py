from __future__ import annotations

from dataclasses import dataclass
from typing import List
from pathlib import Path

from ...data.schema import DatasetSpec


@dataclass
class TrainDEBBirthNetConfig:
    # Output
    outdir: Path

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
    scaling_type: str = "standardize"  # none | standardize | log_standardize

    # Imbalance handling
    use_pos_weight: bool = False
    pos_weight: float = None

    # Run settings
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"  # auto | cpu | cuda

    def __post_init__(self):
        # Coerce outdir to a Path (accept strings or Paths)
        if not isinstance(self.outdir, Path):
            object.__setattr__(self, "outdir", Path(self.outdir))


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
