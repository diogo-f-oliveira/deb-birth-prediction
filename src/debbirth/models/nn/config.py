from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Any
from pathlib import Path

from ...data.schema import DatasetSpec
from ...utils.paths import resolve_repo_path
from ...utils.config import save_config_json
from ...formulations import validate_temperature
from ...data.load import SPLIT_TYPES


@dataclass
class TrainDEBBirthNetConfig:
    # Data
    data_spec: DatasetSpec
    data_splits: str = "train_val_test"  # train_val_test | train_test
    data_dir: Path = Path("data/processed")

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

    # Output
    outdir: Optional[Path] = None
    boundary_temperature: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "data_dir", resolve_repo_path(self.data_dir))
        if self.outdir is not None:
            object.__setattr__(self, "outdir", resolve_repo_path(self.outdir))
        if self.data_splits not in SPLIT_TYPES:
            raise ValueError(f"Unknown data_splits {self.data_splits!r}; expected {SPLIT_TYPES}.")
        if self.scaling_type not in (None, "none", "standardize", "log_standardize"):
            raise ValueError(f"Unknown scaling_type {self.scaling_type!r}.")
        validate_temperature(self.boundary_temperature)
        if self.data_spec.formulation != "boundary" and self.boundary_temperature != 1.0:
            raise ValueError("boundary_temperature applies only to boundary formulations.")

    def save_json(self, path) -> None:
        save_config_json(self, path)

    @classmethod
    def load_json(cls, path: Any) -> TrainDEBBirthNetConfig:
        """
        Load a TrainDEBBirthNetConfig from a JSON file, converting nested structures.

        Raises for malformed settings; no output directories are created.
        """
        p = Path(path)

        raw = json.loads(p.read_text(encoding="utf-8"))

        # Convert nested 'net_config' dict to DEBBirthNetConfig if present
        net = raw.get("net_config")
        if isinstance(net, dict):
            raw["net_config"] = DEBBirthNetConfig(**net)

        # Convert 'data_spec' dict to DatasetSpec if present
        ds = raw.get("data_spec")
        if isinstance(ds, dict):
            raw["data_spec"] = DatasetSpec.from_dict(ds)

        # Instantiate TrainDEBBirthNetConfig (post-init will coerce paths)
        return cls(**raw)


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

    def save_json(self, path) -> None:
        save_config_json(self, path)
