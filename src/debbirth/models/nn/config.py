from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Any
from pathlib import Path

from ...data.schema import DatasetSpec
from ...utils.results import create_run_outdir  # new import


@dataclass
class TrainDEBBirthNetConfig:
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

    # Output
    outdir: Optional[Path] = None

    def __post_init__(self):

        # Ensure data_dir is an absolute Path so trials find data regardless of CWD
        if not isinstance(self.data_dir, Path):
            object.__setattr__(self, "data_dir", Path(self.data_dir))
        # If data_dir is relative, resolve it against the repository root (not the trial CWD)
        if not self.data_dir.is_absolute():
            # Locate repository root by finding the ancestor named 'src' and taking its parent.
            this_file = Path(__file__).resolve()
            repo_root = None
            for anc in this_file.parents:
                if anc.name == "src":
                    repo_root = anc.parent
                    break
            if repo_root is None:
                repo_root = Path.cwd()
            resolved_data_dir = (repo_root / self.data_dir).resolve()
        else:
            resolved_data_dir = self.data_dir.resolve()

        object.__setattr__(self, "data_dir", resolved_data_dir)

        # If outdir was not provided, create a timestamped run dir (safe filename) under cwd.
        if self.outdir is None:
            model_name = "DEBBirthNet"
            run_dir = create_run_outdir(model_name)
            object.__setattr__(self, "outdir", run_dir)
        else:
            # Coerce outdir to a Path (accept strings or Paths)
            if not isinstance(self.outdir, Path):
                object.__setattr__(self, "outdir", Path(self.outdir))

    def save_json(self, path) -> None:
        """Save this TrainDEBBirthNetConfig to a JSON file (creates parent dirs)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True, default=str)

    @classmethod
    def load_json(cls, path: Any) -> TrainDEBBirthNetConfig:
        """
        Load a TrainDEBBirthNetConfig from a JSON file, converting nested structures.

        Returns a TrainDEBBirthNetConfig instance or None if loading/parsing fails.
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
            raw["data_spec"] = DatasetSpec(**ds)

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
        """Save this DEBBirthNetConfig to a JSON file (creates parent dirs)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True, default=str)
