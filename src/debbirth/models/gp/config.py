from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional, Tuple, Union, Mapping
import json

from .functions import DEFAULT_FUNCTIONS
from .constants import GPConstant, DEFAULT_CONSTANTS
from ...data.schema import DatasetSpec
from ...utils.results import create_run_outdir


@dataclass(frozen=True)
class GPConfig:
    """Hyperparameters for gplearn.genetic.SymbolicClassifier."""

    # Evolution
    population_size: int = 1000
    generations: int = 50
    tournament_size: int = 50

    # Primitives
    function_set: Tuple[Any, ...] = DEFAULT_FUNCTIONS

    # Discrete constants: implemented as constant-valued feature columns.
    constants: Tuple[GPConstant, ...] = DEFAULT_CONSTANTS

    # Initialization
    init_depth: Tuple[int, int] = (2, 6)
    init_method: str = "half and half"  # {'grow','full','half and half'}

    # Classification specifics
    transformer: str = "sigmoid"  # default in gplearn
    metric: str = "log loss"  # raw fitness metric for classifier

    # Parsimony / regularization
    parsimony_coefficient: Union[float, str] = 0.001  # float or 'auto'

    # Genetic operation probabilities (reproduction uses remaining probability)
    p_crossover: float = 0.9
    p_subtree_mutation: float = 0.01
    p_hoist_mutation: float = 0.01
    p_point_mutation: float = 0.01
    p_point_replace: float = 0.05


ClassWeight = Union[None, str, Mapping[int, float], Mapping[bool, float]]


@dataclass(frozen=True)
class TrainGPConfig:
    """Single run config: GP hyperparameters + runtime settings."""
    # Nested GP hyperparameters
    gp: GPConfig

    # Data
    data_spec: DatasetSpec
    data_splits: str = "train_val_test"  # train_val_test | train_test
    data_dir: Path = Path("data/processed")

    # Runtime / experiment settings
    outdir: Optional[Path] = None
    # optional run name / model identifier used when auto-creating the run directory
    run_name: Optional[str] = None

    low_memory: bool = False
    verbose: int = 0

    class_weights: ClassWeight = None

    seed: int = 42
    num_workers: int = 1

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

        # Coerce outdir to a Path (accept strings or Paths) or create one if None
        if self.outdir is None:
            # build a timestamped run directory using run_name or fallback to 'DEBBirthSymbolicClassifier'
            model_name = self.run_name if self.run_name else "DEBBirthSymbolicClassifier"
            run_dir = create_run_outdir(model_name)
            object.__setattr__(self, "outdir", run_dir)
        else:
            if not isinstance(self.outdir, Path):
                object.__setattr__(self, "outdir", Path(self.outdir))

    def save_json(self, path) -> None:
        """Save this TrainGPConfig to a JSON file (creates parent dirs)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True, default=str)
