from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple, Union, Mapping

from ...data.schema import DatasetSpec
from .constants import GPConstant, DEFAULT_CONSTANTS


@dataclass(frozen=True)
class GPConfig:
    """Hyperparameters for gplearn.genetic.SymbolicClassifier."""

    # Evolution
    population_size: int = 1000
    generations: int = 50
    tournament_size: int = 50

    # Primitives
    function_set: Tuple[Any, ...] = ("add", "sub", "mul", "div")

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
    data_dir: str = "data/processed"

    # Runtime / experiment settings
    outdir: Path = Path(".")
    low_memory: bool = False
    verbose: int = 0

    class_weights: ClassWeight = None

    seed: int = 42
    num_workers: int = 1

    def __post_init__(self):
        # Coerce outdir to a Path (accept strings or Paths)
        if not isinstance(self.outdir, Path):
            object.__setattr__(self, "outdir", Path(self.outdir))
