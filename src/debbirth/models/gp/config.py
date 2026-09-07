from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Union, Mapping
import json

from .functions import DEFAULT_FUNCTION_SET, GPFunctionSet, primitive_identifier, resolve_primitive
from .constants import GPConstantSet, DEFAULT_CONSTANT_SET, resolve_constant
from ...data.schema import DatasetSpec
from ...utils.paths import resolve_repo_path
from ...utils.config import save_config_json
from ...formulations import validate_temperature
from ...data.load import SPLIT_TYPES


@dataclass(frozen=True)
class GPConfig:
    """Hyperparameters for gplearn.genetic.SymbolicClassifier."""

    # Evolution
    population_size: int = 1000
    generations: int = 50
    tournament_size: int = 50

    # Primitives
    function_set: GPFunctionSet = DEFAULT_FUNCTION_SET

    # Names resolve to internal constant terminals during construction.
    constants: GPConstantSet | Tuple[str, ...] = DEFAULT_CONSTANT_SET

    # Initialization
    init_depth: Tuple[int, int] = (6, 10)
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


    def __post_init__(self):
        object.__setattr__(self, "function_set", tuple(resolve_primitive(p) for p in self.function_set))
        object.__setattr__(self, "constants", tuple(resolve_constant(c) for c in self.constants))
        object.__setattr__(self, "init_depth", tuple(self.init_depth))
        if len({c.name for c in self.constants}) != len(self.constants):
            raise ValueError("GP constant names must be unique.")

    def to_dict(self):
        from dataclasses import fields
        result = {field.name: getattr(self, field.name) for field in fields(self)}
        result["function_set"] = [primitive_identifier(p) for p in self.function_set]
        result["constants"] = [c.name for c in self.constants]
        result["init_depth"] = list(self.init_depth)
        return result


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

    class_weights: ClassWeight = "balanced"

    seed: int = 42
    num_workers: int = 1
    boundary_temperature: float = 1.0
    scaling_type: str = "none"  # The historical GP path is explicitly unscaled.

    def __post_init__(self):
        object.__setattr__(self, "data_dir", resolve_repo_path(self.data_dir))
        if self.scaling_type != "none":
            raise ValueError("The current GP adapter supports only scaling_type='none'.")
        if self.outdir is not None:
            object.__setattr__(self, "outdir", resolve_repo_path(self.outdir))
        if self.data_splits not in SPLIT_TYPES:
            raise ValueError(f"Unknown data_splits {self.data_splits!r}; expected {SPLIT_TYPES}.")
        validate_temperature(self.boundary_temperature)
        if self.data_spec.formulation != "boundary" and self.boundary_temperature != 1.0:
            raise ValueError("boundary_temperature applies only to boundary formulations.")

    def save_json(self, path) -> None:
        save_config_json(self, path)

    @classmethod
    def load_json(cls, path: Any) -> TrainGPConfig:
        """Load a TrainGPConfig from a JSON file, converting nested structures.

        Raises for malformed or unreconstructable training settings.
        """
        p = Path(path)
        raw = json.loads(p.read_text(encoding="utf-8"))

        # Convert nested 'gp' dict to GPConfig if present
        gp_dict = raw.get("gp")
        if isinstance(gp_dict, dict):
            raw["gp"] = GPConfig(**gp_dict)

        # Convert 'data_spec' dict to DatasetSpec if present
        ds = raw.get("data_spec")
        if isinstance(ds, dict):
            raw["data_spec"] = DatasetSpec.from_dict(ds)

        if isinstance(raw.get("class_weights"), dict):
            raw["class_weights"] = {int(key): value for key, value in raw["class_weights"].items()}
        # Instantiate without creating a run directory
        return cls(**raw)

