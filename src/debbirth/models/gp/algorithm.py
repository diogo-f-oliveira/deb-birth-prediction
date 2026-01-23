from __future__ import annotations
import inspect
from typing import Any, Dict, Optional, Tuple, Union, Sequence

import numpy as np
from gplearn.genetic import SymbolicClassifier

from .config import TrainGPConfig, ClassWeight
from .constants import GPConstant


class DEBBirthSymbolicClassifier(SymbolicClassifier):
    """SymbolicClassifier with a fixed, named constant terminal set.

    Feature-name convention:
    - `feature_names`             : names **including** appended constants (what gplearn sees)
    - `feature_names_no_constants`: names for the *original* input features only

    This avoids any runtime swapping/resetting of `feature_names`.
    """

    def __init__(
            self,
            *,
            # --- our additions ---
            base_feature_names: Optional[Sequence[str]] = None,
            constants: Sequence[GPConstant] = (),
            # --- gplearn.genetic.SymbolicClassifier args (0.4.3) ---
            population_size: int = 1000,
            generations: int = 50,
            tournament_size: int = 50,
            const_range: Optional[Tuple[float, float]] = None,
            init_depth: Tuple[int, int] = (2, 6),
            init_method: str = "half and half",
            function_set: Sequence[Any] = ("add", "sub", "mul", "div"),
            transformer: str = "sigmoid",
            metric: str = "log loss",
            parsimony_coefficient: Union[float, str] = 0.001,
            p_crossover: float = 0.9,
            p_subtree_mutation: float = 0.01,
            p_hoist_mutation: float = 0.01,
            p_point_mutation: float = 0.01,
            p_point_replace: float = 0.05,
            class_weight: Any = None,
            max_samples: Union[float, int] = 1.0,
            feature_names: Optional[Sequence[str]] = None,
            low_memory: bool = False,
            n_jobs: int = 1,
            verbose: int = 0,
            random_state: Optional[int] = None,
    ):
        if base_feature_names is not None and feature_names is not None:
            raise ValueError("Pass only one of base_feature_names or feature_names (not both).")

        # Backwards-compat: allow users to pass feature_names like in gplearn.
        if base_feature_names is None and feature_names is not None:
            base_feature_names = feature_names

        self.base_feature_names: Optional[Tuple[str, ...]] = (
            tuple(base_feature_names) if base_feature_names is not None else None
        )
        self.constants: Tuple[GPConstant, ...] = tuple(constants)

        base = self.base_feature_names or ()
        const_names = tuple(c.name for c in self.constants)
        feature_names_with_constants = base + const_names

        # Public convenience attributes
        self.feature_names_no_constants: Tuple[str, ...] = base
        self.feature_names: Tuple[str, ...] = feature_names_with_constants

        super().__init__(
            population_size=population_size,
            generations=generations,
            tournament_size=tournament_size,
            stopping_criteria=0.0,
            const_range=const_range,
            init_depth=init_depth,
            init_method=init_method,
            function_set=function_set,
            transformer=transformer,
            metric=metric,
            parsimony_coefficient=parsimony_coefficient,
            p_crossover=p_crossover,
            p_subtree_mutation=p_subtree_mutation,
            p_hoist_mutation=p_hoist_mutation,
            p_point_mutation=p_point_mutation,
            p_point_replace=p_point_replace,
            max_samples=max_samples,
            class_weight=class_weight,
            feature_names=feature_names_with_constants,
            warm_start=False,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
        )

    def _augment_X(self, X: Any) -> np.ndarray:
        X = np.asarray(X)

        if len(self.constants) == 0:
            return X

        n = X.shape[0]
        const_vals = np.asarray([c.value for c in self.constants], dtype=X.dtype).reshape(1, -1)
        C = np.tile(const_vals, (n, 1))
        return np.concatenate([X, C], axis=1)

    # ---- sklearn-like API overrides ----

    def fit(self, X: Any, y: Any, sample_weight: Any = None):  # type: ignore[override]
        return super().fit(self._augment_X(X), y, sample_weight=sample_weight)

    # def predict(self, X: Any):  # type: ignore[override]
    #     return super().predict(self._augment_X(X))

    def predict_proba(self, X: Any):  # type: ignore[override]
        return super().predict_proba(self._augment_X(X))


def create_gp_classifier(cfg: TrainGPConfig) -> DEBBirthSymbolicClassifier:
    """Instantiate a DEBBirthSymbolicClassifier from TrainGPConfig."""

    base_feature_names = None
    if cfg.data_spec is not None:
        base_feature_names = tuple(cfg.data_spec.feature_cols)

    model = DEBBirthSymbolicClassifier(
        base_feature_names=base_feature_names,
        constants=cfg.gp.constants,
        population_size=cfg.gp.population_size,
        generations=cfg.gp.generations,
        tournament_size=cfg.gp.tournament_size,
        init_depth=cfg.gp.init_depth,
        init_method=cfg.gp.init_method,
        function_set=cfg.gp.function_set,
        transformer=cfg.gp.transformer,
        metric=cfg.gp.metric,
        parsimony_coefficient=cfg.gp.parsimony_coefficient,
        p_crossover=cfg.gp.p_crossover,
        p_subtree_mutation=cfg.gp.p_subtree_mutation,
        p_hoist_mutation=cfg.gp.p_hoist_mutation,
        p_point_mutation=cfg.gp.p_point_mutation,
        p_point_replace=cfg.gp.p_point_replace,
        low_memory=cfg.low_memory,
        n_jobs=int(cfg.num_workers),
        verbose=int(cfg.verbose),
        random_state=int(cfg.seed),
        class_weight=cfg.class_weights,
    )

    return model
