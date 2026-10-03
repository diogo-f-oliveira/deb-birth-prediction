"""Production fixed-offset GP boundary (T06): evolve F(gamma, k[, x_b]) only.

gplearn's SymbolicRegressor is the evolution engine; its raw output is F.
The metric is the T04 fixed-offset BCE, bound to one PreparedSplit: maturity
enters only as log_nu_b inside that metric, never as a tree terminal or in the
labels. Named constants are appended as constant columns, as in
DEBBirthSymbolicClassifier, and no ephemeral random constants are used, so the
program text is exact. Final-program selection is gplearn's: the lowest raw
(unpenalized) training loss in the last generation. The number of generations
is the tuned quantity; parsimony acts only in tournaments.
"""
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from math import isfinite

import numpy as np
from gplearn._program import _Program
from gplearn.fitness import _Fitness
from gplearn.genetic import SymbolicRegressor
from scipy.special import expit
from sklearn.utils.class_weight import compute_sample_weight

from .boundary_prototype import check_backend_contract, weighted_boundary_bce
from .constants import GPConstantSet
from .generation_hook import GenerationCallbackMixin, fit_with_generation_callback
from ...data.prepare import PreparedSplit
from ...formulations import boundary_is_feasible, boundary_margin, validate_temperature

BOUNDARY_FEATURES = (("gamma", "k"), ("gamma", "k", "x_b"))


class BoundaryEngine(GenerationCallbackMixin, SymbolicRegressor):
    """SymbolicRegressor that accepts a per-generation callback; evolution is unchanged."""


def _bound_loss(y, F, w, *, expected_y, log_nu_b, temperature):
    if not np.array_equal(y, expected_y):
        raise ValueError("Boundary fitness context mismatch: fit a newly selected PreparedSplit.")
    return weighted_boundary_bce(y, F, w, log_nu_b=log_nu_b, temperature=temperature)


def _augment(X, constants):
    if not constants:
        return X
    values = np.asarray([c.value for c in constants], dtype=float)
    return np.concatenate([X, np.broadcast_to(values, (len(X), len(values)))], axis=1)


def _validated_features(X, n_features):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != n_features:
        raise ValueError("Incorrect prepared boundary feature shape.")
    if not np.isfinite(X).all() or (X <= 0).any():
        raise ValueError("Expected finite positive prepared boundary inputs.")
    return X


@dataclass(frozen=True)
class GPBoundaryModel:
    """Inference artifact: executable tree, feature order, constants and T.

    It holds no training labels, offsets or metric. Inputs are prepared
    (unscaled) boundary features; log_nu_b is supplied separately. Equality
    (zero margin) is infeasible, with no tolerance or tuned threshold.
    """
    program: _Program
    feature_names: tuple[str, ...]
    constants: GPConstantSet
    temperature: float

    @classmethod
    def from_engine(cls, engine, *, feature_names, constants, temperature):
        validate_temperature(temperature)
        program = deepcopy(engine._program)
        program.metric = None  # execution needs neither the metric nor labels
        return cls(program, tuple(feature_names), tuple(constants), temperature)

    @property
    def expression(self) -> str:
        """Exact gplearn syntax with feature/constant names; no simplification."""
        return str(self.program)

    @property
    def length(self) -> int:
        return self.program.length_

    def predict_boundary(self, X):
        """F = log(Psi) for prepared boundary features."""
        X = _validated_features(X, len(self.feature_names))
        with np.errstate(all="ignore"):
            F = self.program.execute(_augment(X, self.constants))
        if not np.isfinite(F).all():
            raise ValueError("Boundary expression produced nonfinite output.")
        return F

    def predict_margin(self, X, log_nu_b):
        offset = np.asarray(log_nu_b, dtype=float)
        if not np.isfinite(offset).all():
            raise ValueError("Expected finite log_nu_b.")
        return boundary_margin(self.predict_boundary(X), offset)

    def predict_proba(self, X, log_nu_b):
        """Positive-class probability, shape (N,)."""
        return expit(self.predict_margin(X, log_nu_b) / self.temperature)

    def predict(self, X, log_nu_b):
        return boundary_is_feasible(self.predict_margin(X, log_nu_b)).astype(int)


def fit_gp_boundary(split: PreparedSplit, cfg, *, generation_callback=None):
    """Evolve F on one training PreparedSplit; return (GPBoundaryModel, engine).

    The engine carries the dataset-bound metric and run history; use it only
    for diagnostics of this fit. Weights follow cfg.class_weights (None means
    ordinary mean BCE over active rows). generation_callback(engine,
    run_details) runs after each generation (see generation_hook.py).
    """
    check_backend_contract()
    gp = cfg.gp
    names = tuple(split.feature_names)
    if cfg.data_spec.formulation != "boundary" or names not in BOUNDARY_FEATURES \
            or names != tuple(cfg.data_spec.feature_cols):
        raise ValueError("Boundary trees accept gamma, k and optionally x_b only, in the configured order.")
    const_names = tuple(c.name for c in gp.constants)
    if set(const_names) & set(names):
        raise ValueError("Constant names must not coincide with boundary feature names.")
    if split.log_nu_b is None:
        raise ValueError("Boundary preparation must carry log_nu_b separately.")
    if (gp.metric, gp.transformer) != ("log loss", "sigmoid"):
        raise ValueError("The boundary loss is fixed: BCE of sigmoid((F - log_nu_b)/T); "
                         "use metric='log loss' and transformer='sigmoid'.")
    if isinstance(gp.parsimony_coefficient, str) or not isfinite(gp.parsimony_coefficient) \
            or gp.parsimony_coefficient < 0:
        raise ValueError("Boundary GP needs a finite nonnegative fixed parsimony coefficient "
                         "('auto' is undefined when invalid candidates have infinite loss).")
    temperature = cfg.boundary_temperature
    validate_temperature(temperature)

    X, y, offset = (np.array(a, dtype=float, copy=True) for a in (split.features, split.labels, split.log_nu_b))
    X = _validated_features(X, len(names))
    if set(np.unique(y)) != {0.0, 1.0} or not np.isfinite(offset).all():
        raise ValueError("Boundary training requires both binary classes and finite offsets.")
    for a in (X, y, offset):
        a.setflags(write=False)
    weights = (np.ones(len(y)) if cfg.class_weights is None
               else compute_sample_weight(cfg.class_weights, y))  # training rows only
    # Direct _Fitness construction avoids make_fitness's two-row probe (see T04 note).
    metric = _Fitness(partial(_bound_loss, expected_y=y, log_nu_b=offset, temperature=temperature),
                      greater_is_better=False)
    engine = BoundaryEngine(
        population_size=gp.population_size, generations=gp.generations, tournament_size=gp.tournament_size,
        stopping_criteria=0.0, const_range=None, init_depth=gp.init_depth, init_method=gp.init_method,
        function_set=gp.function_set, metric=metric, parsimony_coefficient=gp.parsimony_coefficient,
        p_crossover=gp.p_crossover, p_subtree_mutation=gp.p_subtree_mutation,
        p_hoist_mutation=gp.p_hoist_mutation, p_point_mutation=gp.p_point_mutation,
        p_point_replace=gp.p_point_replace, max_samples=1.0, feature_names=list(names + const_names),
        warm_start=False, low_memory=cfg.low_memory, n_jobs=int(cfg.num_workers), verbose=int(cfg.verbose),
        random_state=int(cfg.seed),
    )
    fit_with_generation_callback(engine, generation_callback, _augment(X, gp.constants), y, sample_weight=weights)
    if not np.isfinite(engine._program.raw_fitness_):
        raise RuntimeError("No finite boundary candidate in the last generation.")
    model = GPBoundaryModel.from_engine(engine, feature_names=names, constants=gp.constants,
                                        temperature=temperature)
    return model, engine
