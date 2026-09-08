"""T04 feasibility prototype, deliberately separate from the T06 trainer.

gplearn evolves F with binary reached_birth targets and a separately bound
offset. The audited backend evaluates full arrays and subsamples by weights.
No maturity or row identifier is a tree terminal; no labels encode offsets.
"""
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
import inspect

import gplearn
import numpy as np
from gplearn._program import _Program
from gplearn.fitness import _Fitness
from gplearn.genetic import BaseSymbolic, SymbolicRegressor, _parallel_evolve
from scipy.special import expit
from sklearn.utils.class_weight import compute_sample_weight

from ...data.prepare import PreparedSplit
from ...formulations import boundary_is_feasible, boundary_margin, output_to_logit, validate_temperature


# Fail closed if the implementation on which row alignment depends changes.
# This narrow prototype guard is not a general backend compatibility layer.
_AUDITED_SOURCE = {
    BaseSymbolic.fit: "3b833f6165bfb2b03d3e359d23d1d1376a08e434a4f0b4629955936416b44bd3",
    _parallel_evolve: "4641ad2918c6b02bd53d6e54cb448f11f7272f6d6f713ba0a647a5d3d89f5b66",
    _Program.raw_fitness: "cf53d394ac4627014db3d8ba5c7460e7ec848b1bdc969f3fbe3e16ceec29b5c8",
    _Program.execute: "2565f6bd1c13a3704acb4311721a02e88cadd40dde76ce87b58095d340eee8da",
}


def check_backend_contract():
    actual = {fn.__qualname__: sha256(inspect.getsource(fn).encode()).hexdigest()
              for fn in _AUDITED_SOURCE}
    if gplearn.__version__ != "0.4.3" or any(
            actual[fn.__qualname__] != digest for fn, digest in _AUDITED_SOURCE.items()):
        raise RuntimeError("Re-audit gplearn row/mask semantics before using the T04 prototype.")
    return actual


def weighted_boundary_bce(y, learned_boundary, weights, *, log_nu_b, temperature):
    """Weighted mean BCE in logit space; zero-weight rows do not contribute.

    Nonfinite candidate outputs on active rows receive infinite loss. Invalid
    observations/weights raise; an empty active subset receives infinite loss.
    """
    y, F, w, offset = (np.asarray(a, dtype=float)
                       for a in (y, learned_boundary, weights, log_nu_b))
    if y.ndim != 1 or F.shape != y.shape or w.shape != y.shape or offset.shape != y.shape:
        raise ValueError("Labels, F, weights and offsets must be aligned one-dimensional arrays.")
    validate_temperature(temperature)
    if not np.isin(y, [0, 1]).all() or not np.isfinite(offset).all():
        raise ValueError("Expected binary reached_birth and finite log_nu_b.")
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("Weights must be finite and nonnegative.")
    active = w > 0
    if not active.any() or not np.isfinite(F[active]).all():
        return float("inf")
    with np.errstate(over="ignore", invalid="ignore"):
        z = output_to_logit(F[active], formulation="boundary", log_nu_b=offset[active],
                            temperature=temperature)
    if not np.isfinite(z).all():
        return float("inf")
    # Equivalent to softplus(z) - y*z, without positive-logit cancellation.
    loss = np.logaddexp(0.0, (1.0 - 2.0 * y[active]) * z)
    return float(np.average(loss, weights=w[active]))


def _bound_loss(y, F, w, *, expected_y, log_nu_b, temperature):
    if not np.array_equal(y, expected_y):
        raise ValueError("Fitness context mismatch: fit a newly selected PreparedSplit.")
    return weighted_boundary_bce(y, F, w, log_nu_b=log_nu_b, temperature=temperature)


@dataclass(frozen=True)
class BoundaryPrototypeConfig:
    population_size: int = 48
    generations: int = 3
    tournament_size: int = 6
    init_depth: tuple[int, int] = (1, 3)
    function_set: tuple[str, ...] = ("add", "sub", "mul", "div", "log")
    const_range: tuple[float, float] = (-2.0, 2.0)
    parsimony_coefficient: float = 0.001
    temperature: float = 1.0
    max_samples: float = 1.0
    n_jobs: int = 1
    seed: int = 42


def fit_boundary_prototype(split: PreparedSplit, cfg=BoundaryPrototypeConfig()):
    """Bind a fresh immutable loss context and fit only the permitted features.

    This returns a regression *engine* whose predict is F, not probability.
    Do not use its R2 score or reuse its dataset-bound metric on another split.
    No warm starts or user-supplied estimator/metric injection are exposed.
    """
    check_backend_contract()
    if split.feature_names not in (("gamma", "k"), ("gamma", "k", "x_b")):
        raise ValueError("Boundary trees require gamma, k and optionally x_b only.")
    if split.log_nu_b is None:
        raise ValueError("Boundary preparation must carry log_nu_b separately.")
    validate_temperature(cfg.temperature)
    if not np.isfinite(cfg.parsimony_coefficient) or cfg.parsimony_coefficient < 0:
        raise ValueError("Use a finite nonnegative fixed parsimony coefficient.")
    X, y, offset = (np.array(a, dtype=float, copy=True)
                    for a in (split.features, split.labels, split.log_nu_b))
    if not np.isfinite(X).all() or (X <= 0).any():
        raise ValueError("Prepared boundary inputs must be finite positive values.")
    if set(np.unique(y)) != {0.0, 1.0} or not np.isfinite(offset).all():
        raise ValueError("Prototype training requires both binary classes and finite offsets.")
    if not 0 < cfg.max_samples <= 1 or int(cfg.max_samples * len(y)) < 1:
        raise ValueError("max_samples must retain at least one training row.")
    for a in (X, y, offset):
        a.setflags(write=False)
    weights = compute_sample_weight("balanced", y)  # training subset only
    # make_fitness probes a synthetic two-row dataset before binding. _Fitness
    # accepts an ordinary picklable partial without fake probe special-casing.
    metric = _Fitness(partial(_bound_loss, expected_y=y, log_nu_b=offset,
                              temperature=cfg.temperature), greater_is_better=False)
    engine = SymbolicRegressor(
        population_size=cfg.population_size, generations=cfg.generations,
        tournament_size=cfg.tournament_size, init_depth=cfg.init_depth,
        function_set=cfg.function_set, const_range=cfg.const_range,
        parsimony_coefficient=cfg.parsimony_coefficient, metric=metric,
        max_samples=cfg.max_samples, n_jobs=cfg.n_jobs, random_state=cfg.seed,
        feature_names=list(split.feature_names), warm_start=False,
    )
    engine.fit(X, y, sample_weight=weights)
    return engine


@dataclass(frozen=True)
class BoundaryExpression:
    """Prepared-input inference artifact with no training rows or loss context.

    Expression is raw gplearn syntax; no algebraic simplification is applied.
    Equality is infeasible, with no tolerance or tuned operating threshold.
    """
    program: _Program
    feature_names: tuple[str, ...]
    temperature: float

    @classmethod
    def from_engine(cls, engine, *, temperature):
        validate_temperature(temperature)
        program = deepcopy(engine._program)
        program.metric = None  # execution needs neither the metric nor labels
        return cls(program, tuple(engine.feature_names), temperature)

    def predict_boundary(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError("Incorrect prepared boundary feature shape.")
        if not np.isfinite(X).all() or (X <= 0).any():
            raise ValueError("Expected finite positive prepared boundary inputs.")
        F = self.program.execute(X)
        if not np.isfinite(F).all():
            raise ValueError("Boundary expression produced nonfinite output.")
        return F

    def predict_margin(self, X, log_nu_b):
        offset = np.asarray(log_nu_b, dtype=float)
        if not np.isfinite(offset).all():
            raise ValueError("Expected finite log_nu_b.")
        return boundary_margin(self.predict_boundary(X), offset)

    def predict_proba(self, X, log_nu_b):
        return expit(self.predict_margin(X, log_nu_b) / self.temperature)

    def predict(self, X, log_nu_b):
        return boundary_is_feasible(self.predict_margin(X, log_nu_b)).astype(int)
