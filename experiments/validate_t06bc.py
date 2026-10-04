"""Direct T06B/T06C checks: unweighted defaults and the revised GP function set.

Run from the repository root: conda run -n debbirth python -m experiments.validate_t06bc
Only small sampled train/val subsets are read. No test data, tuning or final runs.
"""
import ast
from dataclasses import replace
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import sympy as sp
import torch
from gplearn.functions import _Function
from sklearn.utils.class_weight import compute_sample_weight

from src.debbirth.data.prepare import prepare_split
from src.debbirth.data.schema import DatasetSpec
from src.debbirth.models.gp.algorithm import DEBBirthSymbolicClassifier, create_gp_classifier
from src.debbirth.models.gp.config import GPConfig, TrainGPConfig
from src.debbirth.models.gp.constants import EXTENDED_CONSTANT_SET, resolve_constant
from src.debbirth.models.gp.functions import (EXTENDED_FUNCTION_SET, NAMED_FUNCTION_SETS, REVISED_FUNCTION_SET,
                                              primitive_identifier)
from src.debbirth.models.gp.symbolic import program_str_to_sympy, substitute_feature_names
from src.debbirth.models.nn.config import TrainDEBBirthNetConfig
from src.debbirth.models.nn.train import train_net
from src.debbirth.utils.paths import REPO_ROOT
from src.debbirth.utils.results import create_run_outdir

EXP = REPO_ROOT / "experiments"
REVISED = ["add", "sub", "mul", "pdiv", "plog", "min", "cbrt", "square", "cube", "atan"]
EXCLUDED = {"max", "sqrt", "pinv", "neg", "inv", "div", "log", "abs"}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def check_config_resolution():
    """T06C: omitted settings resolve unweighted; legacy weighting remains explicit."""
    spec = DatasetSpec(formulation="normalized")
    cfg = TrainGPConfig(gp=GPConfig(), data_spec=spec)
    assert cfg.class_weights is None and create_gp_classifier(cfg).class_weight is None
    assert TrainDEBBirthNetConfig(data_spec=spec).use_pos_weight is False
    resolved = {}
    for name in ("gp_full_par", "gp_normalized", "gp_boundary", "gp_full_par_balanced"):
        loaded = TrainGPConfig.load_json(EXP / f"{name}.json")
        resolved[name] = {"class_weights": loaded.class_weights,
                          "function_set": [primitive_identifier(p) for p in loaded.gp.function_set],
                          "constants": [c.name for c in loaded.gp.constants],
                          "feature_order": loaded.data_spec.feature_cols}
    for name in ("nn_full_par", "nn_normalized", "nn_boundary"):
        for suffix, expected in (("", False), ("_pos_weight", True)):
            loaded = TrainDEBBirthNetConfig.load_json(EXP / f"{name}{suffix}.json")
            assert loaded.use_pos_weight is expected and loaded.pos_weight is None
            resolved[name + suffix] = {"use_pos_weight": loaded.use_pos_weight}
    assert all(resolved[n]["class_weights"] is None for n in ("gp_full_par", "gp_normalized", "gp_boundary"))
    assert resolved["gp_full_par_balanced"]["class_weights"] == "balanced"
    for name in ("gp_normalized", "gp_boundary"):
        assert resolved[name]["function_set"] == REVISED
    assert resolved["gp_normalized"]["feature_order"] == ["gamma", "k", "nu_b", "x_b"]
    assert resolved["gp_boundary"]["feature_order"] == ["gamma", "k", "x_b"]
    # Explicit legacy settings remain selectable.
    assert TrainGPConfig(gp=GPConfig(), data_spec=spec, class_weights={0: 1.2, 1: 0.8}).class_weights
    # The tuner (T08B, replacing calibrate.py) builds trials from the base JSON and must not
    # hardcode weighting: no call passes class_weights/use_pos_weight and no "balanced" literal.
    tree = ast.parse((REPO_ROOT / "src/debbirth/tuning.py").read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert not any(k.arg in ("class_weights", "use_pos_weight") for c in calls for k in c.keywords)
    assert not any(isinstance(n, ast.Constant) and n.value == "balanced" for n in ast.walk(tree))
    return resolved


def bce(y, p):
    """gplearn's log loss terms (same 1e-15 clipping, applied to p and 1 - p separately)."""
    inv = np.clip(1 - p, 1e-15, 1 - 1e-15)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(inv))


def check_fitness(model, X, y, class_weight):
    """raw/OOB fitness equal the (weighted) mean BCE over in-bag/out-of-bag rows."""
    X_aug = model._augment_X(X)
    base = np.ones(len(y)) if class_weight is None else compute_sample_weight(class_weight, y)
    worst = 0.0
    for program in model._programs[-1]:
        indices, not_indices = program.get_all_indices()
        assert len(np.intersect1d(indices, not_indices)) == 0 and len(indices) + len(not_indices) == len(y)
        terms = bce(y, 1 / (1 + np.exp(-program.execute(X_aug))))
        for rows, value in ((indices, program.raw_fitness_), (not_indices, getattr(program, "oob_fitness_", None))):
            if not len(rows):
                continue
            expected = np.average(terms[rows], weights=base[rows])
            worst = max(worst, abs(expected - value) / max(1.0, abs(value)))
    assert worst < 1e-12, worst
    return worst


def all_programs(model):
    return [p for generation in model._programs for p in (generation or []) if p is not None]


def check_sampling(model):
    """No excluded operator can be sampled; report retained-operator usage."""
    assert [f.name for f in model._function_set] == REVISED
    used = Counter(node.name for p in all_programs(model) for node in p.program if isinstance(node, _Function))
    assert not EXCLUDED & set(used), used
    return dict(used)


def stress_inputs(X, rng):
    """Data rows plus extreme positive inputs and tied columns that activate protection."""
    n = 2000
    gamma = 10 ** rng.uniform(-8, 8, n)
    rows = [X, np.column_stack([gamma, 10 ** rng.uniform(-8, 4, n), 10 ** rng.uniform(-14, 8, n),
                                gamma / (1 + gamma)])]
    tied = X[:500].copy()
    tied[:, 1] = tied[:, 0]
    rows.append(tied)
    return np.vstack(rows)


def check_exports(model, X_eval, n_data_rows):
    """Report (do not assert) agreement of the existing SymPy export with runtime execution.

    The existing export maps pdiv/plog to unprotected a/b and log(a) and cbrt to the
    principal root, so disagreement is expected where protection is active (see T06D).
    """
    constants = {c.name: c.value for c in model.constants}
    names = list(model.feature_names_no_constants)
    X_aug = model._augment_X(X_eval)
    programs = {str(p): p for p in all_programs(model)}
    counts = Counter()
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for text, program in programs.items():
            counts["programs_compared"] += 1
            expr = substitute_feature_names(program_str_to_sympy(text), model.feature_names, names, constants)
            if expr.has(sp.zoo, sp.nan):  # e.g. pdiv(x, c0) read as x/0
                counts["programs_with_undefined_export"] += 1
                continue
            runtime = program.execute(X_aug)
            exported = np.broadcast_to(np.asarray(
                sp.lambdify([sp.Symbol(n) for n in names], expr, "numpy")(*X_eval.T), dtype=complex), runtime.shape)
            agree = np.isclose(exported, runtime, rtol=1e-9, atol=1e-12) | ~np.isfinite(runtime)
            counts["programs_disagreeing_on_data_rows"] += int(not agree[:n_data_rows].all())
            counts["programs_disagreeing_on_any_row"] += int(not agree.all())
    return {"rows_per_program": len(X_eval), "data_rows": n_data_rows, **counts}


def fit(X, y, *, class_weight, max_samples, seed, generations=4, population=300):
    model = DEBBirthSymbolicClassifier(
        base_feature_names=("gamma", "k", "nu_b", "x_b"), constants=EXTENDED_CONSTANT_SET + (resolve_constant("c0"),),
        population_size=population, generations=generations, tournament_size=20, init_depth=(2, 6),
        function_set=REVISED_FUNCTION_SET, parsimony_coefficient=0.0003, class_weight=class_weight,
        max_samples=max_samples, low_memory=False, random_state=seed)
    return model.fit(X, y)


def check_nn(frames, indices):
    """Default NN JSONs train with ordinary mean BCE; no positive weight is created."""
    out = {}
    for formulation in ("normalized", "boundary"):
        cfg = TrainDEBBirthNetConfig.load_json(EXP / f"nn_{formulation}.json")
        cfg = replace(cfg, epochs=2, device="cpu")
        prepared = {key: prepare_split(frame, cfg.data_spec, source_name=f"data/processed/{key}.csv")
                    .select(indices[key]) for key, frame in frames.items()}
        result = train_net(cfg, save=False, prepared=prepared)
        assert result["train_config"].use_pos_weight is False and result["train_config"].pos_weight is None
        val = prepared["val"]
        logits = result["predictor"].predict_prepared(val)["logit"].astype(float)
        expected = np.logaddexp(0, (1 - 2 * val.labels) * logits).mean()
        np.testing.assert_allclose(result["val_metrics"].val_loss, expected, rtol=2e-6, atol=2e-6)
        out[formulation] = {"val_loss": result["val_metrics"].val_loss, "unweighted_mean_bce": float(expected)}
    return out


def main():
    torch.set_num_threads(1)
    started = time.time()
    source_paths = [REPO_ROOT / f"data/processed/{name}.csv" for name in ("train", "val")]
    preserved = [p for p in (REPO_ROOT / "results/models").rglob("*") if p.is_file()] + source_paths
    hashes_before = {str(p.relative_to(REPO_ROOT)): digest(p) for p in preserved}
    rng = np.random.default_rng(42)
    frames = {p.stem: pd.read_csv(p, float_precision="round_trip") for p in source_paths}
    indices = {"train": rng.choice(len(frames["train"]), 2000, replace=False),
               "val": rng.choice(len(frames["val"]), 128, replace=False)}
    summary = {"script_sha256": digest(Path(__file__)), "sampling_seed": 42, "test_set_used": False,
               "source_hashes": {str(p.relative_to(REPO_ROOT)): digest(p) for p in source_paths},
               "rows": {"train": 2000, "val": 128}}

    summary["resolved_configs"] = check_config_resolution()
    assert [primitive_identifier(p) for p in NAMED_FUNCTION_SETS["revised"]] == REVISED
    summary["extended_minus_revised"] = sorted({primitive_identifier(p) for p in EXTENDED_FUNCTION_SET}
                                               - set(REVISED))

    split = prepare_split(frames["train"], DatasetSpec(formulation="normalized", include_x_b=True),
                          source_name="data/processed/train.csv").select(indices["train"])
    X, y = split.features, split.labels
    fitness = {}
    for label, class_weight, max_samples in (("unweighted_subsampled", None, 0.7),
                                             ("unweighted_full", None, 1.0),
                                             ("balanced_subsampled", "balanced", 0.7)):
        model = fit(X, y, class_weight=class_weight, max_samples=max_samples, seed=7)
        fitness[label] = {"max_scaled_fitness_error": check_fitness(model, X, y, class_weight)}
        if label == "unweighted_subsampled":
            summary["operator_usage"] = check_sampling(model)
            summary["existing_export_agreement"] = check_exports(model, stress_inputs(X, rng), len(X))
    summary["fitness"] = fitness
    summary["nn_default_loss"] = check_nn(frames, indices)

    hashes_after = {str(p.relative_to(REPO_ROOT)): digest(p) for p in preserved}
    assert hashes_after == hashes_before
    summary["preserved_files_unchanged"] = len(hashes_before)
    summary["seconds"] = round(time.time() - started, 1)
    outdir = Path(create_run_outdir("t06bc_validation"))
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print("Run directory:", outdir)


if __name__ == "__main__":
    main()
