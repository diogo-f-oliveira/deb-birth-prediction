"""Reproduce the T04 backend experiment; no test split or production training.

Run from the repository root: conda run -n debbirth python -m experiments.prototype_gp_boundary
"""
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.utils.class_weight import compute_sample_weight

from src.debbirth.data.prepare import prepare_split
from src.debbirth.data.schema import DatasetSpec
from src.debbirth.formulations import output_to_logit
from src.debbirth.models.gp.algorithm import create_gp_classifier
from src.debbirth.models.gp.boundary_prototype import (
    BoundaryExpression, BoundaryPrototypeConfig, check_backend_contract,
    fit_boundary_prototype, weighted_boundary_bce,
)
from src.debbirth.models.gp.config import GPConfig, TrainGPConfig
from src.debbirth.utils.paths import REPO_ROOT
from src.debbirth.utils.results import create_run_outdir, save_run_metadata


def audit_population(engine, split, cfg):
    """Recompute losses by physically indexing rows, independently of masks."""
    weights = compute_sample_weight("balanced", split.labels)
    checked = 0
    for population in engine._programs:
        if population is None:
            continue
        for program in population:
            if program is None:
                continue
            assert program.n_features == len(split.feature_names)
            assert all(0 <= node < len(split.feature_names)
                       for node in program.program if isinstance(node, int))
            F = program.execute(split.features)
            inside, outside = program.get_all_indices()
            for indices, recorded in [(inside, program.raw_fitness_)] + (
                    [(outside, program.oob_fitness_)] if cfg.max_samples < 1 else []):
                z = (F[indices] - split.log_nu_b[indices]) / cfg.temperature
                # Independent, stable two-class expression.
                individual_loss = (split.labels[indices] * np.logaddexp(0, -z)
                                   + (1 - split.labels[indices]) * np.logaddexp(0, z))
                expected = np.average(individual_loss, weights=weights[indices])
                np.testing.assert_allclose(recorded, expected, rtol=1e-12, atol=1e-12)
                assert np.isfinite(recorded)
            np.testing.assert_allclose(
                program.fitness_, program.raw_fitness_ + cfg.parsimony_coefficient * program.length_)
            checked += 1
    last = engine._programs[-1]
    assert engine._program is last[np.argmin([p.raw_fitness_ for p in last])]
    return checked


def reload_check(outdir):
    data = np.load(outdir / "inference_inputs.npz")
    artifact = joblib.load(outdir / "boundary_expression.joblib")
    engine = joblib.load(outdir / "boundary_engine.joblib")
    normalized = joblib.load(outdir / "normalized.joblib")
    np.testing.assert_array_equal(artifact.predict_boundary(data["X"]), data["F"])
    np.testing.assert_array_equal(engine.predict(data["X"]), data["F"])
    np.testing.assert_array_equal(artifact.predict_proba(data["X"], data["offset"]), data["p"])
    np.testing.assert_array_equal(normalized.predict_proba(data["X_normalized"]), data["p_normalized"])
    assert artifact.program.metric is None
    print("Fresh-process reload: boundary engine/expression and normalized predictions agree.")


def main():
    cfg = BoundaryPrototypeConfig(temperature=0.7)
    backend_hashes = check_backend_contract()
    # Read only supplied train/validation CSVs. Selection is performed on whole
    # PreparedSplit records and is matched across the two formulations.
    rng = np.random.default_rng(cfg.seed)
    splits, normalized, source_hashes = {}, {}, {}
    for name, count in (("train", 512), ("val", 128)):
        path = REPO_ROOT / "data/processed" / f"{name}.csv"
        source = path.relative_to(REPO_ROOT).as_posix()
        source_hashes[source] = sha256(path.read_bytes()).hexdigest()
        frame = pd.read_csv(path, float_precision="round_trip")
        selection = rng.choice(len(frame), count, replace=False)
        splits[name] = prepare_split(frame, DatasetSpec(formulation="boundary"),
                                     source_name=source).select(selection)
        normalized[name] = prepare_split(frame, DatasetSpec(formulation="normalized"),
                                         source_name=source).select(selection)
        pd.testing.assert_frame_equal(splits[name].source_rows, normalized[name].source_rows)

    outdir = Path(create_run_outdir("t04_gp_backend"))
    save_run_metadata(outdir, cfg, {
        "source_hashes": source_hashes, "backend_source_hashes": backend_hashes,
        "target": "reached_birth", "label_policy": "recorded labels, including solver failures",
        "formulation": "boundary", "feature_order": ["gamma", "k"],
        "preprocessing": "shared normalization, no learned scaling",
        "loss": "sum(w * BCE) / sum(w)", "class_weights": "balanced on selected training rows",
        "decision": "F - log_nu_b > 0; equality infeasible; no tolerance",
        "experiment_script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    for name, split in splits.items():
        split.source_rows.to_csv(outdir / f"{name}_rows.csv", index=False)

    cases, engines = [], {}
    for fraction in (1.0, 0.65):
        for workers in (1, 2):
            run_cfg = replace(cfg, max_samples=fraction, n_jobs=workers)
            start = perf_counter()
            engine = fit_boundary_prototype(splits["train"], run_cfg)
            duration = perf_counter() - start
            checked = audit_population(engine, splits["train"], run_cfg)
            engines[fraction, workers] = engine
            artifact = BoundaryExpression.from_engine(engine, temperature=cfg.temperature)
            val = splits["val"]
            p = artifact.predict_proba(val.features, val.log_nu_b)
            assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            cases.append({"config": asdict(run_cfg), "seconds": duration,
                          "audited_programs": checked, "expression": str(engine._program),
                          "training_raw_fitness": float(engine._program.raw_fitness_),
                          "val_macro_f1": f1_score(val.labels, artifact.predict(val.features, val.log_nu_b), average="macro"),
                          "val_mcc": matthews_corrcoef(val.labels, artifact.predict(val.features, val.log_nu_b))})
            print(f"max_samples={fraction}, n_jobs={workers}: {checked} retained programs audited", flush=True)
        left, right = engines[fraction, 1], engines[fraction, 2]
        assert str(left._program) == str(right._program)
        np.testing.assert_array_equal(left.predict(val.features), right.predict(val.features))
        for a, b in zip(left._programs[-1], right._programs[-1]):
            assert str(a) == str(b)
            np.testing.assert_equal(a.raw_fitness_, b.raw_fitness_)

    # Shuffle the whole record, then bind a new context. Distinct offsets and
    # repeated boundary coordinates remain associated with their own labels.
    shuffled = splits["train"].select(rng.permutation(len(splits["train"].labels)))
    shuffle_cfg = replace(cfg, max_samples=0.65)
    shuffled_engine = fit_boundary_prototype(shuffled, shuffle_cfg)
    shuffled_count = audit_population(shuffled_engine, shuffled, shuffle_cfg)
    mismatch_rejected = False
    try:
        engines[1.0, 1]._metric(shuffled.labels, np.zeros(len(shuffled.labels)), np.ones(len(shuffled.labels)))
    except ValueError:
        mismatch_rejected = True
    assert mismatch_rejected

    # Direct numerical/structural checks; these do not label or solve DEB data.
    y = np.array([0., 1., 0., 1.])
    F = np.array([1000., -1000., -1000., 1000.])
    assert weighted_boundary_bce(y, F, np.ones(4), log_nu_b=np.zeros(4), temperature=1.) == 500.
    assert weighted_boundary_bce(y, np.full(4, np.nan), np.zeros(4), log_nu_b=np.zeros(4), temperature=1.) == np.inf
    masked_F = np.array([np.nan, 0., 0., 0.])
    np.testing.assert_allclose(weighted_boundary_bce(y, masked_F, [0, 1, 1, 1], log_nu_b=np.zeros(4), temperature=1.), np.log(2.))
    engine = engines[1.0, 1]
    artifact = BoundaryExpression.from_engine(engine, temperature=cfg.temperature)
    X = np.repeat(splits["val"].features[:1], 9, axis=0)
    F = artifact.predict_boundary(X)
    offsets = F + np.linspace(-4, 4, len(X))
    assert (np.diff(artifact.predict_proba(X, offsets)) < 0).all()
    assert artifact.predict(X, F).sum() == 0
    assert np.all(artifact.predict_proba(X, F) == 0.5)
    for temperature in (0.2, 1., 3.):
        other = replace(artifact, temperature=temperature)
        np.testing.assert_array_equal(other.predict(X, offsets), artifact.predict(X, offsets))
    zero = BoundaryExpression.from_engine(engine, temperature=cfg.temperature)
    zero.program.program = [0.0]
    k1 = np.array([[0.1, 1.], [1., 1.], [10., 1.]])
    np.testing.assert_array_equal(zero.predict(k1, np.log([0.5, 1., 2.])), [1, 0, 0])
    synthetic = pd.DataFrame({"g": [1., 2., 4., 1.], "k": [3.] * 4,
                              "v_Hb": [0.1, 0.8, 6.4, 0.2], "f": [1., 2., 4., 1.],
                              "reached_birth": [1, 1, 1, 0]})
    equivalent = prepare_split(synthetic, DatasetSpec(formulation="boundary"))
    np.testing.assert_allclose(artifact.predict_proba(equivalent.features, equivalent.log_nu_b)[:3],
                               np.repeat(artifact.predict_proba(equivalent.features[:1], equivalent.log_nu_b[:1]), 3))
    assert artifact.predict_proba(equivalent.features, equivalent.log_nu_b)[3] < artifact.predict_proba(equivalent.features, equivalent.log_nu_b)[0]
    alignment_probe = prepare_split(pd.concat([synthetic] * 12, ignore_index=True), DatasetSpec(formulation="boundary"))
    repeated_engine = fit_boundary_prototype(alignment_probe, shuffle_cfg)
    repeated_count = audit_population(repeated_engine, alignment_probe, shuffle_cfg)

    normal_cfg = TrainGPConfig(
        gp=GPConfig(population_size=48, generations=3, tournament_size=6, init_depth=(1, 3)),
        data_spec=DatasetSpec(formulation="normalized"), num_workers=1, seed=cfg.seed)
    normal_model = create_gp_classifier(normal_cfg)
    normal_model.fit(normalized["train"].features, normalized["train"].labels)
    normal_p = normal_model.predict_proba(normalized["val"].features)
    assert np.isfinite(normal_p).all()
    normal_cfg.save_json(outdir / "normalized_config.json")
    joblib.dump(normal_model, outdir / "normalized.joblib")
    joblib.dump(engine, outdir / "boundary_engine.joblib")
    joblib.dump(artifact, outdir / "boundary_expression.joblib")
    np.savez(outdir / "inference_inputs.npz", X=val.features, offset=val.log_nu_b,
             F=artifact.predict_boundary(val.features), p=artifact.predict_proba(val.features, val.log_nu_b),
             X_normalized=normalized["val"].features, p_normalized=normal_p)
    subprocess.run([sys.executable, "-m", "experiments.prototype_gp_boundary", "--reload-only", str(outdir)],
                   cwd=REPO_ROOT, check=True)
    summary = {"cases": cases, "shuffled_audited_programs": shuffled_count,
               "repeated_coordinates_audited_programs": repeated_count,
               "normalized_expression": str(normal_model._program),
               "checks": ["all retained raw/OOB/penalized fitness recomputed by explicit row indexing",
                          "serial/process-parallel final populations agree", "whole-record shuffle",
                          "repeated coordinates with different maturity/labels", "context mismatch rejection",
                          "stable extreme BCE and zero-weight handling", "maturity monotonicity",
                          "strict ties and temperature-invariant decisions", "k=1 semantics for fixed F=0",
                          "original-input scaling equivalence", "fresh-process engine/expression/normalized reload"],
               "limitations": "Tiny engineering probe, no predictive/backend benchmark; no test data, temperature tuning, or learned k=1 constraint."}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (outdir / "boundary_expression.txt").write_text(str(engine._program) + "\n", encoding="utf-8")
    print(f"T04 complete: {outdir}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--reload-only":
        reload_check(Path(sys.argv[2]))
    else:
        main()
