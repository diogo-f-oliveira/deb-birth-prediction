"""Small T06 engineering runs for normalized and boundary GP; not tuning or a benchmark.

Run from the repository root: conda run -n debbirth python -m experiments.validate_gp_formulations
Only sampled train/val subsets are read; no test data. Outputs go to a distinct ignored run.
"""
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from gplearn.functions import _Function
from sklearn.utils.class_weight import compute_sample_weight

from src.debbirth.data.prepare import prepare_split
from src.debbirth.evaluate.predict import metrics_from_predictions
from src.debbirth.models.gp.boundary import GPBoundaryModel, fit_gp_boundary
from src.debbirth.models.gp.config import TrainGPConfig
from src.debbirth.models.gp.functions import primitive_identifier
from src.debbirth.models.gp.train import load_gp_run, train_gp_classifier
from src.debbirth.utils.paths import REPO_ROOT
from src.debbirth.utils.results import create_run_outdir

EXP = REPO_ROOT / "experiments"
SMALL = dict(population_size=200, generations=8, tournament_size=20)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def small_config(name, outdir, **overrides):
    cfg = TrainGPConfig.load_json(EXP / f"{name}.json")
    gp = replace(cfg.gp, **SMALL)
    return replace(cfg, gp=gp, num_workers=1, verbose=0, outdir=outdir, **overrides)


def stable_bce(y, z):
    return y * np.logaddexp(0, -z) + (1 - y) * np.logaddexp(0, z)


def audit_boundary(engine, model, split, cfg, weights):
    """Independent loss recomputation for every retained program; final-program policy."""
    names = set(cfg.data_spec.feature_cols) | {c.name for c in cfg.gp.constants}
    n_cols = len(names)
    X_aug = np.concatenate([split.features, np.broadcast_to([c.value for c in cfg.gp.constants],
                                                            (len(split.labels), len(cfg.gp.constants)))], axis=1)
    checked = 0
    with np.errstate(all="ignore"):
        for population in engine._programs:
            for program in population or []:
                if program is None:
                    continue
                assert all(0 <= node < n_cols for node in program.program if isinstance(node, int))
                F = program.execute(X_aug)
                if np.isfinite(F).all():
                    z = (F - split.log_nu_b) / cfg.boundary_temperature
                    expected = np.average(stable_bce(split.labels, z), weights=weights)
                    np.testing.assert_allclose(program.raw_fitness_, expected, rtol=1e-12, atol=1e-12)
                else:
                    assert program.raw_fitness_ == np.inf
                np.testing.assert_allclose(program.fitness_, program.raw_fitness_
                                           + cfg.gp.parsimony_coefficient * program.length_)
                checked += 1
    last = engine._programs[-1]
    assert engine._program is last[int(np.argmin([p.raw_fitness_ for p in last]))]
    assert np.isfinite(engine._program.raw_fitness_)
    used = {node.name for node in model.program.program if isinstance(node, _Function)}
    allowed = {primitive_identifier(p) for p in cfg.gp.function_set}
    assert used <= allowed, used - allowed
    terminals = {model.program.feature_names[node] for node in model.program.program if isinstance(node, int)}
    assert terminals <= names and not terminals & {"nu_b", "log_nu_b", "v_Hb", "f", "g"}
    return checked


def boundary_semantics(model, split):
    F = model.predict_boundary(split.features)
    # Row alignment: a whole-record shuffle permutes every output identically.
    perm = np.random.default_rng(3).permutation(len(split.labels))
    shuffled = split.select(perm)
    np.testing.assert_array_equal(model.predict_boundary(shuffled.features), F[perm])
    np.testing.assert_array_equal(model.predict_proba(shuffled.features, shuffled.log_nu_b),
                                  model.predict_proba(split.features, split.log_nu_b)[perm])
    # Maturity monotonicity at fixed (gamma, k[, x_b]); strict ties; temperature leaves decisions unchanged.
    offsets = np.linspace(-6, 6, 25)
    probs = np.stack([model.predict_proba(split.features[:50], F[:50] + d) for d in offsets])
    assert (np.diff(probs, axis=0) <= 0).all() and (np.diff(probs, axis=0) < 0).any()
    assert (model.predict(split.features, F) == 0).all()
    assert (model.predict(split.features, F - 1e-9) == 1).all()
    hot = replace(model, temperature=model.temperature * 5)
    np.testing.assert_array_equal(hot.predict(split.features, split.log_nu_b),
                                  model.predict(split.features, split.log_nu_b))
    return {"monotone_offsets": len(offsets), "tie_rows": len(F)}


def predictions(model, split):
    if isinstance(model, GPBoundaryModel):
        return {"F": model.predict_boundary(split.features),
                "probability": model.predict_proba(split.features, split.log_nu_b),
                "prediction": model.predict(split.features, split.log_nu_b)}
    return {"probability": model.predict_proba(split.features)[:, 1], "prediction": model.predict(split.features)}


def verify_run(result):
    val = result["prepared"]["val"]
    outdir = result["outdir"]
    details = predictions(result["model"], val)
    expected = metrics_from_predictions(val.labels, details["probability"], y_pred=details["prediction"])
    saved = json.loads((outdir / "metrics/val_metrics.json").read_text())
    for key, value in asdict(expected).items():
        np.testing.assert_allclose(saved[key], value, rtol=1e-12, atol=1e-12)
    loaded = load_gp_run(outdir)
    assert loaded["train_cfg"] == result["train_config"]
    for key, value in predictions(loaded["model"], val).items():
        np.testing.assert_array_equal(value, details[key])
    expression = (outdir / "model/expression.txt").read_text(encoding="utf-8")
    assert result["best_program"] in expression and (outdir / "history.csv").exists()
    np.savez(outdir / "reload_probe.npz", features=val.features,
             offset=val.log_nu_b if val.log_nu_b is not None else np.zeros(0), **details)
    return {"path": str(outdir.relative_to(REPO_ROOT)), "val_metrics": saved,
            "expression": expression}


def reload_check(directory):
    for path in directory.glob("*/reload_probe.npz"):
        stored = np.load(path)
        model = load_gp_run(path.parent)["model"]
        if isinstance(model, GPBoundaryModel):
            got = {"F": model.predict_boundary(stored["features"]),
                   "probability": model.predict_proba(stored["features"], stored["offset"]),
                   "prediction": model.predict(stored["features"], stored["offset"])}
        else:
            got = {"probability": model.predict_proba(stored["features"])[:, 1],
                   "prediction": model.predict(stored["features"])}
        for key, value in got.items():
            np.testing.assert_array_equal(value, stored[key])
    print("Fresh-process GP reload checks passed.")


def main():
    started = time.time()
    source_paths = [REPO_ROOT / f"data/processed/{name}.csv" for name in ("train", "val")]
    preserved = [p for p in (REPO_ROOT / "results/models").rglob("*") if p.is_file()] + source_paths
    hashes_before = {str(p.relative_to(REPO_ROOT)): digest(p) for p in preserved}
    rng = np.random.default_rng(42)
    frames = {p.stem: pd.read_csv(p, float_precision="round_trip") for p in source_paths}
    indices = {"train": rng.choice(len(frames["train"]), 2000, replace=False),
               "val": rng.choice(len(frames["val"]), 500, replace=False)}
    run_dir = Path(create_run_outdir("t06_gp_validation"))
    metadata = {"source_hashes": {str(p.relative_to(REPO_ROOT)): digest(p) for p in source_paths},
                "rows": {"train": 2000, "val": 500}, "sampling_seed": 42, "test_set_used": False,
                "target_policy": "Recorded reached_birth, including six-second get_lb2 failures/timeouts",
                "script_sha256": digest(Path(__file__))}
    summary = {**metadata, "runs": {}}

    def prepared_for(spec):
        return {key: prepare_split(frame, spec, source_name=f"data/processed/{key}.csv").select(indices[key])
                for key, frame in frames.items()}

    for name, config, overrides in (("normalized_xb", "gp_normalized", {}),
                                    ("boundary_xb", "gp_boundary", {}),
                                    ("boundary_no_xb_T0.7", "gp_boundary", {"boundary_temperature": 0.7})):
        cfg = small_config(config, run_dir / name, **overrides)
        if name == "boundary_no_xb_T0.7":
            cfg = replace(cfg, data_spec=replace(cfg.data_spec, include_x_b=False))
        prepared = prepared_for(cfg.data_spec)
        result = train_gp_classifier(cfg, save_run=True, prepared=prepared, data_metadata=metadata)
        for key, split in prepared.items():
            split.source_rows.to_csv(result["outdir"] / f"{key}_rows.csv", index=False)
        entry = verify_run(result)
        if cfg.data_spec.formulation == "boundary":
            entry["programs_audited"] = audit_boundary(result["engine"], result["model"], prepared["train"], cfg,
                                                       np.ones(len(prepared["train"].labels)))
            entry["semantics"] = boundary_semantics(result["model"], prepared["val"])
            # The bound metric refuses another split's labels.
            metric = result["engine"]._metric.function
            other = prepared["train"].labels[::-1].astype(float)
            try:
                metric(other, np.zeros_like(other), np.ones_like(other))
                raise AssertionError("context mismatch not detected")
            except ValueError:
                pass
        else:
            assert result["model"].class_weight is None
        summary["runs"][name] = entry

    # Boundary-only checks without saving: process parallelism and explicit legacy weighting.
    cfg = small_config("gp_boundary", None, low_memory=False)
    train = prepared_for(cfg.data_spec)["train"]
    serial, _ = fit_gp_boundary(train, replace(cfg, gp=replace(cfg.gp, generations=3)))
    parallel, _ = fit_gp_boundary(train, replace(cfg, gp=replace(cfg.gp, generations=3), num_workers=2))
    assert serial.expression == parallel.expression
    np.testing.assert_array_equal(serial.predict_boundary(train.features), parallel.predict_boundary(train.features))
    balanced_cfg = replace(cfg, gp=replace(cfg.gp, generations=3), class_weights="balanced")
    model, engine = fit_gp_boundary(train, balanced_cfg)
    summary["balanced_programs_audited"] = audit_boundary(engine, model, train, balanced_cfg,
                                                          compute_sample_weight("balanced", train.labels))
    summary["serial_parallel_identical"] = True

    assert {str(p.relative_to(REPO_ROOT)): digest(p) for p in preserved} == hashes_before
    summary["preserved_files_unchanged"] = len(hashes_before)
    summary["seconds"] = round(time.time() - started, 1)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, "-m", "experiments.validate_gp_formulations", "--reload", str(run_dir)],
                   cwd=REPO_ROOT, check=True)
    for name, entry in summary["runs"].items():
        print(f"== {name}: {entry['path']}\n{json.dumps(entry['val_metrics'])}\n{entry['expression']}")
    print("Run directory:", run_dir)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--reload":
        reload_check(Path(sys.argv[2]))
    else:
        main()
