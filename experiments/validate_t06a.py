"""T06A checks: shared original-parameter inference, evaluation and coordinate-aware plots.

Run from the repository root: conda run -n debbirth python -m experiments.validate_t06a
Uses existing saved models (archived, T05 NN and T06 GP validation runs) and a sampled
validation subset; no test data and no training beyond tiny consolidation probes.
"""
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.debbirth.data.prepare import prepare_split
from src.debbirth.data.schema import FULL_PAR_COLS
from src.debbirth.evaluate.predict import evaluate_binary_classifier, metrics_from_predictions
from src.debbirth.models.gp.boundary import GPBoundaryModel
from src.debbirth.models.gp.config import TrainGPConfig
from src.debbirth.models.gp.train import load_gp_run, train_gp_classifier
from src.debbirth.models.nn.train import load_trained_nn
from src.debbirth.plot.boundary import (draw_critical_curve, plot_critical_surface, plot_decision_mesh,
                                        plot_decision_mesh_with_models, slice_grid, update_legend)
from src.debbirth.utils.paths import REPO_ROOT
from src.debbirth.utils.results import create_run_outdir

T05 = REPO_ROOT / "results/runs/2026-09-08T16-27-53-838745_t05_nn_validation"
T06 = REPO_ROOT / "results/runs/2026-10-02T16-55-47-145588_t06_gp_validation"
MODELS = {
    "gp_archived_full_par": ("gp", REPO_ROOT / "results/models/DEBBirthGP"),
    "nn_archived_full_par": ("nn", REPO_ROOT / "results/models/DEBBirthNet"),
    "nn_normalized": ("nn", T05 / "normalized_log_standardize"),
    "nn_boundary": ("nn", T05 / "boundary_log_standardize"),
    "nn_boundary_xb": ("nn", T05 / "boundary_standardize_xb"),
    "gp_normalized_xb": ("gp", T06 / "normalized_xb"),
    "gp_boundary_xb": ("gp", T06 / "boundary_xb"),
    "gp_boundary_no_xb": ("gp", T06 / "boundary_no_xb_T0.7"),
}


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def load_predictors():
    import warnings
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # archived GP: known historical-config warning
        for name, (family, path) in MODELS.items():
            out[name] = (load_gp_run(path) if family == "gp" else load_trained_nn(path, "cpu"))["predictor"]
    return out


def check_predictor(name, predictor, physical):
    spec = predictor.spec
    details = predictor.predict_details(physical)
    n = len(physical)
    for key, value in details.items():
        assert value.shape == (n,), (name, key)
    p = details["probability"]
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all() and np.isin(details["prediction"], [0, 1]).all()
    # Explicit prepared path gives identical outputs; nothing is normalized twice.
    split = prepare_split(physical.assign(reached_birth=0), spec)
    for key, value in predictor.predict_prepared(split).items():
        np.testing.assert_array_equal(value, details[key])
    np.testing.assert_array_equal(predictor.predict_proba(physical), p)
    np.testing.assert_array_equal(predictor.predict(physical), details["prediction"])
    np.testing.assert_array_equal(predictor.predict(physical, probability_threshold=0.3), (p >= 0.3).astype(int))
    result = {"formulation": spec.formulation, "features": list(spec.feature_cols),
              "positive_rate": float(details["prediction"].mean())}
    if spec.formulation == "boundary":
        margin = details["margin"]
        np.testing.assert_array_equal(details["prediction"], (margin > 0).astype(int))
        T = getattr(predictor, "temperature", None) or predictor.config.boundary_temperature
        np.testing.assert_allclose(details["logit"], margin / T, rtol=1e-6)
        log_psi = predictor.critical_maturity(physical, log=True)
        np.testing.assert_array_equal(log_psi, details["learned_output"].astype(float))
        log_crit = predictor.critical_maturity(physical, normalized=False, log=True)
        f = physical["f"].to_numpy()
        np.testing.assert_allclose(log_crit, log_psi + 3 * np.log(f), rtol=1e-12)
        log_v = np.log(physical["v_Hb"].to_numpy())
        clear = np.abs(log_crit - log_v) > 1e-5  # away from float32 ties
        assert ((log_v < log_crit) == (details["prediction"] == 1))[clear].all()
        np.testing.assert_allclose(predictor.critical_maturity(physical, normalized=False), np.exp(log_crit), rtol=1e-12)
        result["temperature"] = T
    else:
        try:
            predictor.critical_maturity(physical)
            raise AssertionError("critical maturity must require a boundary model")
        except ValueError:
            pass
    if hasattr(predictor, "model") and hasattr(predictor.model, "_program"):  # GP classifier vs gplearn API
        X = split.features
        np.testing.assert_array_equal(predictor.model.predict_proba(X)[:, 1], p)
        np.testing.assert_array_equal(predictor.model.predict(X).astype(int), details["prediction"])
    if isinstance(getattr(predictor, "model", None), GPBoundaryModel):
        np.testing.assert_array_equal(predictor.model.predict_proba(split.features, split.log_nu_b), p)
    if spec.formulation != "full_par":
        # Scaling-equivalent inputs (c*g, k, c^3*v_Hb, c*f) give the same normalized model inputs.
        worst, flips = 0.0, 0
        for c in (0.5, 2.0):
            scaled = physical.assign(g=physical.g * c, v_Hb=physical.v_Hb * c ** 3, f=physical.f * c)
            other = predictor.predict_details(scaled)
            worst = max(worst, float(np.max(np.abs(other["probability"] - p))))
            clear = np.abs(details["logit"].astype(float)) > 1e-4  # float32 NN ties may round either way
            flips += int((other["prediction"] != details["prediction"])[clear].sum())
        assert worst < 1e-5 and flips == 0, (name, worst, flips)
        result["scaling_max_probability_change"] = worst
    return details, result


def nn_refactor_probe(predictors):
    """T05 saved predictions from physical inputs; the moved base methods must reproduce them exactly."""
    checked = 0
    for name, sub in (("nn_normalized", "normalized_log_standardize"), ("nn_boundary", "boundary_log_standardize"),
                      ("nn_boundary_xb", "boundary_standardize_xb")):
        stored = np.load(T05 / sub / "reload_probe.npz")
        details = predictors[name].predict_details(stored["physical"])
        for key, value in details.items():
            np.testing.assert_array_equal(value, stored[key])
        checked += 1
    return checked


def check_evaluation(predictors, val_frame):
    """Shared one-pass evaluation reproduces T06's saved validation metrics on the same rows."""
    rows = pd.read_csv(T06 / "boundary_xb/val_rows.csv")
    assert rows["__source_row"].tolist() == val_frame["__source_row"].tolist()
    out = {}
    for name, sub in (("gp_normalized_xb", "normalized_xb"), ("gp_boundary_xb", "boundary_xb"),
                      ("gp_boundary_no_xb", "boundary_no_xb_T0.7"), ("nn_boundary", None)):
        predictor = predictors[name]
        split = prepare_split(val_frame, predictor.spec)
        metrics = predictor.evaluate_prepared(split)
        details = predictor.predict_prepared(split)
        assert asdict(metrics) == asdict(metrics_from_predictions(split.labels, details["probability"],
                                                                  y_pred=details["prediction"]))
        thresholded = predictor.evaluate_prepared(split, probability_threshold=0.3)
        assert thresholded.tp + thresholded.fn == metrics.tp + metrics.fn
        if sub:
            saved = json.loads((T06 / sub / "metrics/val_metrics.json").read_text())
            for key, value in asdict(metrics).items():
                np.testing.assert_allclose(saved[key], value, rtol=1e-12, atol=1e-12)
        out[name] = {"f1_macro": metrics.f1_macro, "mcc": metrics.mcc}
    return out


def check_training_consolidation(frames, indices):
    """Trainer validation (one predictor pass) matches the historical two-call evaluator."""
    out = {}
    for config in ("gp_normalized", "gp_boundary"):
        cfg = TrainGPConfig.load_json(REPO_ROOT / f"experiments/{config}.json")
        cfg = replace(cfg, gp=replace(cfg.gp, population_size=32, generations=2, tournament_size=4),
                      num_workers=1, verbose=0)
        prepared = {key: prepare_split(frames[key], cfg.data_spec).select(indices[key]) for key in ("train", "val")}
        result = train_gp_classifier(cfg, save_run=False, prepared=prepared)
        if config == "gp_normalized":
            old = evaluate_binary_classifier(result["model"], prepared["val"].features, prepared["val"].labels)
            assert asdict(old) == asdict(result["val_metrics"])
        out[config] = result["val_metrics"].f1_macro
    return out


def make_plots(predictors, outdir):
    paths = {}
    g = np.logspace(-3, 2, 160)
    # (a) Original coordinates at one (k, f) slice: mesh = GP boundary decision.
    df = slice_grid("g", g, "v_Hb", np.logspace(-6, 1, 160), k=0.3, f=0.8)
    for name in ("gp_boundary_xb", "gp_archived_full_par", "nn_archived_full_par", "nn_boundary"):
        df[name] = predictors[name].predict(df[list(FULL_PAR_COLS)])
    fig, ax = plot_decision_mesh_with_models(df, ["gp_archived_full_par", "nn_archived_full_par", "nn_boundary"],
                                             decision_col="gp_boundary_xb",
                                             model_labels=["archived GP", "archived NN", "NN boundary"])
    curve = slice_grid("g", g, "v_Hb", [1.0], k=0.3, f=0.8)
    draw_critical_curve(ax, g, predictors["gp_boundary_xb"].critical_maturity(curve, normalized=False),
                        label=r"GP boundary $f^3\Psi$", color="tab:purple", linestyle=":")
    update_legend(ax, loc="lower left", fontsize=7)
    paths["original_k0.3_f0.8"] = fig
    # (b) Normalized coordinates at k = 0.3 and k = 3: mesh = NN normalized decision.
    gamma = np.logspace(-3, 2, 160)
    for k in (0.3, 3.0):
        df = slice_grid("gamma", gamma, "nu_b", np.logspace(-6, 2, 160), k=k)
        for name in ("nn_normalized", "gp_normalized_xb"):
            df[name] = predictors[name].predict(df[list(FULL_PAR_COLS)])
        fig, ax = plot_decision_mesh_with_models(df, ["gp_normalized_xb"], decision_col="nn_normalized",
                                                 x_col="gamma", y_col="nu_b", model_labels=["GP normalized"])
        curve = slice_grid("gamma", gamma, "nu_b", [1.0], k=k)
        for name, color in (("nn_boundary", "tab:red"), ("gp_boundary_xb", "tab:purple")):
            draw_critical_curve(ax, gamma, predictors[name].critical_maturity(curve), color=color, linestyle=":",
                                label=f"{name} $\\Psi$")
        update_legend(ax, loc="lower left", fontsize=7)
        paths[f"normalized_k{k}"] = fig
    # (c) Critical surfaces F(gamma, k).
    for name in ("nn_boundary", "gp_boundary_xb"):
        df = slice_grid("gamma", gamma, "k", np.logspace(-2, 1.5, 120), nu_b=1.0)
        df["log_psi"] = predictors[name].critical_maturity(df[list(FULL_PAR_COLS)], log=True)
        fig, _ = plot_critical_surface(df, title=f"{name}: learned $F(\\gamma, k)$")
        paths[f"surface_{name}"] = fig
    saved = {}
    for name, fig in paths.items():
        path = outdir / f"{name}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved[name] = str(path.relative_to(REPO_ROOT))
    # Slice assumptions are checked, not read from the first row.
    mixed = slice_grid("g", g[:5], "v_Hb", [1e-3, 1e-2], k=0.3, f=0.8)
    mixed.loc[0, "f"] = 0.5
    mixed["z"] = 1
    try:
        plot_decision_mesh(mixed, "z")
        raise AssertionError("mixed f slice must be rejected")
    except ValueError:
        pass
    plt.close(plot_decision_mesh(mixed.drop_duplicates(["g", "v_Hb"]), "z", reference_lines=False)[0])
    return saved


def reload_check(path):
    stored = np.load(path, allow_pickle=False)
    physical = pd.DataFrame(stored["physical"], columns=FULL_PAR_COLS)
    for name, predictor in load_predictors().items():
        for key, value in predictor.predict_details(physical).items():
            np.testing.assert_array_equal(value, stored[f"{name}__{key}"])
    print("Fresh-process predictor reload checks passed.")


def main():
    started = time.time()
    source_paths = [REPO_ROOT / f"data/processed/{name}.csv" for name in ("train", "val")]
    preserved = [p for p in (REPO_ROOT / "results/models").rglob("*") if p.is_file()] + source_paths
    hashes_before = {str(p.relative_to(REPO_ROOT)): digest(p) for p in preserved}
    from src.debbirth.data.load import load_splits
    frames = load_splits(REPO_ROOT / "data/processed")
    frames.pop("test")  # never used here
    rng = np.random.default_rng(42)  # the T06 validation sample
    indices = {"train": rng.choice(len(frames["train"]), 2000, replace=False),
               "val": rng.choice(len(frames["val"]), 500, replace=False)}
    val_frame = frames["val"].iloc[indices["val"]].reset_index(drop=True)
    physical = val_frame[list(FULL_PAR_COLS)].reset_index(drop=True)
    outdir = Path(create_run_outdir("t06a_validation"))

    predictors = load_predictors()
    summary = {"script_sha256": digest(Path(__file__)), "rows": len(physical), "test_set_used": False,
               "models": {name: str(path.relative_to(REPO_ROOT)) for name, (_, path) in MODELS.items()},
               "predictors": {}}
    probe = {"physical": physical.to_numpy()}
    for name, predictor in predictors.items():
        details, summary["predictors"][name] = check_predictor(name, predictor, physical)
        probe.update({f"{name}__{key}": value for key, value in details.items()})
    summary["nn_refactor_probes_identical"] = nn_refactor_probe(predictors)
    summary["evaluation"] = check_evaluation(predictors, val_frame)
    summary["training_consolidation_f1"] = check_training_consolidation(frames, indices)
    summary["plots"] = make_plots(predictors, outdir)
    np.savez(outdir / "reload_probe.npz", **probe)
    subprocess.run([sys.executable, "-m", "experiments.validate_t06a", "--reload", str(outdir / "reload_probe.npz")],
                   cwd=REPO_ROOT, check=True)
    assert {str(p.relative_to(REPO_ROOT)): digest(p) for p in preserved} == hashes_before
    summary["preserved_files_unchanged"] = len(hashes_before)
    summary["seconds"] = round(time.time() - started, 1)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    print("Run directory:", outdir)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--reload":
        reload_check(Path(sys.argv[2]))
    else:
        main()
