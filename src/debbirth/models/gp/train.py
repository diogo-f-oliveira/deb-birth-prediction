import csv
import random
from pathlib import Path
from typing import Any, Dict
from joblib import dump as joblib_dump, load as joblib_load
import json
import numpy as np

from .data import load_data_gp
from .algorithm import DEBBirthSymbolicClassifier, create_gp_classifier
from .boundary import GPBoundaryModel, fit_gp_boundary
from .config import TrainGPConfig
from .expression import gp_model_text, gp_program_text
from .predict import GPPredictor
from ...data.schema import DatasetSpec
from ...evaluate.metrics import BinaryMetrics
from ...utils.config import validate_training_mode
from ...utils.results import resolve_run_config, save_run_metadata
# from .symbolic import model_program_to_sympy_strings


def _check_prepared(prepared, cfg):
    if not {"train", "val"} <= prepared.keys():
        raise ValueError("Prepared training requires train and val records.")
    for name, split in prepared.items():
        if not len(split.labels) or tuple(split.feature_names) != tuple(cfg.data_spec.feature_cols):
            raise ValueError(f"Empty or mismatched prepared feature schema: {name}.")
        if not np.isin(split.labels, [0, 1]).all():
            raise ValueError("Prepared labels must be binary reached_birth.")
        if (split.log_nu_b is not None) != (cfg.data_spec.formulation == "boundary"):
            raise ValueError("Prepared offsets do not match the formulation.")


def train_gp_classifier(cfg: TrainGPConfig, save_run: bool = True, *, prepared=None,
                        data_metadata=None) -> Dict[str, Any]:
    """Train a GP model and evaluate it on the validation split.

    full_par/normalized evolve a classifier score; boundary evolves F = log(Psi)
    with the fixed-offset loss (see boundary.py). Explicit row-aligned
    `prepared` train/val records (test optional and unused) replace CSV
    loading; the caller then supplies provenance metadata.

    Args:
        cfg: TrainGPConfig
        save_run: if True, create the run directory and persist artifacts (default True).
                  if False, no directory is created and nothing is saved.
    """
    validate_training_mode(cfg, supports_boundary=True)
    # Set random seeds
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    if prepared is None:
        features, targets, prepared, data_metadata = load_data_gp(cfg, return_prepared=True)
    else:
        _check_prepared(prepared, cfg)
        features = {name: split.features for name, split in prepared.items()}
        targets = {name: split.labels for name, split in prepared.items()}
    weighting = "unweighted" if cfg.class_weights is None else f"class_weights={cfg.class_weights!r}"
    data_metadata = dict(data_metadata or {})
    data_metadata.update({
        "loss": (f"mean BCE of sigmoid((F - log_nu_b)/T) over active training rows ({weighting})"
                 if cfg.data_spec.formulation == "boundary"
                 else f"gplearn log loss of sigmoid(S) over active training rows ({weighting})"),
        "decision": ("F - log_nu_b > 0; equality infeasible; no tolerance" if cfg.data_spec.formulation == "boundary"
                     else "sigmoid(S) > 0.5 (gplearn argmax; ties infeasible)"),
        "final_program_selection": "gplearn: lowest raw (unpenalized) training loss in the last generation",
        "temperature_policy": "fixed training temperature; no calibration performed",
    })

    engine = None
    if cfg.data_spec.formulation == "boundary":
        model, engine = fit_gp_boundary(prepared["train"], cfg)
        val_metrics = GPPredictor(model, cfg.data_spec).evaluate_prepared(prepared["val"])
        history = engine.run_details_
    else:
        model = create_gp_classifier(cfg)
        model.fit(features["train"], targets["train"])
        val_metrics = GPPredictor(model, cfg.data_spec).evaluate_prepared(prepared["val"])
        history = getattr(model, "run_details_", None)

    # Only save artifacts when requested
    if save_run:
        cfg = save_gp_run(model=model, cfg=cfg, val_metrics=val_metrics, data_metadata=data_metadata,
                          history=history)

    return {
        "model": model,
        "engine": engine,  # boundary only: dataset-bound engine for diagnostics; never saved
        "train_config": cfg,
        "outdir": cfg.outdir if save_run else None,
        "prepared": prepared,
        "data_metadata": data_metadata,
        "val_metrics": val_metrics,
        "features": features,
        "targets": targets,
        "history": history,
        "best_program": gp_program_text(model),
    }


def save_gp_run(*, model, cfg: TrainGPConfig, val_metrics: BinaryMetrics,
                save_all_programs: bool = False, data_metadata=None, history=None) -> TrainGPConfig:
    """Persist model + config + validation metrics + run details.

    `model` is a DEBBirthSymbolicClassifier or a GPBoundaryModel. A boundary
    model holds no training context; pass its engine's run_details_ as `history`.

    Args:
      save_all_programs: if False (default) remove the attribute '_programs' from the model
                         before saving to avoid storing all intermediate programs. The original
                         model object is restored after saving. If True, the model is saved as-is.
    """
    boundary = isinstance(model, GPBoundaryModel)
    cfg = resolve_run_config(cfg, cfg.run_name or ("DEBBirthGPBoundary" if boundary else "DEBBirthSymbolicClassifier"))

    # Save train config
    cfg.save_json(cfg.outdir / "train_gp_config.json")
    save_run_metadata(cfg.outdir, cfg, data_metadata)

    # Save validation metrics
    val_metrics.save_json(cfg.outdir / "metrics" / "val_metrics.json")

    (cfg.outdir / "model").mkdir(exist_ok=True)

    (cfg.outdir / "model" / "best_program.txt").write_text(gp_program_text(model), encoding="utf-8")
    (cfg.outdir / "model" / "expression.txt").write_text(gp_model_text(model, cfg), encoding="utf-8")

    if boundary:
        joblib_dump(model, cfg.outdir / "model" / "gp_model.joblib")
        if history is not None:
            write_gp_run_history(history, cfg.outdir / "history.csv")
        return cfg

        # # Convert model program to sympy simplified string + srepr and save
        # try:
        #     res = model_program_to_sympy_strings(model)
        #     if res is not None:
        #         (cfg.outdir / "model" / "best_program_sympy.txt").write_text(res["simplified"] + "", encoding="utf-8")
        #         (cfg.outdir / "model" / "best_program_sympy.srepr").write_text(res["srepr"] + "", encoding="utf-8")
        # except Exception:
        #     # don't fail run saving if sympy conversion fails; just continue
        #     pass

    # When not saving all programs, temporarily remove _programs to avoid saving large histories
    restored_programs = None
    removed_programs = False
    if not save_all_programs and hasattr(model, "_programs"):
        try:
            restored_programs = getattr(model, "_programs")
            delattr(model, "_programs")
            removed_programs = True
        except Exception:
            restored_programs = None
            removed_programs = False

    # Also temporarily remove run_details_ (stored separately as CSV) to avoid duplicating it in the saved object
    restored_run_details = None
    removed_run_details = False
    if hasattr(model, "run_details_"):
        try:
            restored_run_details = getattr(model, "run_details_")
            delattr(model, "run_details_")
            removed_run_details = True
        except Exception:
            restored_run_details = None
            removed_run_details = False

    try:
        joblib_dump(model, cfg.outdir / "model" / "gp_model.joblib")
    finally:
        # restore _programs on the original model object if we removed it
        if removed_programs:
            try:
                setattr(model, "_programs", restored_programs)
            except Exception:
                # best-effort restore; do not raise if restore fails
                pass
        # restore run_details_ on the original model object if we removed it
        if removed_run_details:
            try:
                setattr(model, "run_details_", restored_run_details)
            except Exception:
                # best-effort restore; do not raise if restore fails
                pass

    # write run_details_ CSV if present (we restored it above)
    if hasattr(model, "run_details_"):
        details = getattr(model, "run_details_")
        write_gp_run_history(details, cfg.outdir / "history.csv")
    return cfg


def write_gp_run_history(run_details: Any, path: Path) -> None:
    """Write gplearn's run_details_ to CSV."""

    if not isinstance(run_details, dict):
        return

    keys = list(run_details.keys())
    if not keys:
        return

    n = len(run_details[keys[0]])
    for k in keys:
        if len(run_details[k]) != n:
            return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for i in range(n):
            row = {k: run_details[k][i] for k in keys}
            writer.writerow(row)


def load_gp_run(outdir: Path) -> Dict[str, Any]:
    """
    Load a previously saved GP run directory created by save_gp_run.

    Expects structure:
      outdir/
        train_gp_config.json
        model/gp_model.joblib

    Returns a dict:
      {
        "model": loaded joblib model (DEBBirthSymbolicClassifier, or GPBoundaryModel for boundary runs),
        "predictor": GPPredictor for original-parameter inference (None if the schema is unknown),
        "train_cfg": TrainGPConfig or None,
      }
    """
    outdir = Path(outdir)
    if not outdir.exists():
        raise FileNotFoundError(f"Run outdir not found: {outdir}")

    cfg_path = outdir / "train_gp_config.json"
    model_dir = outdir / "model"


    # load model via joblib
    model_path = model_dir / "gp_model.joblib"
    model = joblib_load(model_path)

    # load train config (as TrainGPConfig dataclass) via helper method
    train_cfg_obj = None
    if cfg_path.exists():
        try:
            train_cfg_obj = TrainGPConfig.load_json(cfg_path)
        except (ValueError, TypeError, KeyError) as exc:
            import warnings
            warnings.warn(f"GP model loaded for inference, but training config cannot be reconstructed: {exc}",
                          RuntimeWarning, stacklevel=2)
            train_cfg_obj = None

    # Archived full-parameter classifiers predate reconstructable configs; their
    # schema is the historical unscaled (g, k, v_Hb, f) order.
    spec = train_cfg_obj.data_spec if train_cfg_obj is not None else None
    if spec is None and isinstance(model, DEBBirthSymbolicClassifier) \
            and tuple(model.feature_names_no_constants) == tuple(DatasetSpec().feature_cols):
        spec = DatasetSpec()
    return {
        "model": model,
        "train_cfg": train_cfg_obj,
        "predictor": GPPredictor(model, spec) if spec is not None else None,
    }


if __name__ == "__main__":
    from ...utils.paths import REPO_ROOT

    cfg = TrainGPConfig.load_json(REPO_ROOT / "experiments/gp_full_par.json")
    output = train_gp_classifier(cfg, save_run=True)
    print("Validation metrics:", output["val_metrics"])
    print("Run directory:", output["outdir"])
