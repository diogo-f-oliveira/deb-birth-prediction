import csv
import random
from pathlib import Path
from typing import Any, Dict
from joblib import dump as joblib_dump, load as joblib_load
import json
import numpy as np

from .data import load_data_gp
from .algorithm import DEBBirthSymbolicClassifier, create_gp_classifier
from .config import TrainGPConfig
from ...evaluate.predict import evaluate_binary_classifier
from ...evaluate.metrics import BinaryMetrics
from ...utils.config import validate_training_mode
from ...utils.results import resolve_run_config, save_run_metadata
# from .symbolic import model_program_to_sympy_strings


def train_gp_classifier(cfg: TrainGPConfig, save_run: bool = True) -> Dict[str, Any]:
    """Train a GP classifier and evaluate it on the validation split.

    Args:
        cfg: TrainGPConfig
        save_run: if True, create the run directory and persist artifacts (default True).
                  if False, no directory is created and nothing is saved.
    """
    validate_training_mode(cfg)
    # Set random seeds
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    features, targets, prepared, data_metadata = load_data_gp(cfg, return_prepared=True)

    # Assume data is already correctly formatted.
    X_train = features["train"]
    X_val = features["val"]
    y_train = targets["train"]
    y_val = targets["val"]

    model = create_gp_classifier(cfg)
    model.fit(X_train, y_train)

    val_metrics = evaluate_binary_classifier(model, X_val, y_val)

    # Only save artifacts when requested
    if save_run:
        cfg = save_gp_run(model=model, cfg=cfg, val_metrics=val_metrics, data_metadata=data_metadata)

    return {
        "model": model,
        "train_config": cfg,
        "outdir": cfg.outdir if save_run else None,
        "prepared": prepared,
        "data_metadata": data_metadata,
        "val_metrics": val_metrics,
        "features": features,
        "targets": targets,
        "history": getattr(model, "run_details_", None),
        "best_program": str(model._program) if hasattr(model, "_program") else None,
    }


def save_gp_run(*, model: DEBBirthSymbolicClassifier, cfg: TrainGPConfig, val_metrics: BinaryMetrics,
                save_all_programs: bool = False, data_metadata=None) -> TrainGPConfig:
    """Persist model + config + validation metrics + run details.

    Args:
      save_all_programs: if False (default) remove the attribute '_programs' from the model
                         before saving to avoid storing all intermediate programs. The original
                         model object is restored after saving. If True, the model is saved as-is.
    """
    cfg = resolve_run_config(cfg, cfg.run_name or "DEBBirthSymbolicClassifier")

    # Save train config
    cfg.save_json(cfg.outdir / "train_gp_config.json")
    save_run_metadata(cfg.outdir, cfg, data_metadata)

    # Save validation metrics
    val_metrics.save_json(cfg.outdir / "metrics" / "val_metrics.json")

    (cfg.outdir / "model").mkdir(exist_ok=True)

    program_str = None
    if hasattr(model, "_program") and model._program is not None:  # type: ignore[attr-defined]
        program_str = str(model._program)  # type: ignore[attr-defined]

    if program_str is not None:
        (cfg.outdir / "model" / "best_program.txt").write_text(program_str + "", encoding="utf-8")

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
        "model": loaded joblib model,
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

    return {
        "model": model,
        "train_cfg": train_cfg_obj,
    }


if __name__ == "__main__":
    from ...utils.paths import REPO_ROOT

    cfg = TrainGPConfig.load_json(REPO_ROOT / "experiments/gp_full_par.json")
    output = train_gp_classifier(cfg, save_run=True)
    print("Validation metrics:", output["val_metrics"])
    print("Run directory:", output["outdir"])
