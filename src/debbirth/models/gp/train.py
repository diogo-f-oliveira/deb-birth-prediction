import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Tuple, Mapping
import numpy as np
from joblib import dump as joblib_dump

from ...data.load import load_data_gp
from .algorithm import DEBBirthSymbolicClassifier, create_gp_classifier
from .config import TrainGPConfig
from ...evaluate.predict import evaluate_binary_classifier
from ...evaluate.metrics import BinaryMetrics


def train_gp_classifier(cfg: TrainGPConfig) -> Dict[str, float]:
    """Train a GP classifier and evaluate it on the validation split."""

    np.random.seed(cfg.seed)

    cfg.outdir.mkdir(parents=True, exist_ok=True)

    features, targets = load_data_gp(cfg)

    # Assume data is already correctly formatted.
    X_train = features["train"]
    X_val = features["val"]
    y_train = targets["train"]
    y_val = targets["val"]

    model = create_gp_classifier(cfg)
    model.fit(X_train, y_train)

    val_metrics = evaluate_binary_classifier(model, X_val, y_val)

    save_gp_run(model=model, cfg=cfg, val_metrics=val_metrics)

    return {
        "model": model,
        "val_metrics": val_metrics,
        "features": features,
        "targets": targets,
        "history": getattr(model, "run_details_", None),
        "best_program": str(model._program) if hasattr(model, "_program") else None,
    }


def save_gp_run(*, model: DEBBirthSymbolicClassifier, cfg: TrainGPConfig, val_metrics: BinaryMetrics) -> None:
    """Persist model + config + validation metrics + run details."""


    with (cfg.outdir / "train_gp_config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, sort_keys=True, default=str)

    (cfg.outdir / "metrics").mkdir(exist_ok=True)
    with (cfg.outdir / "metrics" / "val_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(val_metrics), f, indent=2, sort_keys=True)

    (cfg.outdir / "model").mkdir(exist_ok=True)

    program_str = None
    if hasattr(model, "_program") and model._program is not None:  # type: ignore[attr-defined]
        program_str = str(model._program)  # type: ignore[attr-defined]

    if program_str is not None:
        (cfg.outdir / "model" / "best_program.txt").write_text(program_str + "", encoding="utf-8")

    joblib_dump(model, cfg.outdir / "model" / "gp_model.joblib")

    if hasattr(model, "run_details_"):
        details = getattr(model, "run_details_")
        write_gp_run_history(details, cfg.outdir / "history.csv")


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


if __name__ == "__main__":
    # Run example
    from ...data.schema import DatasetSpec
    from .config import GPConfig
    from .functions import *
    from ...utils.results import ensure_outdir

    data_spec = DatasetSpec(
        feature_set="dimensionless"
    )
    gp_cfg = GPConfig(
        # generations=50,
        # population_size=1000,
        # tournament_size=20,
        p_crossover=0.8,
        parsimony_coefficient=0.001,
        function_set=DEFAULT_FUNCTIONS + (ATAN,),
    )
    cfg = TrainGPConfig(
        gp=gp_cfg,
        data_spec=data_spec,
        outdir=None,
        verbose=1,
        class_weights="balanced",
        seed=42,
        num_workers=-1,
    )
    print(cfg)
    print()

    output = train_gp_classifier(cfg)
    print("\nBest program:")
    print(output["best_program"])

    test_metrics = evaluate_binary_classifier(model=output["model"], X=output['features']['test'],
                                              y=output['targets']['test'])
    print("\nTest metrics:")
    print(test_metrics)
