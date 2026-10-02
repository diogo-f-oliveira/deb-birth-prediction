from __future__ import annotations

import json
import csv
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from .structure import build_net
from .predict import NNPredictor
from .config import TrainDEBBirthNetConfig, DEBBirthNetConfig
from .data import load_data_pytorch, tensor_data_from_prepared
from ...data.scalers import save_scaler, TorchStandardScaler, load_scaler
from ...evaluate.metrics import compute_pos_weight, EpochBinaryMetrics
from ...evaluate.predict import evaluate_pytorch_binary_classifier, logits_from_batch
from ...utils.pytorch import set_seed, resolve_device
from ...utils.config import validate_training_mode
from ...utils.results import resolve_run_config, save_run_metadata


def save_run_results(cfg: TrainDEBBirthNetConfig, history: List[EpochBinaryMetrics], model: torch.nn.Module,
                     scaler: TorchStandardScaler | None, data_metadata=None, *, selected_epoch=None) -> TrainDEBBirthNetConfig:
    if not history:
        raise ValueError("Cannot save a run without epoch metrics.")
    if selected_epoch is None:
        if cfg.checkpoint_selection != "final_epoch":
            raise ValueError("best_val_loss saving requires the selected epoch matching the supplied weights.")
        selected_epoch = history[-1].epoch
    selected = next((row for row in history if row.epoch == selected_epoch), None)
    if selected is None:
        raise ValueError("Selected epoch is missing from history.")
    cfg = resolve_run_config(cfg, "DEBBirthNet")
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save training history in .csv
    hist_csv_path = outdir / "history.csv"
    rows = [asdict(h) for h in history]
    if rows:
        # Use the keys of the first row as header (consistent across EpochBinaryMetrics)
        fieldnames = list(rows[0].keys())
        with hist_csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
    else:
        # create empty file if no history
        hist_csv_path.write_text("")

    # Create metrics/ subdir
    metrics_dir = outdir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Metrics correspond to selected weights, even if training continued later.
    metrics_path = metrics_dir / "val_metrics.json"
    selected.save_json(metrics_path)
    checkpoint = {"policy": cfg.checkpoint_selection, "selected_epoch": selected_epoch,
                  "metric": "val_loss" if cfg.checkpoint_selection == "best_val_loss" else None,
                  "selected_val_loss": selected.val_loss,
                  "completed_epochs": history[-1].epoch,
                  "tie_policy": "earliest epoch" if cfg.checkpoint_selection == "best_val_loss" else None}
    (outdir / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    # Save model_state_dict and scaler so they can later be loaded
    model_dir = outdir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # also save raw state_dict for convenience
    torch.save(model.state_dict(), model_dir / "model_state_dict.pth")

    # save config as JSON
    cfg.save_json(cfg.outdir / "train_nn_config.json")
    save_run_metadata(cfg.outdir, cfg, data_metadata)

    # save scaler using scaler-native saver
    scaler_path = model_dir / "scaler.pth"
    if scaler is not None:
        save_scaler(scaler, scaler_path)
    elif scaler_path.exists():
        # An explicit reused run directory must not retain a previous scaler.
        scaler_path.unlink()
    return cfg


def load_trained_nn(outdir: Any, device: Any = None) -> Dict[str, Any]:
    """
    Load a trained model, scaler and config from a saved run directory.

    Args:
      outdir: path to run output dir (string or Path). Expects a 'model/' subdir and a JSON config.
      device: target device for the returned model (string or torch.device). If None, uses cpu.

    Returns:
      dict with keys:
        - "model": instantiated DEBBirthNet with loaded state_dict
        - "scaler": loaded scaler instance or None
        - "train_cfg": TrainDEBBirthNetConfig instance or None
        - "net_cfg": DEBBirthNetConfig instance used to construct DEBBirthNet
    """

    outdir = Path(outdir)
    model_dir = outdir / "model"

    # Load config (use the dataclass loader)
    cfg_path = outdir / "train_nn_config.json"
    train_cfg = TrainDEBBirthNetConfig.load_json(cfg_path)

    # Candidate model state files (state_dict or checkpoint)
    state_path = model_dir / "model_state_dict.pth"

    # Determine device / map_location
    if device is None:
        map_device = torch.device("cpu")
    else:
        map_device = resolve_device(device) if isinstance(device, (str, Path)) else device

    # Load the state file (could be a dict containing 'model_state_dict' or a raw state_dict)
    loaded = torch.load(state_path, map_location=map_device, weights_only=True)
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        state_dict = loaded["model_state_dict"]
    else:
        # assume it's a raw state_dict
        state_dict = loaded

    # Instantiate and load state dict
    model = build_net(train_cfg)
    model.load_state_dict(state_dict)
    model = model.to(map_device)
    model.eval()

    # Attempt to load scaler
    scaler_path = model_dir / "scaler.pth"
    scaler = None
    if scaler_path.exists():
        scaler = load_scaler(scaler_path, map_location=map_device)

    return {
        "model": model,
        "scaler": scaler,
        "train_cfg": train_cfg,
        "net_cfg": train_cfg.net_config,
        "predictor": NNPredictor(model, train_cfg, scaler),
        "checkpoint": (json.loads((outdir / "checkpoint.json").read_text(encoding="utf-8"))
                       if (outdir / "checkpoint.json").exists() else None),
    }


def train_net(cfg: TrainDEBBirthNetConfig, save: bool = False, *, prepared=None, data_metadata=None) -> Dict[str, Any]:
    """Train from configured CSVs or explicit, row-aligned prepared subsets.

    An explicit prepared mapping needs train/val; test is optional and unused.
    Its caller supplies provenance metadata for reproducible subset experiments.
    """
    validate_training_mode(cfg, supports_boundary=True)
    if cfg.net_config is None or cfg.net_config.input_dim != cfg.data_spec.n_features:
        raise ValueError("net_config.input_dim must match data_spec.n_features.")
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    if prepared is None:
        scaled_input_data, targets, dataloaders, datasets, scaler, prepared, data_metadata = load_data_pytorch(
            cfg, return_prepared=True)
    else:
        if not {"train", "val"} <= prepared.keys():
            raise ValueError("Prepared training requires train and val records.")
        for name, split in prepared.items():
            if not len(split.labels) or tuple(split.feature_names) != tuple(cfg.data_spec.feature_cols):
                raise ValueError(f"Empty or mismatched prepared feature schema: {name}.")
            if not np.isin(split.labels, [0, 1]).all():
                raise ValueError("Prepared labels must be binary reached_birth.")
            if (split.log_nu_b is not None) != (cfg.data_spec.formulation == "boundary"):
                raise ValueError("Prepared offsets do not match the formulation.")
        scaled_input_data, targets, dataloaders, datasets, scaler = tensor_data_from_prepared(
            prepared, scaling_type=cfg.scaling_type, batch_size=cfg.batch_size, device=device,
            num_workers=cfg.num_workers, seed=cfg.seed)
    data_metadata = dict(data_metadata or {})
    data_metadata.update({
        "loss": "mean BCEWithLogitsLoss over rows; positive terms multiplied by pos_weight when enabled",
        "decision": ("F - log_nu_b > 0; equality infeasible; no tolerance" if cfg.data_spec.formulation == "boundary"
                     else f"probability >= {cfg.net_config.threshold}"),
        "temperature_policy": "fixed training temperature; no calibration performed",
        "checkpoint_selection": cfg.checkpoint_selection,
    })

    # -------------------------
    # Model
    # -------------------------
    model = build_net(cfg).to(device)

    # Loss + Optimizer
    if cfg.use_pos_weight:
        # If you already computed pos_weight externally, pass it via cfg.pos_weight
        if cfg.pos_weight is None:
            if not (targets['train'] == 0).any() or not (targets['train'] == 1).any():
                raise ValueError("Automatic positive weighting requires both training classes.")
            cfg = replace(cfg, pos_weight=compute_pos_weight(targets['train'].to(device)).item())
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.pos_weight, device=device))
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: List[EpochBinaryMetrics] = []
    selected_epoch, best_loss, selected_state = None, float("inf"), None

    # Train epochs (no early stopping)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_train_loss, train_count = 0.0, 0

        for batch in dataloaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits, y, _ = logits_from_batch(model, batch, device=device,
                                              formulation=cfg.data_spec.formulation, temperature=cfg.boundary_temperature)
            loss = loss_fn(logits, y)
            if not torch.isfinite(loss) or not torch.isfinite(logits).all():
                raise FloatingPointError("Nonfinite training loss/logits.")
            loss.backward()
            optimizer.step()

            total_train_loss += float(loss.item()) * len(y)
            train_count += len(y)

        train_loss = total_train_loss / train_count

        # Validate
        val_metrics, val_loss = evaluate_pytorch_binary_classifier(
            model, dataloaders['val'], loss_fn, device=device,
            formulation=cfg.data_spec.formulation, temperature=cfg.boundary_temperature)
        if not np.isfinite(val_loss):
            raise FloatingPointError("Validation must contain rows with finite loss.")

        # build an EpochBinaryMetrics instance (includes all BinaryMetrics fields + epoch/train_loss/val_loss)
        epoch_row = EpochBinaryMetrics.from_binary_metrics(
            val_metrics,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
        )
        history.append(epoch_row)
        if cfg.checkpoint_selection == "final_epoch":
            selected_epoch = epoch
        elif val_loss < best_loss:
            selected_epoch, best_loss = epoch, val_loss
            selected_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

        print(
            f"[{epoch:03d}/{cfg.epochs}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_f1_macro={val_metrics.f1_macro:.3f} val_f1_pos={val_metrics.f1_pos:.3f} val_f1_neg={val_metrics.f1_neg:.3f}"
        )

    if selected_state is not None:
        model.load_state_dict(selected_state)
    model.eval()
    data_metadata["selected_epoch"] = selected_epoch
    # Save the selected weights and their corresponding metrics.
    if save:
        cfg = save_run_results(cfg, history, model, scaler, data_metadata=data_metadata,
                               selected_epoch=selected_epoch)

    # Pack and return
    return {
        "model": model,
        "history": history,
        "train_config": cfg,
        "outdir": cfg.outdir if save else None,
        "prepared": prepared,
        "data_metadata": data_metadata,
        "val_metrics": next(row for row in history if row.epoch == selected_epoch),
        "selected_epoch": selected_epoch,
        "predictor": NNPredictor(model, cfg, scaler),
        "scaled_input_data": scaled_input_data,
        "targets": targets,
        "dataloaders": dataloaders,
        "datasets": datasets,
        "scaler": scaler,
    }


if __name__ == "__main__":
    from ...utils.paths import REPO_ROOT

    cfg = TrainDEBBirthNetConfig.load_json(REPO_ROOT / "experiments/nn_full_par.json")
    output = train_net(cfg, save=True)
    print("Validation metrics:", output["val_metrics"])
    print("Run directory:", output["outdir"])
