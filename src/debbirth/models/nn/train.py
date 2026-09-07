from __future__ import annotations

import json
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from .structure import DEBBirthNet
from .config import TrainDEBBirthNetConfig, DEBBirthNetConfig
from .data import load_data_pytorch
from ...data.scalers import save_scaler, TorchStandardScaler, load_scaler
from ...evaluate.metrics import compute_pos_weight, EpochBinaryMetrics
from ...evaluate.predict import evaluate_pytorch_binary_classifier
from ...utils.pytorch import set_seed, resolve_device
from ...utils.config import validate_training_mode
from ...utils.results import resolve_run_config, save_run_metadata


def save_run_results(cfg: TrainDEBBirthNetConfig, history: List[EpochBinaryMetrics], model: torch.nn.Module,
                     scaler: TorchStandardScaler | None, data_metadata=None) -> TrainDEBBirthNetConfig:
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

    # Save val_metrics.json (last entry from history)
    metrics_path = metrics_dir / "val_metrics.json"
    last = history[-1]
    # use dataclass serializer on BinaryMetrics/EpochBinaryMetrics
    last.save_json(metrics_path)

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
    model = DEBBirthNet(train_cfg.net_config)
    model.load_state_dict(state_dict)
    model = model.to(map_device)

    # Attempt to load scaler
    scaler_path = model_dir / "scaler.pth"
    scaler = None
    if scaler_path.exists():
        scaler = load_scaler(scaler_path, map_location=map_device)

    return {
        "model": model,
        "scaler": scaler,
        "train_cfg": train_cfg,
    }


def train_net(cfg: TrainDEBBirthNetConfig, save: bool = False) -> Dict[str, Any]:
    validate_training_mode(cfg)
    if cfg.net_config is None or cfg.net_config.input_dim != cfg.data_spec.n_features:
        raise ValueError("net_config.input_dim must match data_spec.n_features.")
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    scaled_input_data, targets, dataloaders, datasets, scaler, prepared, data_metadata = load_data_pytorch(
        cfg, return_prepared=True)

    # -------------------------
    # Model
    # -------------------------
    model_cfg = cfg.net_config
    model = DEBBirthNet(model_cfg).to(device)

    # Loss + Optimizer
    if cfg.use_pos_weight:
        # If you already computed pos_weight externally, pass it via cfg.pos_weight
        if cfg.pos_weight is None:
            cfg.pos_weight = compute_pos_weight(targets['train'].to(device)).item()
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.pos_weight, device=device))
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: List[EpochBinaryMetrics] = []

    # Train epochs (no early stopping)
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        batch_losses: List[float] = []

        for x, y in dataloaders["train"]:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            batch_losses.append(float(loss.item()))

        train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")

        # Validate
        val_metrics, val_loss = evaluate_pytorch_binary_classifier(model, dataloaders['val'], loss_fn, device=device)

        # build an EpochBinaryMetrics instance (includes all BinaryMetrics fields + epoch/train_loss/val_loss)
        epoch_row = EpochBinaryMetrics.from_binary_metrics(
            val_metrics,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
        )
        history.append(epoch_row)

        print(
            f"[{epoch:03d}/{cfg.epochs}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_f1_macro={val_metrics.f1_macro:.3f} val_f1_pos={val_metrics.f1_pos:.3f} val_f1_neg={val_metrics.f1_neg:.3f}"
        )

    # Save outputs: CSV history, last epoch metrics JSON, and model + scaler
    if save:
        cfg = save_run_results(cfg, history, model, scaler, data_metadata=data_metadata)

    # Pack and return
    return {
        "model": model,
        "history": history,
        "train_config": cfg,
        "outdir": cfg.outdir if save else None,
        "prepared": prepared,
        "data_metadata": data_metadata,
        "val_metrics": history[-1],
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
