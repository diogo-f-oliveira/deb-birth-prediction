from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime

import numpy as np
import torch

from .structure import DEBBirthNet
from .config import TrainDEBBirthNetConfig
from ...data.load import load_data_pytorch
from ...evaluate.metrics import compute_pos_weight
from ...evaluate.predict import evaluate_pytorch_binary_classifier
from ...utils.results import ensure_outdir, create_run_outdir  # added create_run_outdir
from ...utils.pytorch import set_seed, resolve_device


# Helpers

def maybe_ray_report(metrics: Dict[str, Any]) -> None:
    """Report to Ray Tune if available; otherwise do nothing."""
    try:
        from ray.air import session  # type: ignore
        session.report(metrics)
        return
    except Exception:
        pass

    try:
        from ray import tune  # type: ignore
        tune.report(**metrics)
        return
    except Exception:
        pass


def save_checkpoint(path: Path, model: torch.nn.Module, train_cfg: TrainDEBBirthNetConfig, scaler: Any) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "train_cfg": asdict(train_cfg),
        "scaler": scaler,  # can be None, or a state_dict, or anything serializable
    }
    torch.save(payload, path)


# -----------------------------
# Core training
# -----------------------------

def train_net(cfg: TrainDEBBirthNetConfig) -> Dict[str, Any]:
    # Generate an outdir for this run inside "<base>/results/runs/<timestamp>_<model_type>"
    model_type = DEBBirthNet.__name__
    base = cfg.outdir if cfg.outdir else Path.cwd()
    generated_outdir = create_run_outdir(model_type, base=base)
    # store Path in config
    cfg.outdir = generated_outdir

    outdir = ensure_outdir(cfg.outdir)
    # convert Path to str for JSON serialization
    cfg_dict = asdict(cfg)
    cfg_dict["outdir"] = str(cfg.outdir)
    (outdir / "config.json").write_text(json.dumps(cfg_dict, indent=2))

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)

    scaled_input_data, targets, dataloaders, datasets, scalers = load_data_pytorch(cfg)

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

    history: List[Dict[str, Any]] = []

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

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_metrics.accuracy,
            "val_precision": val_metrics.precision,
            "val_recall": val_metrics.recall,
            "val_f1": val_metrics.f1,
            "val_auroc": val_metrics.auroc,
            "val_avg_precision": val_metrics.avg_precision,
        }
        history.append(row)
        # maybe_ray_report(row)

        print(
            f"[{epoch:03d}/{cfg.epochs}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_f1={val_metrics.f1:.3f} val_auroc={val_metrics.auroc:.3f} val_ap={val_metrics.avg_precision:.3f}"
        )

    (outdir / "history.json").write_text(json.dumps(history, indent=2))

    # Pack and return
    return {
        "model": model,
        "history": history,
        "scaled_input_data": scaled_input_data,
        "targets": targets,
        "dataloaders": dataloaders,
        "datasets": datasets,
        "scalers": scalers,
    }


# -----------------------------
# Ray Tune entrypoint
# -----------------------------

def trainable(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ray Tune entrypoint. Expects keys compatible with TrainConfig.

    Tip: In Ray trials, set outdir=os.getcwd() so each trial writes to its own directory.
    """
    if "outdir" not in config or not config["outdir"]:
        config = dict(config)
        config["outdir"] = os.getcwd()

    # Allow hidden_dims from Tune to be list-like or comma-separated
    if isinstance(config.get("hidden_dims"), str):
        config = dict(config)
        config["hidden_dims"] = [int(s.strip()) for s in config["hidden_dims"].split(",") if s.strip()]

    cfg = TrainDEBBirthNetConfig(**config)
    return train_net(cfg)


if __name__ == "__main__":
    # Run example
    from ...data.schema import DatasetSpec
    from .config import DEBBirthNetConfig

    data_spec = DatasetSpec(
        feature_set="dimensionless"
    )
    net_config = DEBBirthNetConfig(
        hidden_dims=[32, 32, 32],
        dropout=0.2,
        input_dim=data_spec.n_features,
    )
    out_dir = ensure_outdir("")
    cfg = TrainDEBBirthNetConfig(
        outdir=out_dir,
        data_spec=data_spec,
        data_dir="data/processed/",
        epochs=50,
        batch_size=64,
        lr=1e-4,
        weight_decay=1e-3,
        net_config=net_config,
        scaling_type='log_standardize',
        use_pos_weight=True,
        seed=42,
        num_workers=4,
        device="cpu",
    )
    output = train_net(cfg)

    test_metrics, test_loss = evaluate_pytorch_binary_classifier(
        model=output["model"],
        dataloader=output["dataloaders"]["test"],
        loss_fn=torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(cfg.pos_weight) if cfg.use_pos_weight and cfg.pos_weight is not None else None
        ),
        device=resolve_device(cfg.device)
    )
    print("\nTest metrics:")
    print(test_metrics)
