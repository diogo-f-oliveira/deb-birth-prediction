"""Shared training command line for the GP and NN families (T08A).

Fits and validates one run with the same trainers used from Python
(`train_gp_classifier`, `train_net`); it contains no training logic. Held-out
test evaluation is never performed here.

Example (from the repository root):

    python -m src.debbirth.train --model nn --formulation boundary --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime
from hashlib import sha256
from pathlib import Path

from .formulations import FORMULATIONS
from .utils.config import config_dict, validate_training_mode
from .utils.paths import portable_path, resolve_repo_path

FAMILY_KEYS = {"gp": "gp", "nn": "net_config"}  # top-level key identifying each family's config
SAVED_CONFIG = {"gp": "train_gp_config.json", "nn": "train_nn_config.json"}
# CLI flag -> config field; only runtime/identity settings, never hyperparameters.
OVERRIDES = {"seed": "seed", "data_dir": "data_dir", "run_name": "run_name",
             "num_workers": "num_workers", "device": "device"}

DESCRIPTION = """Train and validate one GP or NN run from an experiment JSON.

Settings resolve as: dataclass defaults, then the config file, then flags given
explicitly on the command line. Hyperparameters (including boundary_temperature)
are set only in the config file. A config's own outdir is treated as provenance:
each run gets a new results/runs/<timestamp>_<name>/ directory unless --outdir is
given. To reproduce a saved run, pass its train_*_config.json as --config.
The test split is never evaluated."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.debbirth.train", description=DESCRIPTION,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=sorted(FAMILY_KEYS), help="model family")
    parser.add_argument("--formulation", required=True, choices=FORMULATIONS,
                        help="must match the config's data_spec.formulation")
    parser.add_argument("--config", type=Path,
                        help="experiment JSON (default: experiments/<model>_<formulation>.json); "
                             "relative paths resolve against the repository root")
    parser.add_argument("--seed", type=int, help="random seed")
    parser.add_argument("--data-dir", type=Path, help="directory with train/val/test CSVs")
    parser.add_argument("--outdir", type=Path,
                        help="exact run directory; must not exist or be empty (default: new timestamped directory)")
    parser.add_argument("--run-name", help="suffix of the auto-created run directory")
    parser.add_argument("--num-workers", type=int, help="GP: gplearn n_jobs; NN: DataLoader workers")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), help="NN only: training device")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve and validate the config, print it and exit without loading data or training")
    return parser


def resolve_config(args, parser):
    """Load, check and override the config; no data access or directory creation."""
    if args.model == "gp" and args.device is not None:
        parser.error("--device applies only to --model nn.")
    config_path = resolve_repo_path(args.config or f"experiments/{args.model}_{args.formulation}.json")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read config {config_path}: {exc}")
    if not isinstance(raw, dict):
        parser.error(f"config {config_path} is not a JSON object.")
    family_key = FAMILY_KEYS[args.model]
    if family_key not in raw:
        other = next(name for name, key in FAMILY_KEYS.items() if key in raw) if any(
            key in raw for key in FAMILY_KEYS.values()) else None
        hint = f"; it looks like a --model {other} config" if other else ""
        parser.error(f"config {config_path} has no '{family_key}' section required by --model {args.model}{hint}.")

    if args.model == "gp":
        from .models.gp.config import TrainGPConfig as config_cls
    else:
        from .models.nn.config import TrainDEBBirthNetConfig as config_cls
    try:
        cfg = config_cls.load_json(config_path)
    except (ValueError, TypeError, KeyError) as exc:
        parser.error(f"invalid {args.model} config {config_path}: {exc}")
    if cfg.data_spec.formulation != args.formulation:
        parser.error(f"--formulation {args.formulation} does not match the config's "
                     f"data_spec.formulation {cfg.data_spec.formulation!r} ({config_path}).")

    overrides = {field: getattr(args, flag) for flag, field in OVERRIDES.items() if getattr(args, flag) is not None}
    outdir = None
    if args.outdir is not None:
        outdir = resolve_repo_path(args.outdir)
        if outdir.exists() and (not outdir.is_dir() or any(outdir.iterdir())):
            parser.error(f"--outdir {outdir} already exists and is not an empty directory.")
    try:
        cfg = replace(cfg, **overrides, outdir=outdir)
        validate_training_mode(cfg, supports_boundary=True)
        if args.model == "nn" and (cfg.net_config is None or cfg.net_config.input_dim != cfg.data_spec.n_features):
            raise ValueError("net_config.input_dim must match data_spec.n_features "
                             f"({cfg.data_spec.n_features}: {', '.join(cfg.data_spec.feature_cols)}).")
    except (ValueError, TypeError, NotImplementedError) as exc:
        parser.error(f"invalid resolved config: {exc}")
    if args.outdir is not None:
        overrides["outdir"] = args.outdir
    return cfg, config_path, overrides


def write_invocation(outdir, args, argv, config_path, overrides):
    """Record how this run was launched next to the trainer's saved config."""
    saved = Path(outdir) / SAVED_CONFIG[args.model]
    record = {
        "argv": list(argv),
        "model": args.model,
        "formulation": args.formulation,
        "config": portable_path(config_path),
        "config_sha256": sha256(config_path.read_bytes()).hexdigest(),
        "overrides": {key: portable_path(value) if isinstance(value, Path) else value
                      for key, value in overrides.items()},
        "saved_config": portable_path(saved),
        "reproduce": (f"python -m src.debbirth.train --model {args.model} --formulation {args.formulation} "
                      f"--config {portable_path(saved)}"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    (Path(outdir) / "cli_invocation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def print_summary(args, output):
    m = output["val_metrics"]
    print(f"\nRun directory: {output['outdir']}")
    print(f"Validation ({args.model}, {args.formulation}): f1_macro={m.f1_macro:.4f} mcc={m.mcc:.4f} "
          f"recall_pos={m.recall_pos:.4f} recall_neg={m.recall_neg:.4f} log_loss={m.log_loss:.4f}")
    print(f"  fp (infeasible accepted)={m.fp}  fn (feasible rejected)={m.fn}  tp={m.tp}  tn={m.tn}")
    if args.model == "gp":
        print(f"  best program: {output['best_program']}")
    else:
        print(f"  checkpoint: {output['train_config'].checkpoint_selection}, "
              f"selected epoch {output['selected_epoch']}")


def main(argv=None):
    """Run the CLI; returns the trainer's output dict, or None for --dry-run."""
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg, config_path, overrides = resolve_config(args, parser)

    print(f"Model: {args.model}  formulation: {args.formulation}  config: {portable_path(config_path)}")
    print(f"Overrides: {json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in overrides.items()})}")
    print("Resolved config:\n" + json.dumps(config_dict(cfg), indent=2, sort_keys=True))
    if args.dry_run:
        print("Dry run: no data loaded, nothing trained or saved.")
        return None

    if args.model == "gp":
        from .models.gp.train import train_gp_classifier
        output = train_gp_classifier(cfg, save_run=True)
    else:
        from .models.nn.train import train_net
        output = train_net(cfg, save=True)
    write_invocation(output["outdir"], args, argv, config_path, overrides)
    print_summary(args, output)
    return output


if __name__ == "__main__":
    main()
