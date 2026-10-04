"""JSON-based hyperparameter search for GP and NN models (T08B).

A search starts from an ordinary experiment JSON (the base config) and a
search space: a dict whose values are either fixed settings or Ray Tune
domains. `resolve_trial` maps each sampled key explicitly onto the family's
config fields, computing dependent parameters; unknown, conflicting or
tuner-managed keys are rejected. `run_search` runs Ray Tune with HyperOpt,
saving every trial as a normal run directory, so the selected trial is the
final artifact and is reproducible from its saved training JSON without the
tuner. Trials receive only the train and validation records; the test split
is never evaluated here.

Layout of results/tune/<timestamp>_<name>/:
    search.json        base config, script identity, space, objective, budget, versions
    base_config.json   resolved base config; trials reload it by registered names
    run_metadata.json  code/data provenance of the base config
    runs/<trial_id>/   one complete run per trial, plus trial.json (sampled params)
    ray/               Ray Tune's own state and per-trial logs
    trials.csv         one row per trial: status, sampled params, metrics, run directory
    best.json          selected trial and the command reproducing it
"""
from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, fields, replace
from datetime import datetime
from hashlib import sha256
from pathlib import Path

from .train import SAVED_CONFIG
from .utils.config import config_dict, validate_training_mode
from .utils.paths import REPO_ROOT, portable_path, resolve_repo_path

FAMILIES = ("gp", "nn")

# Searchable keys, mapped explicitly onto config fields.
GP_FIELDS = ("population_size", "generations", "tournament_size", "parsimony_coefficient", "init_depth",
             "init_method", "p_crossover", "p_subtree_mutation", "p_hoist_mutation", "p_point_mutation",
             "p_point_replace", "function_set", "constants")
GP_TRAIN_FIELDS = ("boundary_temperature", "class_weights", "progress_every", "verbose", "num_workers")
# Dependent GP parameters: tournament_fraction -> tournament_size; the operator group -> crossover and
# mutation probabilities (crossover = 1 - reproduction - mutation total; shares from mutation_shares).
GP_OPERATOR_GROUP = ("p_reproduction", "p_mutation_total", "mutation_u1", "mutation_u2")
GP_OPERATOR_FIELDS = ("p_crossover", "p_subtree_mutation", "p_hoist_mutation", "p_point_mutation")
NN_FIELDS = ("epochs", "batch_size", "lr", "weight_decay", "scaling_type", "checkpoint_selection",
             "boundary_temperature", "use_pos_weight", "progress_every", "num_workers", "device")
NN_NET_FIELDS = ("hidden_dims", "dropout")
NN_WIDTH_GROUP = ("n_layers", "width")  # -> hidden_dims = [width] * n_layers
# Fixed by the base config or managed by the tuner; repeated training seeds belong to T09.
MANAGED_KEYS = ("seed", "data_spec", "data_dir", "data_splits", "outdir", "run_name")
INT_KEYS = {"population_size", "generations", "tournament_size", "progress_every", "verbose", "num_workers",
            "epochs", "batch_size", "n_layers", "width"}
SUMMARY_METRICS = ("f1_macro", "mcc", "recall_pos", "recall_neg", "fp", "fn", "log_loss")


def allowed_keys(family):
    if family == "gp":
        return set(GP_FIELDS + GP_TRAIN_FIELDS + GP_OPERATOR_GROUP) | {"tournament_fraction"}
    if family == "nn":
        return set(NN_FIELDS + NN_NET_FIELDS + NN_WIDTH_GROUP)
    raise ValueError(f"Unknown family {family!r}; expected {FAMILIES}.")


def mutation_shares(u1, u2, a1=1, a2=1, a3=1, eps=1e-12):
    """Map (u1, u2) in [0, 1]^2 deterministically to Dirichlet(a1, a2, a3) shares.

    Returns (subtree, hoist, point) shares summing to 1; moved from the former
    GP tuner (calibrate.py) unchanged.
    """
    from scipy.stats import beta
    u1 = min(max(u1, eps), 1 - eps)
    u2 = min(max(u2, eps), 1 - eps)
    v1 = beta.ppf(u1, a1, a2 + a3)
    v2 = beta.ppf(u2, a2, a3)
    return float(v1), float((1 - v1) * v2), float((1 - v1) * (1 - v2))


def check_keys(family, keys):
    """Reject unknown, tuner-managed, conflicting and incomplete key sets."""
    keys = set(keys)
    managed = sorted(keys & set(MANAGED_KEYS))
    if managed:
        raise ValueError(f"Keys {managed} are fixed by the base config or managed by the tuner.")
    unknown = sorted(keys - allowed_keys(family))
    if unknown:
        raise ValueError(f"Unknown {family} search keys {unknown}; allowed: {sorted(allowed_keys(family))}.")
    groups = ((GP_OPERATOR_GROUP, GP_OPERATOR_FIELDS), (("tournament_fraction",), ("tournament_size",))) \
        if family == "gp" else ((NN_WIDTH_GROUP, ("hidden_dims",)),)
    for group, replaced in groups:
        present = keys & set(group)
        if present and present != set(group):
            raise ValueError(f"Keys {sorted(group)} must be given together; got {sorted(present)}.")
        if present and keys & set(replaced):
            raise ValueError(f"{sorted(group)} determine {sorted(replaced)}; do not set both.")


def _plain(key, value):
    """Convert numpy scalars and integral floats (e.g. from quniform) to plain values."""
    if hasattr(value, "item") and not isinstance(value, (list, tuple, dict)):
        value = value.item()
    if key in INT_KEYS and not isinstance(value, bool):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if not isinstance(value, int):
            raise ValueError(f"{key} must be an integer, got {value!r}.")
    return value


def resolve_trial(base_cfg, family, params):
    """Return the validated training config for one trial; the base is not changed."""
    check_keys(family, params)
    params = {key: _plain(key, value) for key, value in params.items()}
    if family == "gp":
        from .models.gp.constants import NAMED_CONSTANT_SETS
        from .models.gp.functions import NAMED_FUNCTION_SETS
        updates = {key: params[key] for key in GP_FIELDS if key in params}
        for key, named in (("function_set", NAMED_FUNCTION_SETS), ("constants", NAMED_CONSTANT_SETS)):
            if isinstance(updates.get(key), str):
                if updates[key] not in named:
                    raise ValueError(f"Unknown named {key} {updates[key]!r}; choose from {sorted(named)}.")
                updates[key] = named[updates[key]]
        population = updates.get("population_size", base_cfg.gp.population_size)
        if "tournament_fraction" in params:
            fraction = params["tournament_fraction"]
            if not 0 < fraction <= 1:
                raise ValueError("tournament_fraction must be in (0, 1].")
            updates["tournament_size"] = max(2, int(population * fraction))
        if "p_reproduction" in params:
            reproduction, mutation = params["p_reproduction"], params["p_mutation_total"]
            subtree, hoist, point = mutation_shares(params["mutation_u1"], params["mutation_u2"])
            updates.update(p_crossover=1 - reproduction - mutation, p_subtree_mutation=subtree * mutation,
                           p_hoist_mutation=hoist * mutation, p_point_mutation=point * mutation)
        gp = replace(base_cfg.gp, **updates)
        probabilities = [getattr(gp, name) for name in GP_OPERATOR_FIELDS]
        if min(probabilities) < 0 or sum(probabilities) > 1 + 1e-12:
            raise ValueError(f"GP operator probabilities must be nonnegative with sum <= 1: {probabilities}.")
        if not 1 <= gp.tournament_size <= gp.population_size:
            raise ValueError("tournament_size must be between 1 and population_size.")
        cfg = replace(base_cfg, gp=gp, **{key: params[key] for key in GP_TRAIN_FIELDS if key in params})
    else:
        net_updates = {key: params[key] for key in NN_NET_FIELDS if key in params}
        if "n_layers" in params:
            if params["n_layers"] < 1 or params["width"] < 1:
                raise ValueError("n_layers and width must be positive.")
            net_updates["hidden_dims"] = [params["width"]] * params["n_layers"]
        if net_updates and base_cfg.net_config is None:
            raise ValueError("The base NN config has no net_config to update.")
        net = replace(base_cfg.net_config, **net_updates) if net_updates else base_cfg.net_config
        cfg = replace(base_cfg, net_config=net, **{key: params[key] for key in NN_FIELDS if key in params})
        if cfg.net_config is None or cfg.net_config.input_dim != cfg.data_spec.n_features:
            raise ValueError("net_config.input_dim must match data_spec.n_features.")
    validate_training_mode(cfg, supports_boundary=True)
    return cfg


def load_base_config(family, path):
    if family == "gp":
        from .models.gp.config import TrainGPConfig as config_cls
    elif family == "nn":
        from .models.nn.config import TrainDEBBirthNetConfig as config_cls
    else:
        raise ValueError(f"Unknown family {family!r}; expected {FAMILIES}.")
    return config_cls.load_json(path)


def changed_fields(base, new, prefix=""):
    """Dotted config fields whose values differ between two config_dict outputs."""
    changes = {}
    for key, value in new.items():
        old = base.get(key)
        if isinstance(value, dict) and isinstance(old, dict):
            changes.update(changed_fields(old, value, f"{prefix}{key}."))
        elif value != old:
            changes[f"{prefix}{key}"] = value
    return changes


def describe_space(space):
    """Readable, JSON-serializable description of fixed entries and Ray domains."""
    from ray.tune.search.sample import Categorical, Domain
    described = {}
    for key, value in space.items():
        if not isinstance(value, Domain):
            described[key] = {"fixed": value}
            continue
        entry = {"domain": type(value).__name__}
        sampler = value.get_sampler()
        if hasattr(sampler, "q"):
            entry["q"] = sampler.q
            sampler = sampler.sampler
        entry["sampler"] = type(sampler).__name__.lstrip("_")
        if hasattr(sampler, "base"):
            entry["log_base"] = sampler.base
        if isinstance(value, Categorical):
            entry["categories"] = list(value.categories)
        else:
            entry["lower"], entry["upper"] = value.lower, value.upper
        described[key] = entry
    return described


def _file_identity(path):
    if path is None:
        return None
    path = Path(path).resolve()
    return {"path": portable_path(path), "sha256": sha256(path.read_bytes()).hexdigest()}


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _trial(config, *, family, prepared, data_metadata, search):
    """Ray trainable: train, save and validate one trial as a normal run.

    The base config is reloaded from its JSON snapshot rather than pickled:
    unpickled GP primitives are copies that the registry cannot identify.
    """
    from ray import tune
    trial_id = tune.get_context().get_trial_id()
    base = load_base_config(family, search["base_config"])
    cfg = resolve_trial(base, family, config)
    cfg = replace(cfg, outdir=Path(search["runs_dir"]) / trial_id)
    params = {key: _plain(key, value) for key, value in config.items()}
    metadata = {**data_metadata, "tuning": {"search": search["name"], "search_dir": search["search_dir"],
                                            "trial_id": trial_id, "params": params}}
    if family == "gp":
        from .models.gp.train import train_gp_classifier
        output = train_gp_classifier(cfg, save_run=True, prepared=prepared, data_metadata=metadata)
    else:
        from .models.nn.train import train_net
        output = train_net(cfg, save=True, prepared=prepared, data_metadata=metadata)
    outdir = Path(output["outdir"])
    _write_json(outdir / "trial.json", {
        "search": search["name"], "search_dir": search["search_dir"], "trial_id": trial_id, "params": params,
        "changed_fields": changed_fields(config_dict(base), config_dict(output["train_config"]))})
    report = {**asdict(output["val_metrics"]), "run_dir": portable_path(outdir)}
    if family == "nn":
        report["selected_epoch"] = output["selected_epoch"]
    tune.report(report)


def run_search(*, family, base_config, space, name, num_samples, metric="f1_macro", mode="max", search_seed=0,
               max_concurrent=None, cpus_per_trial=None, points_to_evaluate=None, script=None,
               tune_root="results/tune"):
    """Run a HyperOpt search over `space` starting from the base experiment JSON.

    Every trial is saved as a complete run; the best completed trial by the
    validation `metric` (first in trial order on ties) is recorded in best.json.
    No test evaluation and no retraining are performed.
    """
    from importlib.metadata import version

    from ray.tune.search.sample import Domain
    from .data.load import load_prepared_splits
    from .evaluate.metrics import BinaryMetrics
    from .utils.results import save_run_metadata

    if mode not in ("max", "min"):
        raise ValueError("mode must be 'max' or 'min'.")
    if metric not in {field.name for field in fields(BinaryMetrics)}:
        raise ValueError(f"Unknown objective metric {metric!r}; use a BinaryMetrics field.")
    if num_samples < 1:
        raise ValueError("num_samples must be positive.")
    base_path = resolve_repo_path(base_config)
    base = load_base_config(family, base_path)

    # Fail fast: check keys, then resolve one sampled trial and each starting point.
    check_keys(family, space)
    fixed = {key: value for key, value in space.items() if not isinstance(value, Domain)}
    sample = {key: value.sample(random_state=search_seed) if isinstance(value, Domain) else value
              for key, value in space.items()}
    resolve_trial(base, family, sample)
    for point in points_to_evaluate or ():
        if set(point) - set(space):
            raise ValueError(f"points_to_evaluate keys {sorted(set(point) - set(space))} are not in the space.")
        resolve_trial(base, family, {**fixed, **point})

    if cpus_per_trial is None:
        workers = base.num_workers
        cpus_per_trial = (workers if workers > 0 else os.cpu_count()) if family == "gp" else 1 + workers
    search_dir = resolve_repo_path(tune_root) / f"{datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}_{name}"
    runs_dir = search_dir / "runs"
    runs_dir.mkdir(parents=True)

    # Load data once; trials get the same row-aligned train/val records and never the test split.
    splits, data_metadata = load_prepared_splits(base.data_dir, base.data_spec, base.data_splits)
    prepared = {split: splits[split] for split in ("train", "val")}
    data_metadata = {**data_metadata, "rows": {split: len(prepared[split].labels) for split in prepared},
                     "tuning_splits": "train and val only; test not passed to trials"}
    save_run_metadata(search_dir, base, data_metadata)
    base.save_json(search_dir / "base_config.json")
    formulation = base.data_spec.formulation
    _write_json(search_dir / "search.json", {
        "name": name, "family": family, "formulation": formulation,
        "base_config": {**_file_identity(base_path), "resolved": config_dict(base)},
        "script": _file_identity(script),
        "space": describe_space(space), "points_to_evaluate": list(points_to_evaluate or ()),
        "objective": {"metric": metric, "mode": mode, "split": "val",
                      "ties": "first completed trial in trial-id order"},
        "budget": {"num_samples": num_samples, "max_concurrent": max_concurrent, "cpus_per_trial": cpus_per_trial},
        "search_algorithm": {"name": "HyperOptSearch (TPE)", "random_state_seed": search_seed,
                             "note": "with concurrent trials the proposal sequence depends on completion order"},
        "trial_seed": base.seed, "selection": "selected trial is the saved run itself; no retraining",
        "versions": {package: version(package) for package in ("ray", "hyperopt")},
        "created": datetime.now().isoformat(timespec="seconds"),
    })

    import ray
    from ray import tune
    from ray.tune.search import ConcurrencyLimiter
    from ray.tune.search.hyperopt import HyperOptSearch

    started = not ray.is_initialized()
    if started:
        # Workers must import src.debbirth from the repository root.
        ray.init(runtime_env={"env_vars": {"PYTHONPATH": str(REPO_ROOT)}}, include_dashboard=False)
    try:
        algorithm = HyperOptSearch(metric=metric, mode=mode, points_to_evaluate=points_to_evaluate,
                                   random_state_seed=search_seed)
        if max_concurrent is not None:
            algorithm = ConcurrencyLimiter(algorithm, max_concurrent=max_concurrent)
        search_info = {"name": name, "search_dir": portable_path(search_dir), "runs_dir": str(runs_dir),
                       "base_config": str(search_dir / "base_config.json")}
        trainable = tune.with_resources(
            tune.with_parameters(_trial, family=family, prepared=prepared,
                                 data_metadata=data_metadata, search=search_info),
            {"cpu": cpus_per_trial})
        tuner = tune.Tuner(
            trainable, param_space=space,
            tune_config=tune.TuneConfig(search_alg=algorithm, num_samples=num_samples,
                                        trial_dirname_creator=lambda trial: trial.trial_id),
            run_config=tune.RunConfig(name="ray", storage_path=str(search_dir), log_to_file=True, verbose=1))
        results = tuner.fit()
    finally:
        if started:
            ray.shutdown()
    return summarize_search(search_dir, family, formulation, space, results, metric, mode)


def summarize_search(search_dir, family, formulation, space, results, metric, mode):
    """Write trials.csv and best.json from a Ray ResultGrid; return the summary."""
    rows = []
    for result in results:
        metrics = result.metrics or {}
        rows.append({
            "trial_id": Path(result.path).name,
            "status": "ERROR" if result.error else "TERMINATED",
            "error": str(result.error).strip().splitlines()[-1] if result.error else "",
            **{key: _plain(key, result.config.get(key)) for key in space},
            **{key: metrics.get(key) for key in SUMMARY_METRICS + (("selected_epoch",) if family == "nn" else ())},
            "run_dir": metrics.get("run_dir", ""),
        })
    rows.sort(key=lambda row: row["trial_id"])
    if rows:
        with (search_dir / "trials.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def score(row):
        value = next(r for r in results if Path(r.path).name == row["trial_id"]).metrics.get(metric)
        return value if isinstance(value, (int, float)) and math.isfinite(value) else None

    scored = [(row, score(row)) for row in rows if row["status"] == "TERMINATED"]
    scored = [(row, value) for row, value in scored if value is not None]
    best = None
    for row, value in scored:  # strict comparison keeps the first trial on ties
        if best is None or (value > best[1] if mode == "max" else value < best[1]):
            best = (row, value)
    summary = {"search_dir": portable_path(search_dir), "trials": rows, "best": None}
    if best is None:
        _write_json(search_dir / "best.json", {"best": None, "reason": "no completed trial with a finite objective"})
        print(f"No completed trial; see {portable_path(search_dir)}/trials.csv")
        return summary
    row, value = best
    saved = f"{row['run_dir']}/{SAVED_CONFIG[family]}"
    record = {
        "trial_id": row["trial_id"], "objective": {"metric": metric, "mode": mode, "value": value, "split": "val"},
        "params": {key: row[key] for key in space}, "metrics": {key: row[key] for key in SUMMARY_METRICS},
        "run_dir": row["run_dir"], "saved_config": saved,
        "reproduce": f"python -m src.debbirth.train --model {family} --formulation {formulation} --config {saved}",
        "completed_trials": len(scored), "total_trials": len(rows),
    }
    _write_json(search_dir / "best.json", record)
    summary["best"] = record
    print(f"\nSearch: {portable_path(search_dir)}  ({len(scored)}/{len(rows)} trials completed)")
    print(f"Best trial {row['trial_id']}: val {metric}={value:.4f}  run: {row['run_dir']}")
    print(f"Reproduce: {record['reproduce']}")
    return summary
