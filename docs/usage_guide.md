# Usage guide: training, loading and using the models

This guide explains how the code is organised after T03–T06C and T08A and walks through each of the six model combinations step by step. It describes the code as of 2026-10-02. For derivations see `birth_equations.md`; for interface details and validation records see `formulations_and_data.md`, `nn_formulations.md` and `gp_formulations.md`.

All commands run from the repository root in the `debbirth` conda environment:

```text
conda run -n debbirth python -m <module>
```

Python snippets import from `src.debbirth...`, so run them from the repository root (or put the root on `sys.path` in notebooks).

## 1. What changed

### Before (CONTROLO'26 code)

- There was one formulation: four inputs `(g, k, v_Hb, f)`.
- The GP and NN trainers each read the CSVs and built their own arrays/tensors.
- Both trainers used class balancing (GP `class_weights="balanced"`, NN `use_pos_weight=True`).
- NN inference needed the caller to apply the saved log/standardization scaler by hand, and dropout had to be switched off explicitly (`model.eval()`).
- GP and NN had different prediction calls, and evaluation ran each model twice (once for labels, once for probabilities).
- Plots assumed original-variable axes and read `k` and `f` from the first row of the plotted data.

### Now

```text
data/processed/{train,val,test}.csv
        │   load_splits / load_prepared_splits                    data/load.py
        ▼
PreparedSplit  (one per split, rows kept aligned)                 data/prepare.py
  features (formulation order) · labels · source_rows · diagnostics · log_nu_b (boundary only)
        │
        ├──────────────────────────────┬──────────────────────────────┐
        ▼                              ▼                              │
 GP: train_gp_classifier          NN: train_net                       │
  full_par / normalized:            all three formulations            │
   DEBBirthSymbolicClassifier         DEBBirthNet (score)             │
  boundary:                           DEBBirthBoundaryNet (F)         │
   fit_gp_boundary → GPBoundaryModel                                  │
        │                              │                              │
        ▼                              ▼                              │
 results/runs/<timestamp>_<name>/   (config, metadata, metrics, model, history)
        │  load_gp_run                 │  load_trained_nn
        ▼                              ▼
   GPPredictor  ◄── same interface ──► NNPredictor        (ParameterPredictor base)
        │   predict_details · predict · predict_proba · critical_maturity · evaluate_prepared
        ▼
 metrics (evaluate/predict.py) and plots (plot/boundary.py)
```

| Concern | Where | Notes |
| --- | --- | --- |
| Formulation names and output semantics | `src/debbirth/formulations.py` | `full_par`, `normalized`, `boundary`; margin, logit, strict decision |
| Feature schema | `src/debbirth/data/schema.py` (`DatasetSpec`) | Formulation, optional `x_b`, feature order |
| Shared preparation | `src/debbirth/data/prepare.py`, `data/load.py` | Normalization, `log_nu_b`, `PreparedSplit` |
| GP | `src/debbirth/models/gp/` | `config.py`, `train.py`, `boundary.py`, `predict.py`, `expression.py` |
| NN | `src/debbirth/models/nn/` | `config.py`, `train.py`, `structure.py`, `predict.py` |
| Shared inference interface | `src/debbirth/evaluate/predictor.py` | `ParameterPredictor` |
| Metrics | `src/debbirth/evaluate/predict.py`, `metrics.py` | `metrics_from_predictions`, `BinaryMetrics` (now with MCC, log loss, Brier) |
| Plots | `src/debbirth/plot/boundary.py` | Original or normalized axes, critical curves and surfaces |
| Experiment settings | `experiments/*.json` | One JSON per model; validation scripts alongside |
| Tuning | `src/debbirth/tuning.py`, `experiments/tune_{gp,nn}.py` | Ray Tune + HyperOpt from a base JSON; every trial a saved run (T08B) |
| Training CLI | `src/debbirth/train.py` | `python -m src.debbirth.train --model … --formulation …` (T08A) |

The main behavioural changes:

1. **Shared preparation.** Normalization happens once, in shared preparation. Each formulation is a `DatasetSpec`; never compute `gamma` or `nu_b` by hand for the models.
2. **Boundary models.** They exist for both families. They learn `F = log(Psi)`, and maturity enters only through the fixed offset `log_nu_b`.
3. **Unweighted defaults.** Losses are unweighted by default (T06C). The old weighted settings are kept in `*_balanced.json` / `*_pos_weight.json`.
4. **Revised GP function set.** The new GP models use `add, sub, mul, pdiv, plog, min, cbrt, square, cube, atan` and include `x_b` (T06B).
5. **Predictors.** Every saved model loads as a predictor that takes original parameters `(g, k, v_Hb, f)`; scaling and normalization are internal.
6. **One-pass evaluation.** Each validation set is evaluated with a single model execution.
7. **Explicit plots.** They know their coordinate system and check slice assumptions.

## 2. Concepts you need

| Formulation | Model input (order) | Learned output | Probability | Canonical decision |
| --- | --- | --- | --- | --- |
| `full_par` | `g, k, v_Hb, f` | score S | `sigmoid(S)` | GP: `p > 0.5`; NN: `p >= threshold` (0.5) |
| `normalized` | `gamma, k, nu_b[, x_b]` | score S | `sigmoid(S)` | as above |
| `boundary` | `gamma, k[, x_b]` | `F = log(Psi)` | `sigmoid((F - log nu_b)/T)` | strict `F - log nu_b > 0` |

- **Derived inputs.** `gamma = g/f`, `nu_b = v_Hb/f^3`, `log_nu_b = log(v_Hb) - 3 log(f)`, and `x_b = gamma/(1+gamma)` (optional, `include_x_b=True`). Natural logarithms throughout.
- **Boundary in original variables.** Birth is feasible iff `v_Hb < f^3 * exp(F(g/f, k))`. Zero margin is infeasible, with no tolerance.
- **Temperature.** `T > 0` (`boundary_temperature`) sharpens or softens the probability but never moves the zero-margin decision. It is fixed during training; tuning it is T08.
- **Operating threshold.** `probability_threshold=t` is an explicit alternative decision, `p >= t`, reported separately from the canonical rule.
- **Selection policy.** GP keeps gplearn's last-generation lowest-raw-loss program; NN uses `checkpoint_selection` (`final_epoch` or `best_val_loss`).
- **Target.** `reached_birth`, with get_lb2 timeouts and failures retained as negatives.

## 3. Configurations

| File | Family / formulation | Notes |
| --- | --- | --- |
| `experiments/gp_full_par.json` | GP / full_par | Historical function set, unweighted |
| `experiments/gp_full_par_balanced.json` | GP / full_par | Previous balanced version, for comparisons |
| `experiments/gp_normalized.json` | GP / normalized | Revised set, `x_b`, unweighted |
| `experiments/gp_boundary.json` | GP / boundary | Revised set, `x_b`, T = 1, unweighted |
| `experiments/nn_full_par.json` | NN / full_par | Unweighted |
| `experiments/nn_normalized.json`, `nn_boundary.json` | NN / new formulations | Unweighted, `best_val_loss` checkpoints |
| `experiments/nn_*_pos_weight.json` | NN | Previous weighted versions |

The evolution and training settings are untuned examples, not final experiment settings (T08/T09). Load a config and change settings with `dataclasses.replace`; configs are immutable records and loading creates no directories:

```python
from dataclasses import replace
from src.debbirth.models.gp.config import TrainGPConfig
from src.debbirth.models.nn.config import TrainDEBBirthNetConfig

gp_cfg = TrainGPConfig.load_json("experiments/gp_boundary.json")
gp_cfg = replace(gp_cfg, gp=replace(gp_cfg.gp, generations=20), boundary_temperature=0.7, num_workers=4)

nn_cfg = TrainDEBBirthNetConfig.load_json("experiments/nn_normalized.json")
nn_cfg = replace(nn_cfg, epochs=20, device="cpu")
```

Rules enforced when a config is built:

- `boundary_temperature` other than 1 is only valid for `boundary`.
- GP `class_weights` must be `None`, `"balanced"` or `{0: w0, 1: w1}`.
- NN `pos_weight` requires `use_pos_weight=True`.
- NN `net_config.input_dim` must equal the number of features. For example, normalized with `x_b` has 4 inputs and boundary without `x_b` has 2.
- GP uses unscaled inputs (`scaling_type="none"`).

To add `x_b` to an NN, or remove it from a GP, change the data spec. Remember to change `input_dim` for NN:

```python
nn_cfg = replace(nn_cfg, data_spec=replace(nn_cfg.data_spec, include_x_b=True),
                 net_config=replace(nn_cfg.net_config, input_dim=4))
```

## 4. Common step: data for a run

By default both trainers load `data/processed/{train,val,test}.csv` themselves (`data_splits="train_val_test"`, the 80/10/10 split). Test rows are loaded but never used during training.

For quick checks or sample-efficiency subsets, pass explicit, row-aligned `PreparedSplit` records instead. Select whole records with `.select(...)`, never labels or features separately, so that labels, offsets and source rows stay together:

```python
import numpy as np
from src.debbirth.data.load import load_splits
from src.debbirth.data.prepare import prepare_split

frames = load_splits("data/processed")            # train/val/test DataFrames with source-row columns
rng = np.random.default_rng(42)
rows = {"train": rng.choice(len(frames["train"]), 2000, replace=False),
        "val": rng.choice(len(frames["val"]), 500, replace=False)}

def subset(spec):
    """Matched train/val subset for any formulation (same rows for every model)."""
    return {name: prepare_split(frames[name], spec).select(rows[name]) for name in ("train", "val")}
```

Use the same `rows` for every model you compare. Pass `data_metadata={...}` to the trainer if you want the subset's provenance recorded in `run_metadata.json`.

## 5. Step by step for each model

Each training call below uses `prepared=subset(...)` and small settings so that it finishes in seconds. For a real run, drop `prepared=` (the full splits are then loaded) and the size overrides.

**From the command line.** Any of the six models can be trained on the full configured splits with the shared CLI (T08A). It calls the same trainers:

```text
conda run -n debbirth python -m src.debbirth.train --model gp --formulation boundary --seed 42
conda run -n debbirth python -m src.debbirth.train --model nn --formulation normalized --device cpu --run-name nn_norm
```

- **Config.** `--config` defaults to `experiments/<model>_<formulation>.json`.
- **Overrides.** Only runtime and identity settings: `--seed`, `--data-dir`, `--outdir`, `--run-name`, `--num-workers`, and `--device` (NN only). Hyperparameters, including temperature, stay in the config.
- **Dry run.** `--dry-run` validates and prints the resolved config without loading data.
- **No save.** `--no-save` trains and validates in memory, printing progress and the summary. It creates no directory or file, so the run cannot be reloaded later. It cannot be combined with `--outdir` or `--run-name`. From Python, the equivalent is `train_gp_classifier(cfg, save_run=False)` or `train_net(cfg, save=False)`.
- **Output directory.** A config's `outdir` is ignored; each run gets a new directory unless `--outdir` names a new or empty one.
- **Reproducing a run.** Use `--config results/runs/<run>/train_*_config.json`. `cli_invocation.json` in each CLI run records the exact command.
- **Not available from the CLI.** Prepared subsets and test evaluation stay in Python.

### 5.1 GP, full-parameter (historical benchmark)

1. **Train with the module entry point** (full data, settings from `gp_full_par.json`, saves a run):

   ```text
   conda run -n debbirth python -m src.debbirth.models.gp.train
   ```

2. **Or train from Python:**

   ```python
   from dataclasses import replace
   from src.debbirth.models.gp.config import TrainGPConfig
   from src.debbirth.models.gp.train import train_gp_classifier

   cfg = TrainGPConfig.load_json("experiments/gp_full_par.json")
   cfg = replace(cfg, gp=replace(cfg.gp, population_size=100, generations=3, tournament_size=10), num_workers=1)
   out = train_gp_classifier(cfg, save_run=True, prepared=subset(cfg.data_spec))
   print(out["outdir"], out["val_metrics"].f1_macro, out["best_program"])
   ```

3. **Inspect the run directory** `results/runs/<timestamp>_DEBBirthSymbolicClassifier/`:
   - `train_gp_config.json`, `run_metadata.json` (data hashes, loss, decision rule, selection policy, versions, code hashes), and `cli_invocation.json` when launched from the CLI;
   - `metrics/val_metrics.json`, `history.csv`, `progress.csv` (per generation, written during training);
   - `model/gp_model.joblib`, `model/best_program.txt`, `model/expression.txt`.

4. **Load and predict** (section 6). As with NN, `out["predictor"]` is ready to use after training, including unsaved runs; `load_gp_run(out["outdir"])["predictor"]` gives the same predictions after reloading.

To reproduce the historical balanced setting, use `gp_full_par_balanced.json`. The archived paper model is in `results/models/DEBBirthGP` (section 7).

### 5.2 GP, normalized

1. **Load the config.** `experiments/gp_normalized.json` uses features `gamma, k, nu_b, x_b`, the revised function set and unweighted log loss.
2. **Train:**

   ```python
   cfg = TrainGPConfig.load_json("experiments/gp_normalized.json")
   cfg = replace(cfg, gp=replace(cfg.gp, population_size=100, generations=3, tournament_size=10), num_workers=1)
   out = train_gp_classifier(cfg, save_run=True, prepared=subset(cfg.data_spec))
   ```

3. **Read the expression.** `model/expression.txt` gives `S(gamma, k, nu_b, x_b) = ...` in exact gplearn syntax, the input definitions in original variables, constant values and primitive definitions (e.g. `pdiv`, `plog` protection).
4. **Load and predict from original parameters** (section 6); normalization is internal.

### 5.3 GP, boundary

1. **Load the config.** `experiments/gp_boundary.json` evolves `F(gamma, k, x_b)` and carries `log_nu_b` separately. The config must keep `metric="log loss"`, `transformer="sigmoid"` and a numeric parsimony coefficient.
2. **Train.** The same trainer dispatches to `fit_gp_boundary`:

   ```python
   cfg = TrainGPConfig.load_json("experiments/gp_boundary.json")
   cfg = replace(cfg, gp=replace(cfg.gp, population_size=100, generations=3, tournament_size=10),
                 num_workers=1, boundary_temperature=1.0)
   out = train_gp_classifier(cfg, save_run=True, prepared=subset(cfg.data_spec))
   model = out["model"]          # GPBoundaryModel: program, feature order, constants, T
   print(model.expression)       # F in gplearn syntax
   ```

   `out["engine"]` is the gplearn engine with its dataset-bound loss. Use it only to inspect that fit (e.g. `engine.run_details_`); it is not saved.
3. **Read the rule.** `model/expression.txt` states `F = ...`, `Psi = exp(F)`, and the rule `log(v_Hb) - 3 log(f) < F`, i.e. `v_Hb < f^3 exp(F)`.
4. **Load and predict** (section 6). Boundary predictors additionally give the margin and the critical maturity.
5. **Low-level use (prepared inputs only).** `GPBoundaryModel.predict_boundary(X)`, `predict_margin(X, log_nu_b)`, `predict_proba(X, log_nu_b)` and `predict(X, log_nu_b)` take unscaled prepared features. Prefer the predictor.

### 5.4 NN, full-parameter (historical benchmark)

1. **Train with the module entry point** (full data, `nn_full_par.json`):

   ```text
   conda run -n debbirth python -m src.debbirth.models.nn.train
   ```

2. **Or train from Python:**

   ```python
   from dataclasses import replace
   from src.debbirth.models.nn.config import TrainDEBBirthNetConfig
   from src.debbirth.models.nn.train import train_net

   cfg = TrainDEBBirthNetConfig.load_json("experiments/nn_full_par.json")
   cfg = replace(cfg, epochs=2, device="cpu", num_workers=0)
   out = train_net(cfg, save=True, prepared=subset(cfg.data_spec))
   predictor = out["predictor"]
   print(out["outdir"], out["selected_epoch"], out["val_metrics"].f1_macro)
   ```

3. **Inspect the run directory** `results/runs/<timestamp>_<run_name>/`. The default name is `DEBBirthNet`, or `DEBBirthBoundaryNet` for boundary runs; set `run_name` in the config to change it. The directory contains:
   - `train_nn_config.json`, `run_metadata.json`, `checkpoint.json` (selection rule and epoch), and `cli_invocation.json` when launched from the CLI;
   - `history.csv` (per epoch), `progress.csv` (per epoch, written during training), `metrics/val_metrics.json` (selected epoch);
   - `model/model_state_dict.pth`, `model/scaler.pth` (absent when `scaling_type="none"`).
4. **Load and predict** (section 6). The saved scaler is applied automatically.

### 5.5 NN, normalized

1. **Load the config.** `experiments/nn_normalized.json` uses `gamma, k, nu_b`, 3 inputs, log-standardization fitted on training rows, `best_val_loss` checkpoints and unweighted BCE.
2. **Train:**

   ```python
   cfg = TrainDEBBirthNetConfig.load_json("experiments/nn_normalized.json")
   cfg = replace(cfg, epochs=2, device="cpu")
   out = train_net(cfg, save=True, prepared=subset(cfg.data_spec))
   ```

3. **Load and predict** (section 6).

### 5.6 NN, boundary

1. **Load the config.** `experiments/nn_boundary.json` uses `gamma, k` (2 inputs). The network outputs unrestricted `F`; the trainer computes `(F - log_nu_b)/T` outside the network. `net_config.threshold` must stay 0.5, because the decision is the strict margin.
2. **Train:**

   ```python
   cfg = TrainDEBBirthNetConfig.load_json("experiments/nn_boundary.json")
   cfg = replace(cfg, epochs=2, device="cpu", boundary_temperature=0.7)
   out = train_net(cfg, save=True, prepared=subset(cfg.data_spec))
   ```

3. **Load and predict** (section 6), including margins and critical maturity.

### 5.7 Validation progress during training

Every GP generation and NN epoch is scored on the validation split while the run is training (T08D). Both families print the same aligned table:

```text
GP boundary | 5 generations | population 150 | seed 7
progress: results/runs/2026-10-03T18-25-16-108724_t08d_gp_boundary/progress.csv
gen |   train      val  | f1_macro      mcc   f1_pos   f1_neg  |  len  |    time     eta
----+-------------------+--------------------------------------+-------+----------------
  1 |  0.2989   0.3013  |   0.8473   0.6949   0.8214   0.8731  |   14  |    0:06    0:22
  2 |  0.2829   0.2857* |   0.8631*  0.7271   0.8449   0.8813  |   19  |    0:08    0:12
  3 |  0.2793   0.2799* |   0.8684*  0.7369   0.8473   0.8895  |   13  |    0:10    0:06
  4 |  0.2793   0.2799  |   0.8684   0.7369   0.8473   0.8895  |   13  |    0:12    0:03
  5 |  0.2790   0.2798* |   0.8684   0.7369   0.8475   0.8893  |   10  |    0:15    0:00
best val 0.2798 at gen 5 | best f1_macro 0.8684 at gen 3 | 0:15 total  (* = new best)
```

- **Banner.** Model family, formulation, number of steps, seed and the `progress.csv` path.
- **Columns.**
  - `train` is the training loss. For GP it is the generation's best raw training loss (gplearn's `best_fitness`). For NN it is the running minibatch mean during the epoch, with dropout on.
  - `val` is `val_bce` (below).
  - GP adds `len`, the best program's length.
  - `time` is the elapsed time and `eta` the estimated time remaining.
- **Markers and repeats.** `*` marks a new best validation loss or macro-F1 at the printed precision. The header repeats every 25 rows.
- **Summary.** When training finishes, a final line gives the best values and the step where each occurred.

Saved runs also write `progress.csv` in the run directory. With saving off (`--no-save`, `save_run=False`, `save=False`), progress is printed only and no file is written.

- **Writing.** The header is written at the start, then one row is appended and flushed per step. The file can be followed while the run trains, and an interrupted run keeps its completed rows. The run directory is created before training for this reason, so an interrupted run leaves a directory with only `progress.csv`.
- **Shared columns.**
  - `step_kind` (`generation`/`epoch`) and `step`, the number of completed generations or epochs. For GP this is gplearn's generation index + 1, so the last step equals `generations`.
  - `elapsed_s` (since training began) and `val_eval_s` (time to score this step).
  - `train_loss`, as in the printed `train` column. It is the loss each family optimizes, so it is weighted when class weighting is enabled. The GP value is a full-pass loss of the best program and the NN value is an in-epoch running mean, so the two are not strictly comparable.
  - `val_bce`: the unweighted mean BCE of the classification logit on validation, computed in float64 for every formulation. For boundary models the logit is `(F - log nu_b)/T`. It stays unweighted when class weighting is enabled.
  - Every `BinaryMetrics` field: macro-F1, per-class precision/recall/F1, MCC, AUROC/AP, confusion counts, log loss and Brier score.
- **GP columns.** `best_length` and the population `average_length`. Each GP row scores the generation's lowest raw-loss program, which is the program the run would return if it stopped at that step. Logging only reads the population, so the final program is identical with logging on or off. gplearn's own `verbose` table is not printed while progress logging is on.
- **NN column.** `val_loss`, the training loss function on validation. It is weighted when `use_pos_weight` is on, and `best_val_loss` uses it. With `best_val_loss`, the saved metrics correspond to the selected epoch's row, not necessarily the last.
- **Frequency.** `progress_every` in the training config (GP and NN, default 1) logs every n-th step and always the last; `0` disables progress logging. For GP, `progress_every=0` restores gplearn's `verbose` table. Unsaved runs print without writing a file. Tuning trials (section 5.8) keep their own `progress.csv`; their printed output goes to Ray's per-trial log files.

`history.csv` is unchanged and still written at the end of the run: gplearn's `run_details_` for GP and `EpochBinaryMetrics` for NN.

### 5.8 Hyperparameter tuning (T08B)

A search starts from an ordinary experiment JSON (the base config) and a search space. The space is a dict of fixed values and Ray Tune domains, defined in a small script:

```text
conda run -n debbirth python -m experiments.tune_gp --formulation boundary --num-samples 50 [--config base.json] [--max-concurrent N]
conda run -n debbirth python -m experiments.tune_nn --formulation normalized --num-samples 50
```

From Python, call `run_search(family=..., base_config=..., space=..., name=..., num_samples=...)` in `src/debbirth/tuning.py`. `resolve_trial(base_cfg, family, params)` shows how one sampled point becomes a training config, without running anything.

| Family | Direct keys (config fields) | Derived keys |
| --- | --- | --- |
| GP | `population_size`, `generations`, `tournament_size`, `parsimony_coefficient`, `init_depth`, `init_method`, `p_crossover`, `p_*_mutation`, `p_point_replace`, `function_set`, `constants`, `boundary_temperature`, `class_weights`, `progress_every`, `verbose`, `num_workers` | `tournament_fraction` gives `tournament_size = max(2, int(pop * fraction))`; `p_reproduction`, `p_mutation_total`, `mutation_u1`, `mutation_u2` (together) give `p_crossover = 1 - p_reproduction - p_mutation_total` and Dirichlet mutation shares |
| NN | `epochs`, `batch_size`, `lr`, `weight_decay`, `scaling_type`, `checkpoint_selection`, `boundary_temperature`, `use_pos_weight`, `progress_every`, `num_workers`, `device`, `hidden_dims`, `dropout` | `n_layers`, `width` (together) give `hidden_dims = [width] * n_layers` |

- **Named sets.** `function_set` and `constants` accept a registered set name (`NAMED_FUNCTION_SETS`; `NAMED_CONSTANT_SETS`: `none`, `default`, `extended`, `extended_c0`) or a list of registered names.
- **Rejected keys.** Unknown keys, a derived key combined with the field it determines, incomplete groups, and tuner-managed keys (`seed`, `data_spec`, `data_dir`, `data_splits`, `outdir`, `run_name`) are rejected. All trials share the base seed; repeated seeds are T09.
- **Weighting.** The tuner never adds weighting. An explicit `class_weights` or `use_pos_weight` in the base config is kept and recorded.
- **Fail-fast checks.** Before Ray starts, the space is checked and one sampled point (plus any `points_to_evaluate`) is resolved.
- **Data.** The data are loaded once, and each trial receives the same row-aligned train/val records. Test data never reach the tuner.
- **Output** in `results/tune/<timestamp>_<name>/`:
  - `search.json`: base config identity and resolved settings, script hash, space description, objective, budget, HyperOpt seed, Ray/HyperOpt versions;
  - `run_metadata.json`: code and data provenance;
  - `base_config.json`: trials reload the base from this snapshot by registered names;
  - `runs/<trial_id>/`: a complete run directory per trial, plus `trial.json` (sampled params and changed config fields);
  - `ray/`: Ray state and per-trial logs;
  - `trials.csv` and `best.json`.
- **Selection.** The objective is validation `f1_macro` (max) by default; any `BinaryMetrics` field can be used. On ties, the first trial in trial-id order wins. The selected trial is the final artifact: it is not retrained. `best.json` gives the command that reproduces it through the training CLI.
- **Reproducibility.** Each trial is reproducible from its saved config. With concurrent trials, HyperOpt's sequence of proposals depends on completion order, so the search itself is not exactly repeatable.
- **Temperature.** `boundary_temperature` can be searched as a training temperature: each candidate is fit on train and selected on validation. Whether to search it, and post-training calibration, are T08 decisions.

## 6. Loading a saved run and predicting

Both loaders return a predictor with the same methods:

```python
from src.debbirth.models.gp.train import load_gp_run
from src.debbirth.models.nn.train import load_trained_nn

gp = load_gp_run("results/runs/<gp run>")          # {"model", "train_cfg", "predictor"}
nn = load_trained_nn("results/runs/<nn run>", "cpu")  # {"model", "scaler", "train_cfg", "net_cfg", "predictor", "checkpoint"}
predictor = gp["predictor"]                          # or nn["predictor"]
```

Pass **original parameters**: a DataFrame with columns `g, k, v_Hb, f`, or an `(N, 4)` array in that order. Never pass `gamma`/`nu_b` or scaled values here.

```python
import pandas as pd

params = pd.DataFrame({"g": [0.5, 5.0], "k": [0.3, 3.0], "v_Hb": [1e-3, 0.5], "f": [0.8, 0.2]})
details = predictor.predict_details(params)   # dict of (N,) arrays
p = predictor.predict_proba(params)           # P(reached_birth = 1)
y = predictor.predict(params)                 # canonical decision
y_op = predictor.predict(params, probability_threshold=0.3)   # explicit operating point, p >= 0.3
```

| `details` key | Meaning |
| --- | --- |
| `learned_output` | S (score models) or F = log Psi (boundary) |
| `logit` | S, or `(F - log nu_b)/T` |
| `probability` | positive-class probability |
| `prediction` | canonical 0/1 decision |
| `margin` | boundary only: `F - log nu_b` |

Boundary models also give:

```python
predictor.predict_margin(params)                           # F - log nu_b
predictor.critical_maturity(params)                        # Psi(gamma, k)
predictor.critical_maturity(params, normalized=False)      # f^3 * Psi: critical v_Hb
predictor.critical_maturity(params, log=True)              # log Psi, safe for extreme values
```

`critical_maturity` needs a positive `v_Hb` column only to pass preparation; its value does not affect Psi.

If you already have an unscaled `PreparedSplit`, for example from `subset(...)`, call `predictor.predict_prepared(split)`. It skips normalization, so it must never receive manually scaled features.

## 7. Archived (paper) models

```python
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")     # the archived GP config predates reconstructable configs
    gp_archived = load_gp_run("results/models/DEBBirthGP")["predictor"]
nn_archived = load_trained_nn("results/models/DEBBirthNet", "cpu")["predictor"]
gp_archived.predict_proba(params), nn_archived.predict_proba(params)
```

Both are full-parameter models with inputs `(g, k, v_Hb, f)`. The archived GP's training config cannot be rebuilt (historical function-object strings), so its predictor uses the historical unscaled schema; the joblib model itself is unchanged.

## 8. Evaluating

Metrics are computed from labels, probabilities and an explicit decision; the model runs once.

```python
val = prepare_split(frames["val"], predictor.spec)          # unscaled, aligned
metrics = predictor.evaluate_prepared(val)                  # canonical decision
metrics_op = predictor.evaluate_prepared(val, probability_threshold=0.3)
print(metrics)                                              # per-class, macro, MCC, AUROC/AP, log loss, Brier
```

`BinaryMetrics` holds:

- per-class precision, recall and F1;
- macro averages and accuracy;
- confusion counts `tp, fp, tn, fn`;
- AUROC and average precision, for the positive class and macro-averaged;
- `mcc`, `log_loss` and `brier_score`.

`fp` counts infeasible sets accepted as feasible; `fn` counts feasible sets rejected. Use validation data for choices. Evaluate the test split (`frames["test"]`) only for a final, frozen comparison.

## 9. Plotting

```python
import numpy as np
from src.debbirth.plot.boundary import (slice_grid, plot_decision_mesh_with_models, draw_critical_curve,
                                        update_legend, plot_critical_surface)

# (a) Normalized slice at fixed k: axes gamma, nu_b (f-free). slice_grid adds g, v_Hb, f for the predictors.
gamma, nu_b = np.logspace(-3, 2, 150), np.logspace(-6, 2, 150)
df = slice_grid("gamma", gamma, "nu_b", nu_b, k=0.3)
df["score_model"] = score_predictor.predict(df[["g", "k", "v_Hb", "f"]])
fig, ax = plot_decision_mesh_with_models(df, ["score_model"], decision_col="score_model",
                                         x_col="gamma", y_col="nu_b", model_labels=["score model"])
curve = slice_grid("gamma", gamma, "nu_b", [1.0], k=0.3)
draw_critical_curve(ax, gamma, boundary_predictor.critical_maturity(curve), label=r"$\Psi$", linestyle=":")
update_legend(ax, loc="lower left", fontsize=7)

# (b) Original slice at fixed (k, f): axes g, v_Hb; the boundary curve is f^3 Psi.
df = slice_grid("g", np.logspace(-3, 2, 150), "v_Hb", np.logspace(-6, 1, 150), k=0.3, f=0.8)

# (c) Learned critical surface F(gamma, k), with the analytical k = 1 reference (F = 0).
surf = slice_grid("gamma", gamma, "k", np.logspace(-2, 1.5, 120), nu_b=1.0)
surf["log_psi"] = boundary_predictor.critical_maturity(surf[["g", "k", "v_Hb", "f"]], log=True)
fig, ax = plot_critical_surface(surf)
```

`plot_decision_mesh` and `plot_decision_mesh_with_models` draw the analytical screening lines in the plotted coordinate (`v_Hb = f^3/k`, ... or `nu_b = 1/k`, ...). They raise if `k` (and `f` for original axes) is not single-valued in the data; pass `reference_lines=False` for other plots. Existing notebook calls on single `(k, f)` grid files behave as before. After adding curves, call `update_legend` rather than `ax.legend()`, which would drop the region and contour entries.

## 10. Validation scripts

| Command | What it checks |
| --- | --- |
| `python -m experiments.validate_nn_formulations` | T05 NN training, checkpoints, reload, semantics |
| `python -m experiments.validate_gp_formulations` | T06 GP training, fitness audit, alignment, reload, expressions |
| `python -m experiments.validate_t06a` | T06A predictors, evaluation, plots for archived and saved models |
| `python -m experiments.validate_t06bc` | T06B/C function set and unweighted losses |

All use sampled train/validation rows, never test data, and write to `results/runs/`. `validate_t06a` reads specific earlier run directories, so edit its paths if those runs are not present.

## 11. Not available yet

- **Simplified exports.** `symbolic.py` simplified SymPy/MATLAB output drops the protected semantics of `pdiv`, `plog` and the real `cbrt` (T06D). Use `expression.txt` or the predictor.
- **Final settings.** Example configs, example search spaces and the small runs here are engineering checks; final settings, spaces, budgets and comparisons are T08/T09.
