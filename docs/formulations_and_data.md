# Formulations, preparation, and run configuration

Implemented for T03 on 2026-09-06. This is the shared foundation for T04-T06, not a completed new-model comparison or a proof of the critical surface.

## Mathematical and feature interfaces

`DatasetSpec` defaults to `formulation="full_par"`, `feature_set="dimensionless"`, and `target_col="reached_birth"`. Historical configs without a formulation retain these defaults. Other historical feature sets remain available with `full_par`.

| Formulation | Learned features, in order | Learned output | Final logit |
| --- | --- | --- | --- |
| `full_par` | `g, k, v_Hb, f` | Unconstrained score | Score |
| `normalized` | `gamma, k, nu_b` | Unconstrained score | Score |
| `boundary` | `gamma, k` | `F = log(Psi)` | `(F - log_nu_b) / T` |

Here `gamma = g/f`, `nu_b = v_Hb/f^3`, and `log_nu_b = log(v_Hb) - 3*log(f)`. The normalization code uses natural logs and calculates `nu_b` through exponentiation of `log_nu_b`, avoiding intermediate overflow of `f^3`. Boundary preparation never needs to exponentiate maturity. `include_x_b=True` appends `x_b = gamma/(1+gamma)` for normalized/boundary inputs; it is disabled by default and rejected for `full_par`.

`data.prepare.prepare_features(frame, spec)` accepts physical input columns and returns a feature DataFrame plus the boundary offset array (otherwise `None`). It does not apply learned scaling. Missing columns, nonnumeric/nonfinite inputs, and nonpositive physical parameters for normalized/boundary preparation are errors with column/row information. Finite nonpositive values remain available to historical nonlog full-parameter callers. Log preprocessing requires strictly positive features. Neither preparation nor scaling drops observations or clips physical inputs. Overflow/underflow that makes requested features unrepresentable raises an error, including conversion to NN float32.

`formulations.boundary_margin(F, log_nu_b)` requires equal shapes, preventing accidental cross-row broadcasting. `output_to_logit` applies the fixed offset and a finite positive scalar `temperature` only for boundary outputs. `output_to_probability(..., sigmoid=...)` accepts a stable backend operation such as `scipy.special.expit` or `torch.sigmoid`; Torch gradients are preserved. `boundary_is_feasible(margin)` uses strict `margin > 0`: equality is infeasible, with no numerical tolerance. Positive temperature does not change this decision. A separately tuned probability threshold is not implemented here.

Unlike the shorthand in `next_steps.md`, `F` is the log critical maturity, not the final classification logit; `Psi` is normalized maturity, and the threshold in physical coordinates is `f^3 * exp(F)`. The accepted PDF's Section II calls `k*v_Hb < f^3` sufficient; the displayed constraints instead imply a necessary screening condition. The PDF is unchanged. T07 remains responsible for completing the analytical proof; the preparation checks below do not establish it. The analytical `k=1` identity is not imposed on a learner by T03.

## Shared preparation and row identity

```python
from src.debbirth.data.schema import DatasetSpec
from src.debbirth.data.load import load_prepared_splits

spec = DatasetSpec(formulation="boundary", include_x_b=True)
splits, metadata = load_prepared_splits("data/processed", spec)
train = splits["train"]
subset = train.select([3, 0, 2])  # same selection for features, labels, offsets, and metadata
```

Each `PreparedSplit` contains `features`, `feature_names`, `labels`, `source_rows`, `diagnostics`, and optional `log_nu_b`. Its `select` method accepts positional indices, slices, or boolean masks. Prepare first, then select or permute this record to retain alignment.

`load_splits` reads CSVs with round-trip float parsing and adds reserved `__source_file` and `__source_row` columns in memory. Record positions are zero-based in the source CSV, before selection/merging. Repository-contained source paths are relative; external paths are absolute. Supplied CSVs containing those reserved names are rejected. SHA-256 hashes are stored in frame attributes and returned preparation metadata. In-memory preparation without source columns uses the supplied `source_name` and the frame's current record positions; it does not infer a raw-data mapping.

The loader supports `train_val_test` and `train_test`; the latter explicitly concatenates the supplied train and validation files while preserving their source identities. Unknown modes fail. Both trainers require `train_val_test` and reject other modes before data loading or output creation. No split generation, deduplication, or relabeling occurs. The current get_lb2 dataset retains negatives for failures/nonconvergence and its six-second timeouts, as required for practical screening.

The shared loading/preparation modules import neither Torch nor model training configurations. GP/NN adapters live in their respective model directories. Historical `data.load.load_data_gp` and `load_data_pytorch` imports remain lazy wrappers returning the existing two-item and five-item tuples. Those legacy tuples reject boundary preparation rather than lose offsets; use `load_prepared_splits` for boundary data.

## NN batching and scaling

```python
from src.debbirth.models.nn.data import tensor_data_from_prepared

x, y, loaders, datasets, scaler = tensor_data_from_prepared(
    splits, scaling_type="log_standardize", batch_size=128,
    device="cpu", num_workers=0, seed=42,
)
batch = next(iter(loaders["train"]))
# batch: x, y, row_index, log_nu_b
# provenance: datasets["train"].prepared.source_rows.iloc[batch["row_index"].numpy()]
```

This explicit prepared path emits dictionary batches. `row_index` indexes that dataset's prepared record and travels with the features, label, and optional offset through the sampler. Historical full-parameter training continues to receive `(x, y)` batches; its datasets also retain the prepared record.

NN scaling supports `none`, `standardize`, and `log_standardize`. Existing Torch scaler classes and serialization formats are retained. Fit occurs on training features only; supplied fitted scalers are reused without refitting. Offsets are never standardized. Disabling scaling still converts features and labels to float32 tensors and saves no scaler file. Training batches shuffle reproducibly using a seeded DataLoader generator. Non-CPU device-resident datasets require `num_workers=0`. CPU is the validated device; CUDA was unavailable.

## Configurations and run lifecycle

The dataclass JSON interfaces save formulation, explicit feature order, optional `x_b`, scaling, boundary temperature, family settings, and paths. Saved feature order must agree with the spec. `boundary_temperature` defaults to 1 and other values require `boundary`; GP currently supports only `scaling_type="none"`.

GP JSON stores stable primitive identifiers from `PRIMITIVE_REGISTRY`, an ordered list of constant names, and explicit depth settings. Constant values are defined once in `CONSTANT_REGISTRY` in `src/debbirth/models/gp/constants.py`. Add new constants there before selecting their names in a config. Existing names keep their existing values. Loading restores primitive objects, constant dataclasses, tuples, and numeric class-weight keys. Unknown identifiers and historical object-address strings raise clear errors rather than guessing a primitive. New saves contain only constant names, for example `"constants": ["c1", "c1_2", "sqrt2", "c0"]`. Unknown names and duplicate constants are rejected. Older name/value entries can still be read when their values exactly match the registry; overrides are rejected. Protected primitive implementations are unchanged. Archived joblib inference still works when its training configuration cannot be reconstructed; the loader warns and returns `train_cfg=None`.

### Class weighting (T06C)

Classifier losses are unweighted by default for every family and formulation. `TrainGPConfig.class_weights` defaults to `None`; `"balanced"` or `{0: w0, 1: w1}` (finite positive weights for both classes) is an explicit opt-in; other values are rejected when the config is built. In gplearn 0.4.3, `class_weight=None` gives unit sample weights. Subsampling still zeroes out-of-bag rows, so the raw fitness is the ordinary mean log loss over in-bag rows and the OOB fitness is the mean over out-of-bag rows. Balanced weighting instead uses a weighted mean over in-bag rows. Changing the weights changes the loss scale relative to the fixed parsimony coefficient, so tuned parsimony values from weighted runs do not transfer automatically. The NN default is `use_pos_weight=False` (see [NN interfaces](nn_formulations.md)). The GP tuner builds trials without `class_weights`, so tuning trials are unweighted too. Archived configs record `"balanced"`/`use_pos_weight: true` explicitly and keep that behavior. The T04 boundary prototype keeps its internal balanced weights as historical evidence; the production boundary trainer (T06) must not inherit them.

Shipped experiment configs follow the default. The previous weighted versions are retained for comparisons as `experiments/gp_full_par_balanced.json` and `experiments/nn_{full_par,normalized,boundary}_pos_weight.json`.

### Revised GP function set (T06B)

`REVISED_FUNCTION_SET` in `models/gp/functions.py` (tuner name `"revised"`) is the search set for the new normalized and boundary GP experiments. It is the extended set without `max`, `sqrt`, `pinv` and `neg`, plus `cube`. The resolved order and runtime semantics are:

| Primitive | Runtime definition |
| --- | --- |
| `add`, `sub`, `mul` | gplearn stock, unprotected |
| `pdiv(x, y)` | repository protected division: `(x + 1e-12) / y'`, with `y' = copysign(1e-12, y)` when `abs(y) < 1e-12` (zero maps to `+1e-12`) |
| `plog(x)` | repository protected log: `log(max(x, 1e-12))`; negative inputs map to `log(1e-12)`, unlike gplearn's `log(abs(x))` |
| `min` | gplearn stock `np.minimum` |
| `cbrt` | real cube root `np.cbrt` (negative for negative inputs) |
| `square`, `cube`, `atan` | `x**2`, `x**3`, `arctan(x)` |

The repository's protected primitives are used, not gplearn's stock `div`/`log`. Historical sets (`arithmetic`, `default`, `extended`), primitive definitions and serialized names are unchanged for archived models. Constants are chosen separately. The new configs use the same named constants as the full-parameter example (`c1, c2, c3, c1_2, c1_3, sqrt2, sqrt3, c0`); `sqrt2`/`sqrt3` are constant terminals, not the excluded `sqrt` operator.

`experiments/gp_normalized.json` (features `gamma, k, nu_b, x_b`) and `experiments/gp_boundary.json` (features `gamma, k, x_b`) select the revised set, `include_x_b=true` and unweighted loss, with the full-parameter example's untuned evolution settings. Setting `include_x_b=false` gives the otherwise matched ablation. Both train through `train_gp_classifier` (see [GP formulations](gp_formulations.md)).

### Symbolic export semantics

The existing `symbolic.py` export does not preserve these runtime semantics. `pdiv` becomes `a/b`, `plog` becomes `log(a)`, and `cbrt` becomes SymPy's principal cube root, which is complex for negative arguments. The T06B validation run checked 339 sampled revised-set programs: the export disagreed with runtime execution on data rows for 16 of them and was undefined for 8 more (for example `pdiv(x, c0)` read as `x/0`). Do not use it for inference until T06D is resolved.

Constructing or loading a config creates no directories. Relative data/output paths resolve against the repository root and serialize as relative paths when possible. Saving a run resolves `outdir=None` to a timestamped directory with microseconds, returning a new resolved config. Both trainers return `train_config`, `outdir`, `prepared`, and `data_metadata` alongside their historical outputs. Unsaved training returns `outdir=None` and creates no run directory. Use the returned config/path; do not assume the input config's `outdir` changes. Standalone save helpers also return the resolved config.

Saved runs include `run_metadata.json` with source CSV hashes, row counts, split policy, configuration, Python/dependency versions, Git revision, and source-code hashes identifying uncommitted code. This does not bundle the source code or datasets themselves. Explicitly saving a run creates its directories; explicitly saving a config creates the requested JSON parent.

Existing module entry points use `experiments/gp_full_par.json` and `experiments/nn_full_par.json`, preserving the prior example hyperparameters except that both are now unweighted (T06C). They train and report validation metrics; held-out test evaluation is separate. These examples are not tiny checks and do not reproduce the archived paper settings exactly. T04 selected gplearn and verified separate normalized/boundary GP prototypes; see [the decision and experiment](gp_backend_decision.md). T05 now implements NN normalized/boundary training, checkpoint selection, evaluation-mode loading and original-parameter inference; see [NN interfaces and validation](nn_formulations.md). T06 implements normalized/boundary GP training; see [GP formulations](gp_formulations.md). No common training CLI is included.

## Shared plotting (T06A)

`src/debbirth/plot/boundary.py` plots in original or normalized coordinates.

- **Slice grids:** `slice_grid(x_col, x, y_col, y, **fixed)` builds a mesh in `(g, v_Hb)` at fixed `(k, f)`, in `(gamma, nu_b)` at fixed `k`, or in `(gamma, k)`, and adds the equivalent original parameters for the predictors. Normalized slices are f-free and use `f = 1` unless given.
- **Decision meshes:** `plot_decision_mesh` and `plot_decision_mesh_with_models` label `gamma`, `nu_b`, `x_b` and `F` correctly. With `reference_lines=True` (the default) they draw the analytical screening lines in the plotted coordinate: `v_Hb = f^3/k` plus `f^3` or `(f/k)^3`, or the normalized `nu_b = 1/k` plus `1` or `1/k^3`. The slice values (k, and f for original axes) are verified to be single-valued instead of being read from the first row; mixed slices raise unless `reference_lines=False`. Existing single-slice notebook calls keep their output.
- **Critical curves:** `draw_critical_curve` overlays a boundary model's `Psi` (normalized) or `f^3 Psi` (original) from `critical_maturity`. `update_legend` rebuilds the legend with regions, lines, curves and model contours.
- **Critical surfaces:** `plot_critical_surface` maps `F = log(Psi)` over `(gamma, k)` with its learned zero contour and the analytical `k = 1` reference (`F = 0`).

## Direct validation results

Executed in conda `debbirth` on CPU. Data access required approved execution outside the sandbox; the data/tests directories are present and were not restored or replaced.

- Shared-loader import in a fresh process loaded neither Torch nor model configs.
- Both new representations prepared all supplied observations: 159,991 train, 19,999 validation, 19,999 test, preserving labels and membership.
- Equivalent scaled physical inputs produced matching normalized features/offsets. Feature orders, optional `x_b`, invalid inputs, extreme log offsets, float32 underflow, strict ties, temperature, probability monotonicity, and Torch gradients passed direct checks.
- Positional selection, train+validation merging, and shuffled dictionary batches preserved row/label/offset/diagnostic alignment. Changing validation features did not change fitted training scaler statistics.
- Both families' configs round-tripped for all three formulations; every registered GP primitive round-tripped. A patched directory-creation call confirmed config construction/loading performed no directory creation. Unsupported trainer modes failed before data access.
- Tiny full-parameter GP: population 32, two generations, 128 training and 32 validation rows. A separate one-generation unsaved run created no directories. The saved model's predictions matched after loading.
- Tiny full-parameter NN: one 8-unit hidden layer, two epochs, batch size 32, and separate `none`, `standardize`, and `log_standardize` runs on the same fixture. Losses were finite and saved/reloaded validation probabilities agreed exactly. The unscaled run saved/loaded with `scaler=None`.
- A separate one-epoch unsaved NN run created no files/directories. Nullable missing physical inputs reported the offending column; all source modules parsed successfully.
- Archived GP and NN predictions matched the four T01 synthetic probes. NN inference used explicit `eval()`, as before; making the loader default to evaluation mode remains T05. All supplied data and archived file hashes were unchanged.

The session check script, fixtures, and full summary are under `results/runs/t03_validation/` (ignored by Git). NN runs are its `nn_none`, `nn_standardize`, and `nn_log_standardize` directories; the final GP run is `results/runs/2026-09-06T17-40-35-699933_T03_smoke_GP/`. No full training, tuning, scientific performance comparison, new test suite, or dependency installation was performed.

On 2026-09-07, the constant-name registry follow-up passed direct checks for every registered value, names-only JSON round-trip without directory creation, rejection of unknown/duplicate/conflicting constants, runtime constant feature columns, legacy pairs, archived GP predictions, and symbolic constant substitution. This configuration-only change required no retraining or artifact rewriting.

On 2026-09-08, the GP tuner's selected-model saving branch was corrected to retain the resolved `train_config`/`outdir` returned by `save_gp_run` and pass the selected run's data metadata. Subsequent explicit test-metric saving now uses that resolved directory. A one-generation, population-16 GP model on the existing T03 fixture (128 train, 32 validation, 32 test rows) exercised all four combinations of `save_best_model` and `evaluate_on_test`. Returned paths/configs, conditional metric files, source hashes/split metadata, unchanged unsaved behavior, and exact reload prediction agreement passed. The probe and summary are under `results/runs/t03_save_validation/`; the summary identifies both saved runs.

Ray and HyperOpt were absent from `debbirth`. The probe executed the actual `hyperopt_calibration` function extracted from source, supplying simulated search orchestration/results and a real trained-model result. Saving, metrics, and reloading used the real implementations. This verifies the finalization regression, not end-to-end tuning. No dependencies were installed or source datasets/archived artifacts changed. T03 is complete; JSON-based tuning integration and end-to-end search verification remain T08B.
