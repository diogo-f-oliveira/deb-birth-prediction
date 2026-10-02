# Development tasks

Last updated: 2026-10-02

## Objective

Develop normalized and critical-boundary birth-feasibility models for both GP and NN, compare them fairly, and use the learned boundary to explore AmP species. Retain the full-parameter four-input models for benchmarking. The scientific motivation is in `docs/next_steps.md`; the equations are in `docs/birth_equations.md`; working conventions are in `AGENTS.md`.

## How to maintain this file

- Read this file at the start of development work. Follow the user's current request; when asked to continue generally, take the first ready task in the order below.
- Keep at most one task marked `IN PROGRESS`. Independent work can proceed when a dependency is blocked, but record why the order changed.
- Task states are `TODO`, `IN PROGRESS`, `BLOCKED`, `DONE`, and `DEFERRED`. Keep the checkbox unchecked until `DONE`. For blocked/deferred tasks, state the reason and the next action that would resolve it.
- Update the affected task before ending a work session: record the result, relevant file/run paths, actual checks, and remaining work. A task is done only when its completion criterion is met; a plan or unexecuted script is not a result.
- Add a brief dated entry to the progress log for completed milestones or decisions that change the plan. Keep detailed results in linked notes or run artifacts rather than growing the log into a transcript.
- Revise dependencies and completion criteria when findings change the approach. Record decisions and their rationale; do not silently drop work or overwrite earlier experimental results.
- Use conda `debbirth` for code execution. Prefer direct scientific checks and small experiments over new test files. Do not create package infrastructure or a general experiment framework just to complete this backlog.
- This file is a development plan, not a request to start every experiment now. Execute the scope of the active user request; no background scheduling is implied.

**Current task:** None. T01-T06, T06A, T06B and T06C are complete.

**Next action:** T06D before relying on simplified GP exports; T07 can proceed independently; T08/T08B before T09. Follow the active user request. JSON-based tuning integration remains T08B, before T09.

## Model comparison

**Before final experiments:** the revised GP function set (T06B) and unweighted classifier defaults (T06C) are implemented in configs and the existing tuner; T08B must carry them into JSON-based tuning. These are prerequisites for T09; historical artifacts retain their recorded settings.

| Formulation | GP | NN | Role |
| --- | --- | --- | --- |
| Full-parameter (`full_par`): `(g, k, v_Hb, f)` | Existing gplearn classifier | Existing DEBBirthNet | Historical benchmark |
| Normalized: `(gamma, k, nu_b)` | New unconstrained score | New unconstrained score | Main comparison |
| Boundary: `(gamma, k)` | Learn `F = log(Psi)` | Learn `F = log(Psi)` | Main comparison |

Use `gamma = g/f`, `nu_b = v_Hb/f^3`, and boundary probability `sigmoid((F - log(nu_b))/T)`, with `T > 0`. The four new models are the priority. Historical artifacts provide context; claims about the effect of formulation require comparable data, preprocessing, and training/tuning budgets, or an explicit explanation of the differences.

The target is practical feasibility: retain negative labels for get_lb2 timeouts/nonconvergence within the solver budget. Recovering theoretical feasibility with another solver is not required for this application. Diagnostics remain useful, but alternative-solver relabeling is outside the current plan. Exact DEB invariance constrains the model construction; it does not guarantee exact reproduction of numerical failure patterns. Use the current generated distribution for the main comparison and reserve AmP-specific evaluation for T12.

## Ordered backlog

### T01 - Verify the working environment and baseline access

- [x] **DONE** | Dependencies: none.
- Locate conda and confirm that repository code executes in `debbirth`. Record Python, NumPy, PyTorch, scikit-learn, and gplearn versions and CPU/GPU availability. Inspect optional tuning dependencies only when needed.
- Check actual accessibility of the processed splits, raw LHS data, and archived model files. Earlier sandbox reads could not access `data/` and `tests/`; determine the current state without restoring or recreating files based on Git's apparent deletions.
- Inspect existing exploratory notebooks before adding another one. Load the archived GP and NN and obtain predictions for a small shared set of valid inputs, preserving feature order and the saved NN scaler.
- **Done when:** a short environment/baseline note records working commands, artifact paths, and whether each baseline loads. Any remaining access or compatibility limitation is explicit. Do not reinstall requirements or launch full training as setup.
- **Result (2026-09-06):** `docs/environment_baseline.md` records the verified `debbirth` environment (CPU execution), data-file access outside the sandbox, existing notebook inspection, and successful inference from both archived models on four shared synthetic inputs. NN inference required explicit `eval()`. No dependencies, datasets, model artifacts, or source code were changed.

### T02 - Audit the data in normalized coordinates

- [x] **DONE** | Dependencies: T01 data access.
- For each supplied split, compute `gamma`, `nu_b`, and `log_nu_b` without changing the source CSVs. Record row counts, invalid/nonpositive values, class counts, ranges, and coverage across `k < 1`, `k = 1`, and `k > 1`.
- Count exact duplicate normalized triples and inspect near-equivalent points using a stated numerical tolerance. Check cross-split overlap and conflicting labels; preserve source-row identity. Do not deduplicate merely because the feature dimension decreased.
- Inspect solver diagnostics and timeout/error frequencies where available. If raw-to-processed linkage is needed, establish a reliable mapping before attributing diagnostics to rows.
- Plot a small set of normalized coverage/boundary slices. Decide whether the existing data suffice for initial training and identify specific gaps, rather than requesting a new dataset by default.
- **Done when:** a reproducible notebook or short analysis plus a saved summary records counts, plots, label caveats, and a concrete reuse/generation decision. Keep the historical splits intact.
- **Result (2026-09-06):** `experiments/audit_normalized_data.py` ran in `debbirth`. `docs/normalized_data_audit.md` and `docs/data_audit/` preserve the report, summaries, hashes, and inspected plots. All 199,989 observations remain distinct after normalization; no cross-split duplicate pairs at either audited tolerance. All split rows map uniquely to the processed/raw source with matching labels and boolean diagnostics. Reuse the splits for initial training. Note sparse gamma extremes, absent exact k=1 LHS points, 65,927 unsuccessful rows, 606 timeout zero placeholders, and one inconsistent timeout message. Full row provenance is under `results/runs/normalized_data_audit/`.

### T03 - Separate formulations, data preparation, and configuration

- [x] **DONE** | Dependencies: T02. Shared foundation and regression follow-up verified.
- Extend the data/schema path with explicit formulation names and feature order. Keep the full-parameter `(g, k, v_Hb, f)` behavior intact.
- Introduce a small formulation module (for example `src/debbirth/formulations.py`) that explicitly separates learned output, boundary margin, and probability. Share the mathematical definitions across GP and NN without forcing their training loops or array backends into one abstraction.
- Separate CSV/split loading and shared feature preparation from NN tensor conversion, device placement, and batching. Shared data preparation should not import model training configurations. Carry labels, source-row identity, and boundary offsets together through selection and shuffling.
- Centralize normalization and `log_nu_b = log(v_Hb) - 3*log(f)`. For boundary training, supply `(gamma, k)` to the learned model and carry `log_nu_b` separately for the fixed comparison. Expose `x_b = gamma/(1+gamma)` as an optional derived feature.
- Specify invalid-input behavior. Apply any learned standardization using training data only. Save formulation and transformation settings with each model; avoid copying transformations across notebooks and training modules.
- Fix split configuration inconsistencies: GP currently ignores `cfg.data_splits`, and trainers assume a validation split. Implement or explicitly reject unsupported modes; never silently substitute another split policy.
- Keep configuration construction/loading free of directory creation. Resolve shared paths in one place and create run directories only when starting/saving a run. Store registered GP primitive and constant names instead of relying on function-object strings from `default=str`; resolve constant values from the code registry.
- Move reusable experiment settings out of hardcoded training-module `__main__` blocks into a small `experiments/` directory. Keep dataclass configs and separate family-specific trainers; avoid a broad framework or unrelated file moves.
- **Done when:** both new representations can be obtained through shared preparation, a direct calculation confirms equivalent original inputs yield identical normalized inputs, the original input path still works, and config loading has no output-directory side effects. Split behavior and settings needed to reconstruct a run are explicit.
- **Result (2026-09-06):** Added `full_par`/`normalized`/`boundary` schemas, shared preparation and output semantics, aligned provenance/offset records, family adapters, pure config/path handling, registered GP serialization, and JSON experiment settings. `docs/formulations_and_data.md` records interfaces and actual checks. Both new representations prepared all 199,989 supplied rows. Direct equivalence/alignment/scaling/config checks and tiny full-parameter GP/NN save/load runs passed, including NN without scaling. Archived predictions and supplied data/model hashes are unchanged. Artifacts and the session check script are under `results/runs/t03_validation/`; its summary identifies the GP run. Boundary training and the new-model pilots remain T04-T06; CPU only was validated.

- **Follow-up (2026-09-07):** At the user's request, GP JSON now selects constants by name only. `CONSTANT_REGISTRY` in `models/gp/constants.py` supplies values. Updated the example, serializer, symbolic example, and documentation; direct checks covered all registry values, JSON round-trip without directory creation, invalid/duplicate/conflicting entries, runtime constant columns, archived predictions, and symbolic substitution. No training or artifact migration was needed.
- **Regression follow-up (reported 2026-09-07; resolved 2026-09-08):** `src/debbirth/models/gp/calibrate.py` now retains the resolved config/output path returned by `save_gp_run` and passes the selected run's data metadata. A direct check exercised all four combinations of saving and explicit test evaluation with a real GP model (population 16, one generation): paths/configs, conditional metric files, provenance, reload predictions, and unsaved behavior passed. The actual finalization function was executed from source with simulated search orchestration/results because Ray and HyperOpt are absent from `debbirth`; no end-to-end search is claimed. Probe, summary, and selected run paths: `results/runs/t03_save_validation/`. The earlier foundation checks remain valid; JSON-based tuning integration remains T08B.

### T04 - Choose the GP implementation approach

- [x] **DONE** | Dependencies: T01; T03 for a small end-to-end prototype.
- Inspect the installed gplearn implementation and current wrapper. Determine separately whether it supports the normalized classifier and the boundary classifier cleanly.
- Prototype the boundary fitness: a tree receives only `(gamma, k)` or their permitted transforms; the fitness uses row-aligned labels and `log_nu_b` in weighted BCE of `(F - log_nu_b)/T`, plus parsimony. Check how subsampling, parallel evaluation, prediction, and saving preserve this alignment.
- Reject approaches that allow maturity into the evolved tree, hide it in the target labels, or depend on fragile implicit row order. Evaluate a small custom extension versus DEAP or another suitable backend if gplearn's interfaces are insufficient.
- **Done when:** a short decision note names the chosen backend(s), explains the tradeoff, and links a working minimal boundary-fitness experiment. Preserve the existing gplearn benchmark without building a general backend abstraction.
- **Result (2026-09-08):** `docs/gp_backend_decision.md` recommends gplearn for all formulations and compares DEAP/PySR benefits and costs. `src/debbirth/models/gp/boundary_prototype.py` uses a raw-output SymbolicRegressor engine with binary labels and separately bound weighted BCE offsets, guarded by an audited backend source contract. `experiments/prototype_gp_boundary.py` passed in `debbirth` on matched 512/128 training/validation rows: serial/two-process agreement, full/65% sample masks, explicit raw/OOB/parsimony recomputation, record shuffling/repeated coordinates, strict ties, monotonicity, scaling and fresh-process reload. Results: `results/runs/2026-09-08T15-45-43-866373_t04_gp_backend/`. No test-set use, dependency installation, full tuning or production integration. T06 must integrate the adapter, intended primitives/constants, exports and explicit final-program selection; gplearn's final selection uses raw loss despite parsimony in tournaments.
- **Documentation follow-up (2026-09-08):** Expanded the decision note with the threshold-to-log-margin representation, sigmoid probability, Bernoulli-to-stable-BCE derivation, class weights/sample masks, parsimony, a worked maturity example and the explicit gplearn metric binding. Confirmed reuse of the existing `full_par` training/tuning as the historical benchmark; matched-budget retraining would be a separate experimental choice. Checked equations against the prototype and reviewed the content/diff; no code or model runs changed.

### T05 - Implement normalized and boundary neural networks

- [x] **DONE** | Dependencies: T03. Uses the fixed-offset representation documented by T04.
- Extend the existing NN configuration/training path to support the normalized three-input score and the two-input `F(gamma, k)` output. Compute boundary logits outside the learned network using the fixed maturity offset and positive temperature.
- Keep input preprocessing configurable and recorded. The final `F` output must be unrestricted in sign; positivity applies to `Psi = exp(F)`, not to `F`.
- Save and load formulation, feature settings, scaler, weights, and temperature. Expose the boundary margin and critical maturity for analysis as well as birth probabilities.
- No-scaling tensor conversion and saving/loading with no scaler were completed in T03. Still return inference-loaded models in evaluation mode so dropout is inactive by default.
- Make checkpoint selection explicit: final epoch or best validation checkpoint, with the metric and selected epoch recorded. Save metrics corresponding to the saved weights. Keep historical reproduction settings available.
- Run a small training experiment for each new formulation. Check finite loss/probabilities, prediction agreement after reload, and that the boundary probability decreases with increasing maturity at fixed `(gamma, k)`.
- **Done when:** both variants train, save, reload, and predict through the existing workflow, with run paths and checks recorded. This task does not require full tuning.
- **Result (2026-09-08):** `docs/nn_formulations.md` documents normalized/boundary NN training, unrestricted F, fixed-offset BCE, preprocessing, positive temperature, original-parameter inference and explicit final/best-validation checkpoint selection. `experiments/validate_nn_formulations.py` passed in `debbirth` on matched 512/128 train/validation rows, including two eight-epoch main runs, no-scaling and optional-feature runs, and synthetic CSV/save-selection probes. Saved metrics match selected weights; fresh-process predictions agree exactly. Alignment, train-only scaling, monotonicity, strict ties, invariance, extreme offsets/logit underflow and archived NN compatibility passed. Artifacts: `results/runs/2026-09-08T16-27-53-838745_t05_nn_validation/`. Existing data, archived artifacts, historical configs and GP tuning source retained identical hashes. No supplied test data, full tuning, GP integration, plotting refactor or dependency installation was involved.

### T06 - Implement normalized and boundary GP models

- [x] **DONE** | Dependencies: T03 and T04.
- Implement both formulations with gplearn as selected in T04 (`docs/gp_backend_decision.md`). For the boundary model, evolve `F(gamma, k)` with the fixed offset outside the tree and a documented positive training temperature. Integrate the prototype's explicit row/mask contract; keep loss normalization separate from tournament parsimony. **Decision (2026-10-02):** keep gplearn's last-generation final-program selection (raw loss); the number of generations is a tuned hyperparameter, analogous to final-epoch NN checkpoints under dropout. Document it; alternative selection policies are T08C, outside T06.
- Use the repository's protected primitives and named constant terminals in the boundary engine rather than the prototype's stock `div`/`log` and `const_range` constants (displayed random constants are rounded). Constant columns must not be mistaken for, or widen, the permitted `(gamma, k[, x_b])` features. Exports used for inference depend on T06D; until then use the executable gplearn program.
- Integrate the revised function set and x_b feature configuration specified in T06B; retain historical primitive definitions for old artifacts. Apply T06C's unweighted defaults to the production boundary loss as well as the normalized classifier.
- Save sufficient settings and the raw expression to reproduce inference. Export readable expressions with the actual feature names and full original-variable feasibility rule, including normalization and `exp(F)` where applicable.
- Keep historical gplearn class/module import paths and primitive definitions available for loading archived artifacts. Confine backend-specific evolution and serialization to the GP implementation; analysis code should not need to access gplearn private attributes.
- Run a small evolution for each formulation. Confirm finite fitness, row alignment, save/load prediction agreement, and numerical agreement between the executable expression and any export used for inference.
- **Done when:** both variants train and produce reusable symbolic models, with short-run artifacts and an example boundary rule. Do not claim simplified expressions preserve protected semantics without checking.
- **Result (2026-10-02):** Implemented in `src/debbirth/models/gp/boundary.py`, `expression.py` and `train.py`; documented in `docs/gp_formulations.md`.
  - **Boundary engine:** `fit_gp_boundary` evolves F with gplearn's SymbolicRegressor and the T04 fixed-offset BCE. It reuses the prototype's audited loss and backend contract. Only `gamma, k[, x_b]` are terminals. It uses registered primitives and named constant columns with no ephemeral constants, unit weights by default, and the configured training temperature. The final program is the last generation's lowest raw loss.
  - **Inference artifact:** `GPBoundaryModel` holds no training context and exposes F, margin, probability and strict `margin > 0`. Normalized/full_par use the existing classifier.
  - **Trainer:** `train_gp_classifier` handles all formulations and accepts explicit row-aligned `prepared` subsets, like the NN trainer. Runs record loss, decision and selection policy. Saved runs add `model/expression.txt`: the exact program with named features/constants, runtime primitive definitions and the original-variable rule.
  - **Exports:** no SymPy simplification; simplified exports remain T06D. The joblib model is the inference artifact, so no text export is used for inference.
  - **Validation:** `experiments/validate_gp_formulations.py` passed on 2,000/500 sampled train/val rows (no test data) for normalized+`x_b`, boundary+`x_b` (T=1) and boundary without `x_b` (T=0.7). It covered independent fitness recomputation for all retained programs (718/575, plus 250 with explicit balanced weights), the final-program policy, permitted terminals/primitives, serial/two-process identity, shuffle alignment, metric context rejection, maturity monotonicity, strict ties, temperature-invariant decisions, saved-metric recomputation, and exact in-process and fresh-process reload. Archived artifacts and source CSVs were unchanged. A one-generation smoke run trained all three formulations through ordinary CSV loading.
  - **Example boundary rule (8 generations, not tuned):** `F = sub(sqrt2, k)`, i.e. `v_Hb < f^3 * exp(sqrt(2) - k)`. It ignores gamma and violates `F(gamma, 1) = 0`; it demonstrates execution only.
  - **Run:** `results/runs/2026-10-02T16-55-47-145588_t06_gp_validation/`. Tuning/temperature remain T08/T08B. A shared original-parameter GP predictor and plotting remain T06A.

### T06A - Unify inference and simplify evaluation

- [x] **DONE** | Dependencies: T03, T05, and T06. Implement alongside those tasks where useful.
- Add a lightweight predictor that bundles the learned model, formulation, feature order, preprocessing, and temperature. Its public inference entry point accepts the original parameters for every formulation; internal prepared-input paths must be explicit to prevent applying transformations twice.
- Return a consistent positive-class probability shape. For boundary models, also expose signed margin and normalized/original critical maturity. Keep low-level model APIs available for training and historical use.
- Separate metric computation from model execution: compute metrics from labels, probabilities, and an explicit decision rule. Collect NN predictions and loss in one batched pass instead of collecting the full input dataset and predicting again. Avoid separate model calls for labels and probabilities.
- Add MCC and the probability metrics needed for the temperature experiments. Keep strict zero-margin classification distinct from any tuned operating threshold; use the same rules across model families.
- Adapt plotting so normalized axes and critical surfaces have correct labels and reference bounds. Remove the assumption that every plot has original-variable `f` and `k` annotations taken from the first row; check slice assumptions where they are needed.
- **Done when:** a small shared input set can be passed through each supported formulation without caller-managed scaling, predictions survive saving/loading, and shared evaluation/plotting works for both original and normalized coordinates. Record direct checks rather than creating a new test suite.
- **Partial implementation with T05 (2026-09-08):** Added `NNPredictor` for all three NN formulations, original-parameter and explicit prepared-input paths, shape-(N,) positive probabilities, boundary margins/critical maturities, and explicit operating-threshold overrides. NN validation now collects loss/predictions in one batched pass; `metrics_from_predictions` separates execution from metrics and adds MCC, log loss and Brier score with historical metric loading retained. Checks are recorded with T05. T06A remains TODO for GP predictor integration, generic execution consolidation and coordinate-aware plotting; no broader framework was introduced.
- **Result (2026-10-02):** Completed the GP and plotting parts.
  - **Shared predictor:** `ParameterPredictor` (`src/debbirth/evaluate/predictor.py`, no backend imports) holds original-parameter parsing, explicit operating thresholds, margins, critical maturity and one-pass `evaluate_prepared`. `NNPredictor` now uses it; its moved methods are unchanged and its outputs match the saved T05 probes exactly.
  - **GP predictor:** `GPPredictor` (`models/gp/predict.py`) gives GP classifiers and `GPBoundaryModel` the same interface. Classifier outputs come from one program execution and equal gplearn's `predict_proba`/`predict`. `load_gp_run` returns a predictor, including for the archived full-parameter GP via its historical schema. `train_gp_classifier` validates through one predictor pass.
  - **Decision rules:** boundary decisions are strict in both families. Score models keep their historical rules (GP `> 0.5`, NN `>= threshold`), which differ only at exactly 0.5.
  - **Plotting:** coordinate-aware (`slice_grid`, `reference_bounds` with verified single-valued slices, normalized labels, `draw_critical_curve`, `update_legend`, `plot_critical_surface` with the analytical k=1 reference). Existing calls keep their output.
  - **Validation:** `experiments/validate_t06a.py` passed on 500 sampled validation rows (no test data) for eight saved models: archived GP/NN, three T05 NN and three T06 GP runs.
    - **Interface checks:** shapes, finiteness and probability ranges; identity of the original-parameter and prepared paths; threshold path; strict boundary decisions and logit = margin/T; and critical maturity in log/exp and original/normalized forms, consistent with the decisions.
    - **Equivalence checks:** scaling invariance for normalized/boundary models (max probability change 7e-16 GP, 0 NN; no decision flips); equality with gplearn APIs; and one-pass metrics reproducing T06's saved validation metrics. The trainer's validation metrics equal the historical evaluator.
    - **Reload and plots:** exact fresh-process reload of all eight predictors. Original, normalized (k=0.3, 3) and critical-surface plots were inspected; a mixed-f slice is rejected. Archived artifacts and source CSVs were unchanged.
  - **Follow-up (2026-10-02):** `train_gp_classifier` now returns its `predictor`, as `train_net` does, so unsaved GP runs need no manual `GPPredictor`. A direct check confirmed that for saved and unsaved normalized/boundary runs, the returned, freshly built and reloaded predictors agree exactly and reproduce the validation metrics.
  - **Run:** `results/runs/2026-10-02T17-07-29-933074_t06a_validation/`. The tuner's test-set evaluation still uses the historical `evaluate_binary_classifier`; JSON-based tuning is T08B.

### T06B - Create and configure the revised GP function set

- [x] **DONE** | Dependencies: T03's registries and shared features; coordinate with T06 production integration. Required before finalizing T08 and running T09 final experiments.
- Implement a separately named function set following `docs/next_steps.md`: remove `max`, square root, inversion, and negation from the new search set. Enumerate the retained operators and their protected semantics; do not silently substitute gplearn's stock protected operators for the repository's definitions.
- Include `x_b = g/(f+g) = gamma/(1+gamma)` through shared feature preparation and explicit experiment configuration. It is a derived feature, not an independent parameter or a reason to expose f to the boundary tree. Retain an otherwise matched without-x_b option for a later ablation if useful.
- Keep historical function sets, primitive definitions, and serialized names available for old models. Removing an operator from the new set does not remove named constants such as sqrt2/sqrt3; state constant choices separately.
- Make the revised set and feature schema selectable in GP JSON experiments and tuning through the existing registries. Use them explicitly for the final normalized/boundary GP experiments; historical full_par artifacts keep their recorded set.
- **Done when:** the named set, resolved operator list, feature order, and constant choices are documented and reconstructable from an experiment config; a small direct/integration check confirms excluded operators cannot be sampled and exports agree with runtime semantics. Link the final configs and actual checks here; no new test framework is required.
- **Result (2026-10-02):** `REVISED_FUNCTION_SET` (`"revised"` in `NAMED_FUNCTION_SETS`, used by the tuner): `add, sub, mul, pdiv, plog, min, cbrt, square, cube, atan` with the repository's protected `pdiv`/`plog`. Semantics, constants (`c1, c2, c3, c1_2, c1_3, sqrt2, sqrt3, c0`, as in the full-parameter example) and feature orders are documented in `docs/formulations_and_data.md`. Configs: `experiments/gp_normalized.json` (`gamma, k, nu_b, x_b`) and `experiments/gp_boundary.json` (`gamma, k, x_b`); `include_x_b=false` is the matched ablation. Historical sets/primitives are unchanged. `experiments/validate_t06bc.py` ran in `debbirth` on 2,000 sampled training rows. The function set resolves from JSON. Over 5 generations x 300 programs, only retained operators were sampled, and all ten were used. Run: `results/runs/2026-10-02T16-16-03-355538_t06bc_validation/`.
- **Criterion change (2026-10-02):** export agreement is moved to T06D at the user's request, because changing the symbolic pipeline is outside this task's scope. The same run measured the existing export against runtime execution and recorded a finding, without asserting it. Of 339 unique programs, 16 disagree on data rows, 18 on data/extreme/tied rows, and 8 are undefined (e.g. `pdiv(x, c0)` parsed as `x/0`). The cause is that `pdiv`/`plog` lose protection and `cbrt` becomes a principal root. `symbolic.py` is unchanged.

### T06C - Make classifiers unweighted by default

- [x] **DONE** | Dependencies: existing family configs; coordinate with T05/T06 and T08B. Required before finalizing T08 and running T09 final experiments.
- Default to ordinary unweighted BCE/log loss for GP and NN across full_par, normalized, and boundary formulations. Use GP `class_weights=None` and NN `use_pos_weight=False` with no active positive-class weight. Class balancing remains an explicit opt-in for historical reproduction or a deliberate separate experiment.
- Inspect every path that can reintroduce weighting: config defaults, shipped experiment JSON, training entry points, the boundary adapter, tuning defaults/overrides, and examples. The T04 prototype currently constructs balanced weights internally; production integration must not inherit that behavior implicitly. Preserve completed prototype results as historical evidence.
- Keep gplearn subsampling/OOB masks intact: removing class balancing means unit class weights, not disabling sample inclusion masks. Compute default loss as a mean over active observations. Record any consequent change to the effective loss/parsimony scale.
- Make validation probability loss and temperature fitting unweighted by default too. Preserve evaluation metrics such as macro-F1 and MCC; these do not require class-weighted training. Explicit weighting must remain visible in the resolved config.
- Preserve archived model/config files and reproduction settings. Mark previous weighted runs as historical or pilot results. Train/tune new final models under the unweighted policy; an older weighted artifact can remain a disclosed historical benchmark, not a matched unweighted run.
- **Done when:** omitted weighting settings resolve to unweighted loss through direct training, JSON/CLI, and tuning paths where available; a small calculation verifies ordinary BCE on active rows, and explicit legacy weighting remains selectable. Record actual checks and resolved settings; do not run final searches as part of changing defaults.
- **Result (2026-10-02):** `TrainGPConfig.class_weights` now defaults to `None`; `"balanced"` or a class mapping remains an explicit opt-in. NN `use_pos_weight=False` was already the default. The shipped `gp_full_par.json` and `nn_{full_par,normalized,boundary}.json` are unweighted; the previous weighted versions are kept as `gp_full_par_balanced.json` and `nn_*_pos_weight.json`. Archived configs record their weighting explicitly, and the archived GP still loads as `balanced` and predicts. Checks in `experiments/validate_t06bc.py` (same run as T06B):
  - Default and JSON resolution: GP configs give `None`, NN configs give `False`.
  - The GP tuner builds trials without `class_weights`. This was checked by static inspection because Ray/HyperOpt are absent.
  - gplearn raw/OOB fitness equals ordinary mean log loss over in-bag/out-of-bag rows with 70% subsampling and with full sampling. Explicit `"balanced"` still equals the weighted mean (max relative error 4e-16).
  - Two-epoch default normalized/boundary NN runs report validation loss equal to unweighted mean BCE.

  **Follow-up (2026-10-02, user-approved):** `TrainGPConfig` now rejects `class_weights` other than `None`, `"balanced"` or a mapping of both classes to finite positive weights. NN configs reject a `pos_weight` set while `use_pos_weight` is false. A direct check confirmed the valid/legacy settings, the rejections, and that all shipped and saved configs load as before. The exceptions are the archived/January GP configs with historical function-object strings, an unchanged limitation; their models still load via `load_gp_run`.

  Unweighted loss changes the loss scale relative to parsimony, so earlier tuned values do not transfer automatically. The T04 prototype still balances internally as historical evidence; the T06 production trainer must not. The T08A CLI loads the same JSON configs and applies no weighting override. JSON-based tuning remains T08B.

### T06D - Make GP symbolic exports agree with runtime semantics

- [ ] **TODO** | Dependencies: T06B. Needed before any symbolic/MATLAB export is used for inference or reported as the model (T06 exports, T09/T12).
- **Problem:** `src/debbirth/models/gp/symbolic.py` maps protected primitives to plain mathematics: `pdiv` to `a/b`, `plog` to `log(a)` and `pinv` to `1/a`. It also maps `cbrt` to SymPy's principal (complex for negatives) root, and gplearn's stock protected `div`/`log`/`sqrt`/`inv` to unprotected forms. `sp.simplify` can then remove expressions whose runtime behavior depends on protection. On the T06B run (`results/runs/2026-10-02T16-16-03-355538_t06bc_validation/`), 16 of 339 revised-set programs disagreed with runtime execution on data rows and 8 exported as undefined. AGENTS.md requires that simplification not silently discard protection semantics.
- Decide the export contract: e.g. keep protected primitives as unevaluated named functions with numerical implementations for an executable export, alongside a readable form whose assumptions are stated. Keep historical readable output available for the archived model, and update the existing symbolic tests only as needed.
- **Done when:** the exported expressions used for inference agree numerically with runtime execution on data and protection-activating inputs, including after simplification. Where the readable form is only conditionally valid, it is labeled as such. The archived model's export behavior is documented.

### T07 - Review and complete the mathematical proof

- [ ] **TODO** | Dependencies: `docs/birth_equations.md`. Can proceed independently of model implementation; does not block initial training or tuning.
- Check the assumptions, parameter domain, scaling transformation, and integral expressions in the existing derivation.
- Establish that the viable trajectory family attains every maturity below the proposed critical value, making the feasible maturity set the stated interval.
- Justify why the specified endpoint gives the maximum attainable viable maturity, particularly for `k > 1`. Distinguish a supremum on the strict viable region from a value attained on its limiting boundary.
- Check existence and selection of `lambda_R`. Determine whether uniqueness is required for the characterization and prove it if so; do not assume uniqueness without an argument.
- Confirm both directions of `nu_b < Psi(gamma, k)`, the strict inequality, and the treatment of equality, including the `k=1` case.
- Update `docs/birth_equations.md` with missing arguments and explicit assumptions. If a claim remains unresolved, identify the exact claim and missing argument; do not present the characterization as fully proven or mark this task complete merely because the gap is documented.
- **Done when:** every step leading to `nu_b < Psi(gamma, k)` is justified analytically, with assumptions stated. Numerical experiments are not a substitute for proof and are not required by this task.
- **Scope:** mathematical review only. Implementation checks belong to T03/T05/T06; no DEBtool comparisons, numerical Phi/Psi evaluator, solver retries, or timeout relabeling are required. Complete this review before presenting the full critical-boundary characterization as proven.

### T08 - Finalize the experiment and temperature protocol

- [ ] **TODO** | Dependencies: T02, T05, and T06 pilot runs; T06B and T06C before finalizing the protocol.
- Write the experiment matrix, matched split/subset identities, preprocessing choices, weighting, selection metrics, seed list, and tuning budgets before full training. Use pilot runtime to set practical budgets. Include repeated training seeds and distinguish archived benchmarks from models retrained under the new protocol.
- Require T06B's revised GP function set/feature settings and T06C's unweighted defaults in the final new-model configs. Weighted or old-function-set comparisons must be explicit historical benchmarks or separately identified experiments.
- Use validation macro-F1 for classification model selection unless a documented research reason favors another metric. Report MCC and per-class errors as well. Reserve test results for the finalized comparison.
- Separate training temperature from post-training calibration. If training temperature is searched, fit each candidate on training data and select using validation data. If fitting a post-training temperature with fixed `F`, minimize validation log loss (state weighting); macro-F1 at the zero-margin threshold cannot select temperature because positive temperature does not move that boundary.
- State whether any extra operating threshold is tuned, and report it separately from the canonical strict `margin > 0` decision. Do not confuse class-weighted scores with calibrated probabilities under the original class prevalence.
- Record whether NN comparison uses the last epoch or a validation-selected checkpoint. Save the resolved experiment settings and selected epoch with each run so the comparison can be reproduced.
- **Done when:** a saved, runnable experiment specification defines the four main models, historical comparison, budgets, selection rules, and temperature treatment without test-data tuning.

### T08A - Add a small shared training CLI

- [x] **DONE** | Dependencies: T03, T05, and T06; use T08 for full-experiment configurations.
- Provide one repository-root module entry point for training, backed by the same Python functions used in notebooks: `python -m src.debbirth.train`.
- Use a small standard-library argument parser. Expose model family (`gp`/`nn`), formulation (`full_par`/`normalized`/`boundary`), an experiment config file, and a few common overrides such as seed, data directory, output directory, and device/workers where applicable. Keep detailed architecture, primitive sets, and search settings in the config file rather than creating a flag for every parameter.
- Define precedence explicitly: defaults, then config, then explicitly supplied CLI overrides. Validate family/formulation compatibility and inputs before expensive work. Save the fully resolved config and invocation with the run and print the output directory and validation summary.
- A training invocation should fit and validate one run. Keep held-out test evaluation explicit and separate from routine training/tuning. Leave experiment grids and hyperparameter search to simple scripts calling the same training functions initially.
- Preserve callable training functions and historical entry points where practical. Do not require installation as a package, add orchestration infrastructure, or implement CLI-only training logic.
- **Done when:** help is useful, a small NN run and a small GP run can each be launched and reproduced from a saved configuration through the CLI, and invalid combinations fail clearly before training. Document actual commands in `README.md`.

Working command:

```text
conda run -n debbirth python -m src.debbirth.train --model nn --formulation boundary --config experiments/nn_boundary.json --seed 42
```

- **Result (2026-10-02):** Implemented `src/debbirth/train.py`. It is an argparse wrapper around `train_gp_classifier` / `train_net` with no training logic of its own.
  - **Inputs and precedence.** `--config` defaults to `experiments/<model>_<formulation>.json`. Precedence is defaults → config → explicit flags.
  - **Overrides.** Runtime and identity settings only: `--seed`, `--data-dir`, `--outdir`, `--run-name`, `--num-workers`, and `--device` (NN only). At the user's direction, hyperparameters including `boundary_temperature` stay in config files.
  - **Output directory.** At the user's direction, the CLI treats a config's `outdir` as provenance: each run gets a fresh directory unless `--outdir` names a new or empty one.
  - **Dry run and record.** `--dry-run` resolves and prints without loading data. Each run writes `cli_invocation.json` (argv, config path and sha256, overrides, reproduce command) next to the trainer's usual files.
  - **NN `run_name`.** NN configs now accept `run_name`, matching GP. The default directory suffix is `DEBBirthBoundaryNet` for boundary runs and `DEBBirthNet` otherwise. Old NN configs load with `None`; the archived model and a January run reload and predict finitely. The field is not used by model construction.
  - **Checks run in `debbirth`:**
    - `--help` reads correctly.
    - Nine invalid invocations exit with code 2 and a clear message before data loading, with no new run directory. The cases were: GP/NN config under the wrong family (both directions), formulation mismatch, `--device` with GP, temperature on a normalized config, NN `input_dim` mismatch, missing config, non-empty `--outdir`, and an unknown formulation.
    - Dry runs of all six shipped configs resolve, with overrides visible in the printed config.
    - The GP path does not import torch.
  - **Small real runs** on the full splits, with `--seed 7` and scratch configs derived from `gp_boundary.json` (population 100, 3 generations, tournament 10, 1 worker) and `nn_boundary.json` (2 epochs, CPU):
    - Original runs: `results/runs/2026-10-02T18-00-29-879113_t08a_cli_gp/` and `2026-10-02T18-00-44-294086_DEBBirthBoundaryNet/`.
    - Reproductions through `--config <run>/train_*_config.json`: `2026-10-02T18-01-07-652664_t08a_cli_gp/` and `2026-10-02T18-01-22-517688_t08a_cli_nn_repro/`.
    - The reproductions gave an identical GP program, identical NN state dict and checkpoint, equal validation metrics, and identical reloaded validation predictions and probabilities for both families.
    - No test-split evaluation was performed.

### T08B - Integrate experiment configurations with hyperparameter tuning

- [ ] **TODO** | Dependencies: T03 for the existing GP path; T05/T06 for the new NN/GP variants. Use T08 for full-experiment search budgets and selection rules. T08A is not required: tuning calls the same Python trainers.
- Load a base experiment JSON through the family dataclass loader. Construct each trial from that base plus sampled overrides; preserve all unsearched settings, including formulation, data paths, split policy, preprocessing, weights, and runtime options. Do not mutate the base config between trials.
- Keep search-space definitions in small Python experiment scripts initially, using the existing Ray/HyperOpt approach where suitable. Avoid embedding executable distributions in ordinary training JSON or introducing a general configuration framework. Inspect optional tuning dependencies in `debbirth` before executing a tuning check.
- Map sampled values to actual config fields explicitly. Compute dependent parameters such as tournament size, crossover probability, and mutation shares before validating the resolved trial config. Reject unknown or unused override keys instead of silently ignoring `gp_config_params`; use the same resolution when rerunning the selected trial.
- Use registered primitive and constant names for fixed settings and categorical choices between alternative lists. Resolve names through the existing registries when constructing each trial; do not add separate tuning-only definitions.
- Carry T06B's revised function set and x_b settings into the final GP base configs. Follow T06C's unweighted defaults; do not hardcode balancing when constructing or rerunning trials. Preserve and record any explicit weighting override.
- Preserve matched data/subset identities and use validation data for selection. Follow T08 for training-temperature searches versus post-training calibration; held-out test evaluation stays explicit and separate from the search.
- Save the base configuration, search specification/script identity, search seed/budget/objective, sampled trial parameters, and fully resolved ordinary training JSON for each trial. The selected run must retain its resolved output path, model/scaler state, metrics, and data/code provenance and be reproducible without invoking the tuner.
- **Done when:** tiny GP and NN tuning runs exercise fixed settings, sampled overrides, dependent parameters, and named constant/function choices as applicable; selected artifacts reload consistently, and a selected run can be reproduced from its saved training JSON with matching data and seed. Record actual commands, resolved configurations, and validation results without full tuning or a new test framework.
- **Current gap (2026-09-07):** The existing GP tuner constructs configs independently of the JSON examples, uses hardcoded named function/constant sets, and ignores extra GP configuration overrides. No JSON-based tuning integration or end-to-end tuning verification is claimed yet. The best-model saving regression was corrected separately under T03 on 2026-09-08.

### T08C - Consider GP final-program selection as a hyperparameter

- [ ] **TODO (proposed)** | Dependencies: T06 and T08B. Not required for T06; the default stays gplearn's last generation.
- The default policy keeps the last generation's best raw-loss program, with the number of generations tuned. This mirrors final-epoch NN checkpoints under dropout. Alternatives include selecting across generations or among final-generation candidates by validation loss/macro-F1, or by parsimony-penalized fitness. They change model selection and need retained programs, so treat them as an explicit tuning choice, not a silent default.
- If pursued, compare policies on validation data only with matched budgets. Record the policy, metric and selected generation with each run, and keep the last-generation policy as the reference.
- **Done when:** a recorded decision either keeps last-generation selection with rationale, or implements a selectable policy that resolves in configs/tuning and is validated on a small run.

### T09 - Train, tune, and compare the models

- [ ] **TODO** | Dependencies: T08, T08B, T06A, T06B, and T06C; T08A's CLI (`python -m src.debbirth.train`) is available for command-line execution.
- Before final training/tuning, confirm T06B and T06C are complete and inspect resolved configs for the revised GP set, feature schema, and disabled class weighting. Earlier pilot runs do not satisfy this prerequisite.
- Run the agreed experiments in distinct run directories. Record configurations, seeds, timing, model size/expression complexity, training history, and selected artifacts. Keep interrupted or failed runs identifiable.
- Freeze model choices using validation results, then evaluate the test split. Produce a consolidated table with macro-F1, MCC, per-class precision/recall, confusion counts, AUROC/AP, and probability quality where relevant.
- Plot learned boundary slices and `F`/`Psi` across the three maintenance regimes using the existing data and analytical constraints. Report agreement with the practical labels, including failures/timeouts, and identify extrapolation. A separate numerical critical-surface reference is not required.
- Broad-coverage versus near-boundary summaries may be useful diagnostics if results warrant them; do not generate extra evaluation datasets by default. AmP-specific performance belongs to T12.
- Measure both single-sample and batch inference cost under a recorded hardware/timing protocol. Include preprocessing and boundary comparison in the end-to-end surrogate timing.
- **Done when:** a reproducible comparison notebook/report links selected runs, tables, plots, uncertainty across seeds, and limitations. Performance conclusions must follow the results rather than the expected GP advantage.

### T10 - Measure sample efficiency and targeted ablations

- [ ] **TODO** | Dependencies: T09.
- Use nested stratified subsets of the training split, shared across models and seeds. Initial candidate fractions are 1%, 5%, 10%, 25%, 50%, and 100%; adjust after checking class counts and runtime, and record the final choice.
- Keep validation/test membership fixed. State whether hyperparameters remain fixed from T09 or are retuned with matched budgets; these answer different questions. Do not use test learning curves to tune the models.
- Plot performance versus training rows with variation across seeds. Use a small number of controlled ablations to investigate log preprocessing, `x_b`, or the revised primitive set; do not confound backend, formulation, and preprocessing changes when attributing gains.
- **Done when:** learning curves and an interpretation distinguish sample-efficiency evidence from preprocessing/tuning effects, with linked run identities.

### T11 - Extend simulation generation only where justified

- [ ] **DEFERRED** | Dependencies: T02/T09 identify a concrete sampling gap, or T12 establishes a need.
- **Decision (2026-09-06):** T02 supports reuse of the existing 199,989 rows for initial classification experiments. Revisit only for a demonstrated sampling need. Do not regenerate merely to remove f as an independent input or retry failures solely to recover theoretical feasibility.
- If transformed existing data are sufficient, mark this task `DEFERRED` with that finding. Revisit if boundary diagnostics or AmP coverage expose gaps.
- Otherwise, adapt a generator to sample normalized `(gamma, k, nu_b)`, passing `(g=gamma, k, v_Hb=nu_b, f=1)` to the existing solver for compatibility. Specify ranges and boundary-focused sampling from the measured gaps.
- Keep new datasets separate and save solver version/settings, timeout, sampling seed, and diagnostic/label policy. Retain timeouts/nonconvergence as negative practical-feasibility labels and keep their diagnostics separate from returned-solution constraint violations.
- **Done when:** either the no-new-data decision is recorded, or a small pilot confirms the new generator and its provenance before the needed dataset is produced. If new data change the benchmark, version the protocol and repeat affected comparisons explicitly.

### T12 - Explore AmP species relative to the learned boundary

- [ ] **TODO** | Dependencies: T09; access to suitable AmP data.
- Identify a versioned AmP source and retain species identity/model type. Check applicability of the standard embryo equations. Separate fitted species parameter sets from the repository's perturbed-parameter simulation dataset.
- Evaluate the learned classifier on AmP as a later application study. Expect strong feasibility skew and report false rejections/feasible-case recall appropriately; do not mistake performance on mostly feasible species for evidence of discrimination on both sides of the boundary. Do not redesign training around the AmP class proportions by default.
- Compute `eta = v_Hb/g^3` and plot species in `(eta, k)` space. At explicitly stated maternal food levels, compute `gamma`, `nu_b`, and signed margin `F(gamma,k) - log(nu_b)`.
- Find food-boundary intersections using `log(eta) + 3*log(gamma) = F(gamma,k)`, then `f_crit = g/gamma_crit`. Search an explicit admissible food interval, handle absent/multiple roots, and verify which side is feasible before calling a root the minimum viable food level.
- Plot margin and food thresholds with species metadata where available. Flag out-of-domain species and uncertain boundary estimates. Numerical comparisons are optional tools for a specific interpretation question, not required recovery of timeout-labeled points.
- **Done when:** a reproducible analysis links species provenance, derived values, plots, root-handling decisions, and supported ecological observations without turning associations into causal claims.

### T13 - Consolidate the research outputs

- [ ] **TODO** | Dependencies: completed relevant modeling/analysis milestones.
- Update `README.md` with working commands, the new formulations, selected artifacts, and result locations. Keep the accepted PDF unchanged and document methodological corrections separately.
- Summarize which hypotheses were supported, which were not, remaining solver/coverage limitations, and the strongest findings for the next paper. Keep historical and new results distinguishable.
- Before presenting the full critical-boundary characterization as proven, require T07's analytical completion; otherwise state the unresolved mathematical claim explicitly.
- Make this backlog reflect the actual remaining work; add follow-up tasks only when results or the user justify them.
- **Done when:** another session can reproduce the selected analyses using the recorded environment, data, configurations, and commands, and can identify the next unfinished research step.

## Progress and decisions

- **2026-10-02:** Completed T08A: shared training CLI `python -m src.debbirth.train`, with `cli_invocation.json` records. Per the user: hyperparameters stay in configs, and a config's outdir is provenance (fresh run directory unless `--outdir`). Added NN `run_name` for parity with GP. Small GP/NN boundary runs reproduced exactly from their saved configs.
- **2026-10-02:** Completed T06A: shared `ParameterPredictor` with `NNPredictor`/`GPPredictor`, one-pass GP validation, and coordinate-aware plotting (normalized axes, critical curves/surfaces, verified slices). Run: `results/runs/2026-10-02T17-07-29-933074_t06a_validation/`.
- **2026-10-02:** Completed T06: production boundary GP engine (`boundary.py`), normalized/boundary training through `train_gp_classifier`, exact `expression.txt` with original-variable rules; validation run `results/runs/2026-10-02T16-55-47-145588_t06_gp_validation/`. Simplified exports stay with T06D.
- **2026-10-02:** Completed T06C (unweighted defaults; weighted configs kept as `*_balanced`/`*_pos_weight`) and T06B (revised function set and `gp_normalized`/`gp_boundary` configs with `x_b`) before T06, so T06 pilots use final settings. Decided that GP final-program selection stays at the last generation (generations are tuned); added T08C for alternative policies. Moved export-semantics agreement from T06B to the new T06D after reverting an out-of-scope `symbolic.py` change. Validation run: `results/runs/2026-10-02T16-16-03-355538_t06bc_validation/`.
- **2026-09-08:** Added T06B for the revised GP function set from `docs/next_steps.md` and T06C for unweighted classifier defaults at the user's request. Both are prerequisites for final protocol/results (T08/T09), with propagation into JSON-based tuning (T08B). Preserve historical weighted models/function sets for disclosed benchmarks. Planning-only update; no code, experiment configs, or models changed.

- **2026-09-08:** Completed T05 and the NN interfaces needed from T06A. Both variants train/save/reload through the existing NN workflow; selected checkpoint metrics and original-parameter inference were verified in `debbirth`, including exact fresh-process reload and historical NN predictions. See `docs/nn_formulations.md` and the T05 run summary. Preserved practical solver-failure labels, supplied data, archived artifacts and historical tuning; left GP integration, shared plotting, temperature/full tuning and the rest of T06A for subsequent tasks.

- **2026-09-06:** Repository orientation completed and `AGENTS.md` written. User specified conda `debbirth`, minimal test creation, priority for new formulations, and openness to a GP backend change if needed. Created this backlog. No environment verification, new model implementation, or training is claimed complete.
- **2026-09-06:** Added the requested structural improvements to T03/T05/T06/T08 and added T06A for shared inference/evaluation. Recorded source-inspection findings to address: ignored GP split settings, directory creation during config construction, NN inference mode/no-scaling handling, checkpoint selection, repeated prediction passes, and original-coordinate plotting assumptions. Added T08A as a proposed minimal training CLI. No refactor or CLI implementation has been performed.
- **2026-09-06:** Completed T01 and recorded commands/results in `docs/environment_baseline.md`. The conda environment and both archived model loaders work. Approved reads outside the sandbox confirmed the data are intact. Next: T02 normalization/coverage audit.
- **2026-09-06:** Completed T02 with the reproducible audit script, report, source hashes, row mapping, and plots. Normalization preserves 199,989 distinct rows without audited cross-split near-duplicate leakage. Retained all original labels and splits, documented solver-failure/timeout caveats, and deferred T11. Next: T03 shared formulation/data/config work.
- **2026-09-06:** User clarified that solver timeouts/nonconvergence are infeasible for the intended practical screening task. Retained that policy without alternative-solver relabeling. Kept AmP evaluation in T12 and extra distribution-specific evaluations optional. Replaced T07's numerical-reference project with lightweight analytical-contract/implementation work and removed it as a dependency for training/tuning.
- **2026-09-06:** User approved T07 as an explicit mathematical-proof review: assumptions, scaling/integrals, attainable-maturity interval, critical endpoint, lambda_R existence/selection and any necessary uniqueness, and strict boundary treatment. Completion requires analytical justification, not numerical verification. Initial model work can proceed independently, but a claim of a complete proof depends on T07. This chat is now planning-only.
- **2026-09-06:** Completed T03 using the user-selected `full_par` name, JSON experiment settings, and explicit rejection of validation-free training. Shared preparation preserves row identity and maturity offsets; configs no longer create directories and GP settings are reconstructable by named primitives. Direct checks and small GP/NN runs passed in `debbirth`; see `docs/formulations_and_data.md` and `results/runs/t03_validation/summary.json`. No new-model tuning, GP backend migration, solver work, or data relabeling was performed.
- **2026-09-07:** Replaced GP name/value JSON constants with registered names only. Old entries remain readable only when matching the registry; new saves always use names. Direct validation passed, with archived GP inference unchanged.
- **2026-09-07:** Added T08B for base-JSON plus sampled-override tuning integration and made it a dependency of T09. Reopened T03 narrowly for the best-model saving path/provenance regression found during tuning-code inspection. Preserved the completed shared-foundation results. This update changes the backlog only; neither the regression fix nor tuning integration has been implemented or run.
- **2026-09-08:** Completed the T03 regression correction: selected-model saving returns its resolved config/path and retains data provenance. Four save/test-evaluation combinations passed direct checks with a tiny real GP model; saved models reloaded with identical predictions. Ray/HyperOpt are absent, so search orchestration/results were simulated while executing the actual finalization function. No dependencies were installed, and no full search, source-data changes, or archived-artifact changes were made. T03 is DONE; T08B remains TODO.
- **2026-09-08:** Completed T04 with gplearn as the priority and selected backend for all formulations. A small normalized classifier and fixed-offset boundary prototype trained and reloaded successfully; boundary sample-mask, parallelism and alignment checks passed. `docs/gp_backend_decision.md` records the experiment, the private fitness/source dependency, final-selection/parsimony distinction, and the conditional benefits of DEAP/PySR. Production GP integration remains T06; no backend migration, data relabeling or test-set evaluation was performed.
- **2026-09-08:** User accepted the gplearn recommendation and requested a clearer boundary explanation. Expanded `docs/gp_backend_decision.md` with the representation/loss derivation and its mapping to the implemented fitness. Existing `full_par` artifacts/tuning remain reusable without a backend-driven rerun; comparisons must disclose historical protocol differences. Documentation-only follow-up; T04 remains DONE.
