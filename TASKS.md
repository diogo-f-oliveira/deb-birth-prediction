# Development tasks

Last updated: 2026-09-06

## Objective

Develop normalized and critical-boundary birth-feasibility models for both GP and NN, compare them fairly, and use the learned boundary to explore AmP species. Retain the original four-input models for benchmarking. The scientific motivation is in `docs/next_steps.md`; the equations are in `docs/birth_equations.md`; working conventions are in `AGENTS.md`.

## How to maintain this file

- Read this file at the start of development work. Follow the user's current request; when asked to continue generally, take the first ready task in the order below.
- Keep at most one task marked `IN PROGRESS`. Independent work can proceed when a dependency is blocked, but record why the order changed.
- Task states are `TODO`, `IN PROGRESS`, `BLOCKED`, `DONE`, and `DEFERRED`. Keep the checkbox unchecked until `DONE`. For blocked/deferred tasks, state the reason and the next action that would resolve it.
- Update the affected task before ending a work session: record the result, relevant file/run paths, actual checks, and remaining work. A task is done only when its completion criterion is met; a plan or unexecuted script is not a result.
- Add a brief dated entry to the progress log for completed milestones or decisions that change the plan. Keep detailed results in linked notes or run artifacts rather than growing the log into a transcript.
- Revise dependencies and completion criteria when findings change the approach. Record decisions and their rationale; do not silently drop work or overwrite earlier experimental results.
- Use conda `debbirth` for code execution. Prefer direct scientific checks and small experiments over new test files. Do not create package infrastructure or a general experiment framework just to complete this backlog.
- This file is a development plan, not a request to start every experiment now. Execute the scope of the active user request; no background scheduling is implied.

**Current task:** None. T01 and T02 are complete; model implementation has not started.

**Next action:** T03, separate formulations, data preparation, and configuration. Reuse existing splits; see `docs/normalized_data_audit.md`.

## Model comparison

| Formulation | GP | NN | Role |
| --- | --- | --- | --- |
| Original: `(g, k, v_Hb, f)` | Existing gplearn classifier | Existing DEBBirthNet | Historical benchmark |
| Normalized: `(gamma, k, nu_b)` | New unconstrained score | New unconstrained score | Main comparison |
| Boundary: `(gamma, k)` | Learn `F = log(Psi)` | Learn `F = log(Psi)` | Main comparison |

Use `gamma = g/f`, `nu_b = v_Hb/f^3`, and boundary probability `sigmoid((F - log(nu_b))/T)`, with `T > 0`. The four new models are the priority. Historical artifacts provide context; claims about the effect of formulation require comparable data, preprocessing, and training/tuning budgets, or an explicit explanation of the differences.

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

- [ ] **TODO** | Dependencies: T02.
- Extend the data/schema path with explicit formulation names and feature order. Keep the original `(g, k, v_Hb, f)` behavior intact.
- Introduce a small formulation module (for example `src/debbirth/formulations.py`) that explicitly separates learned output, boundary margin, and probability. Share the mathematical definitions across GP and NN without forcing their training loops or array backends into one abstraction.
- Separate CSV/split loading and shared feature preparation from NN tensor conversion, device placement, and batching. Shared data preparation should not import model training configurations. Carry labels, source-row identity, and boundary offsets together through selection and shuffling.
- Centralize normalization and `log_nu_b = log(v_Hb) - 3*log(f)`. For boundary training, supply `(gamma, k)` to the learned model and carry `log_nu_b` separately for the fixed comparison. Expose `x_b = gamma/(1+gamma)` as an optional derived feature.
- Specify invalid-input behavior. Apply any learned standardization using training data only. Save formulation and transformation settings with each model; avoid copying transformations across notebooks and training modules.
- Fix split configuration inconsistencies: GP currently ignores `cfg.data_splits`, and trainers assume a validation split. Implement or explicitly reject unsupported modes; never silently substitute another split policy.
- Keep configuration construction/loading free of directory creation. Resolve shared paths in one place and create run directories only when starting/saving a run. Store GP primitive identifiers and constant values explicitly instead of relying on function-object strings from `default=str`.
- Move reusable experiment settings out of hardcoded training-module `__main__` blocks into a small `experiments/` directory. Keep dataclass configs and separate family-specific trainers; avoid a broad framework or unrelated file moves.
- **Done when:** both new representations can be obtained through shared preparation, a direct calculation confirms equivalent original inputs yield identical normalized inputs, the original input path still works, and config loading has no output-directory side effects. Split behavior and settings needed to reconstruct a run are explicit.

### T04 - Choose the GP implementation approach

- [ ] **TODO** | Dependencies: T01; T03 for a small end-to-end prototype.
- Inspect the installed gplearn implementation and current wrapper. Determine separately whether it supports the normalized classifier and the boundary classifier cleanly.
- Prototype the boundary fitness: a tree receives only `(gamma, k)` or their permitted transforms; the fitness uses row-aligned labels and `log_nu_b` in weighted BCE of `(F - log_nu_b)/T`, plus parsimony. Check how subsampling, parallel evaluation, prediction, and saving preserve this alignment.
- Reject approaches that allow maturity into the evolved tree, hide it in the target labels, or depend on fragile implicit row order. Evaluate a small custom extension versus DEAP or another suitable backend if gplearn's interfaces are insufficient.
- **Done when:** a short decision note names the chosen backend(s), explains the tradeoff, and links a working minimal boundary-fitness experiment. Preserve the existing gplearn benchmark without building a general backend abstraction.

### T05 - Implement normalized and boundary neural networks

- [ ] **TODO** | Dependencies: T03. Can proceed while the GP decision is unresolved.
- Extend the existing NN configuration/training path to support the normalized three-input score and the two-input `F(gamma, k)` output. Compute boundary logits outside the learned network using the fixed maturity offset and positive temperature.
- Keep input preprocessing configurable and recorded. The final `F` output must be unrestricted in sign; positivity applies to `Psi = exp(F)`, not to `F`.
- Save and load formulation, feature settings, scaler, weights, and temperature. Expose the boundary margin and critical maturity for analysis as well as birth probabilities.
- Fix the no-scaling path so it still converts inputs to tensors and supports saving/loading with no scaler. Return inference-loaded models in evaluation mode so dropout is inactive by default.
- Make checkpoint selection explicit: final epoch or best validation checkpoint, with the metric and selected epoch recorded. Save metrics corresponding to the saved weights. Keep historical reproduction settings available.
- Run a small training experiment for each new formulation. Check finite loss/probabilities, prediction agreement after reload, and that the boundary probability decreases with increasing maturity at fixed `(gamma, k)`.
- **Done when:** both variants train, save, reload, and predict through the existing workflow, with run paths and checks recorded. This task does not require full tuning.

### T06 - Implement normalized and boundary GP models

- [ ] **TODO** | Dependencies: T03 and T04.
- Implement both formulations using the selected approach. For the boundary model, evolve `F(gamma, k)` with the fixed offset outside the tree and a documented positive training temperature.
- Define the new primitive set without `max`, square root, inversion, or negation; retain historical primitive definitions for old artifacts. Include optional `x_b`; document protected operations and constant choices.
- Save sufficient settings and the raw expression to reproduce inference. Export readable expressions with the actual feature names and full original-variable feasibility rule, including normalization and `exp(F)` where applicable.
- Keep historical gplearn class/module import paths and primitive definitions available for loading archived artifacts. Confine backend-specific evolution and serialization to the GP implementation; analysis code should not need to access gplearn private attributes.
- Run a small evolution for each formulation. Confirm finite fitness, row alignment, save/load prediction agreement, and numerical agreement between the executable expression and any export used for inference.
- **Done when:** both variants train and produce reusable symbolic models, with short-run artifacts and an example boundary rule. Do not claim simplified expressions preserve protected semantics without checking.

### T06A - Unify inference and simplify evaluation

- [ ] **TODO** | Dependencies: T03, T05, and T06. Implement alongside those tasks where useful.
- Add a lightweight predictor that bundles the learned model, formulation, feature order, preprocessing, and temperature. Its public inference entry point accepts the original parameters for every formulation; internal prepared-input paths must be explicit to prevent applying transformations twice.
- Return a consistent positive-class probability shape. For boundary models, also expose signed margin and normalized/original critical maturity. Keep low-level model APIs available for training and historical use.
- Separate metric computation from model execution: compute metrics from labels, probabilities, and an explicit decision rule. Collect NN predictions and loss in one batched pass instead of collecting the full input dataset and predicting again. Avoid separate model calls for labels and probabilities.
- Add MCC and the probability metrics needed for the temperature experiments. Keep strict zero-margin classification distinct from any tuned operating threshold; use the same rules across model families.
- Adapt plotting so normalized axes and critical surfaces have correct labels and reference bounds. Remove the assumption that every plot has original-variable `f` and `k` annotations taken from the first row; check slice assumptions where they are needed.
- **Done when:** a small shared input set can be passed through each supported formulation without caller-managed scaling, predictions survive saving/loading, and shared evaluation/plotting works for both original and normalized coordinates. Record direct checks rather than creating a new test suite.

### T07 - Establish a numerical reference for the new boundary

- [ ] **TODO** | Dependencies: T02; T03 for shared notation/transforms.
- Review the critical-boundary construction, including the interpretation of the first stationary maturation point for `k > 1`. Separate analytically established identities from claims still needing numerical support.
- Start with exact `k=1` checks and a modest set of reliable existing solver cases. Check scaling invariance at several food levels when solver access permits.
- Use T02's audit to select checks: existing k=1 grids, sparse gamma extremes, and unsuccessful points inside the necessary bound. Treat timeout `lb=tb=0` values as placeholders rather than numerical solutions; see `docs/normalized_data_audit.md`.
- Where needed, implement a small numerical evaluator of `Phi`/`Psi` from the derivation: handle the infinite integration endpoint, convergence, and the first admissible `lambda_R` root explicitly. Check sensitivity to numerical settings and compare with an independent birth solver where available.
- **Done when:** a reference note and reproducible calculations cover all three `k` regimes and document discrepancies and tolerances. If external solver access prevents independent comparison, retain that limitation and avoid presenting the derived evaluator as independently validated ground truth.
- **Scope:** this supports scientific boundary evaluation; it need not delay initial model prototypes or become a large dataset-generation project.

### T08 - Finalize the experiment and temperature protocol

- [ ] **TODO** | Dependencies: T02, T05, and T06 pilot runs.
- Write the experiment matrix, matched split/subset identities, preprocessing choices, weighting, selection metrics, seed list, and tuning budgets before full training. Use pilot runtime to set practical budgets. Include repeated training seeds and distinguish archived benchmarks from models retrained under the new protocol.
- Use validation macro-F1 for classification model selection unless a documented research reason favors another metric. Report MCC and per-class errors as well. Reserve test results for the finalized comparison.
- Separate training temperature from post-training calibration. If training temperature is searched, fit each candidate on training data and select using validation data. If fitting a post-training temperature with fixed `F`, minimize validation log loss (state weighting); macro-F1 at the zero-margin threshold cannot select temperature because positive temperature does not move that boundary.
- State whether any extra operating threshold is tuned, and report it separately from the canonical strict `margin > 0` decision. Do not confuse class-weighted scores with calibrated probabilities under the original class prevalence.
- Record whether NN comparison uses the last epoch or a validation-selected checkpoint. Save the resolved experiment settings and selected epoch with each run so the comparison can be reproduced.
- **Done when:** a saved, runnable experiment specification defines the four main models, historical comparison, budgets, selection rules, and temperature treatment without test-data tuning.

### T08A - Add a small shared training CLI

- [ ] **TODO (proposed)** | Dependencies: T03, T05, and T06; use T08 for full-experiment configurations.
- Provide one repository-root module entry point for training, backed by the same Python functions used in notebooks. A proposed interface is `python -m src.debbirth.train`; this module and the example config below do not exist yet.
- Use a small standard-library argument parser. Expose model family (`gp`/`nn`), formulation (`original`/`normalized`/`boundary`), an experiment config file, and a few common overrides such as seed, data directory, output directory, and device/workers where applicable. Keep detailed architecture, primitive sets, and search settings in the config file rather than creating a flag for every parameter.
- Define precedence explicitly: defaults, then config, then explicitly supplied CLI overrides. Validate family/formulation compatibility and inputs before expensive work. Save the fully resolved config and invocation with the run and print the output directory and validation summary.
- A training invocation should fit and validate one run. Keep held-out test evaluation explicit and separate from routine training/tuning. Leave experiment grids and hyperparameter search to simple scripts calling the same training functions initially.
- Preserve callable training functions and historical entry points where practical. Do not require installation as a package, add orchestration infrastructure, or implement CLI-only training logic.
- **Done when:** help is useful, a small NN run and a small GP run can each be launched and reproduced from a saved configuration through the CLI, and invalid combinations fail clearly before training. Document actual commands in `README.md`.

Proposed command, to be documented as working only after implementation:

```text
conda run -n debbirth python -m src.debbirth.train --model nn --formulation boundary --config experiments/nn_boundary.json --seed 42
```

### T09 - Train, tune, and compare the models

- [ ] **TODO** | Dependencies: T08 and T06A; use T08A for command-line execution once implemented; T07 for scientific boundary comparisons.
- Run the agreed experiments in distinct run directories. Record configurations, seeds, timing, model size/expression complexity, training history, and selected artifacts. Keep interrupted or failed runs identifiable.
- Freeze model choices using validation results, then evaluate the test split. Produce a consolidated table with macro-F1, MCC, per-class precision/recall, confusion counts, AUROC/AP, and probability quality where relevant.
- Plot boundary slices and `F`/`Psi` across the three maintenance regimes. Separate agreement with dataset labels from agreement with the numerical reference; label extrapolation and unresolved solver cases.
- Measure both single-sample and batch inference cost under a recorded hardware/timing protocol. Include preprocessing and boundary comparison in the end-to-end surrogate timing.
- **Done when:** a reproducible comparison notebook/report links selected runs, tables, plots, uncertainty across seeds, and limitations. Performance conclusions must follow the results rather than the expected GP advantage.

### T10 - Measure sample efficiency and targeted ablations

- [ ] **TODO** | Dependencies: T09.
- Use nested stratified subsets of the training split, shared across models and seeds. Initial candidate fractions are 1%, 5%, 10%, 25%, 50%, and 100%; adjust after checking class counts and runtime, and record the final choice.
- Keep validation/test membership fixed. State whether hyperparameters remain fixed from T09 or are retuned with matched budgets; these answer different questions. Do not use test learning curves to tune the models.
- Plot performance versus training rows with variation across seeds. Use a small number of controlled ablations to investigate log preprocessing, `x_b`, or the revised primitive set; do not confound backend, formulation, and preprocessing changes when attributing gains.
- **Done when:** learning curves and an interpretation distinguish sample-efficiency evidence from preprocessing/tuning effects, with linked run identities.

### T11 - Extend simulation generation only where justified

- [ ] **DEFERRED** | Dependencies: T02 identifies gaps; T07 informs boundary targeting.
- **Decision (2026-09-06):** T02 supports reuse of the existing 199,989 rows for initial classification experiments. Revisit after T07/T09 reveal a specific boundary/reference gap or T12 establishes out-of-domain species requirements. Do not regenerate merely to remove f as an independent input.
- If transformed existing data are sufficient, mark this task `DEFERRED` with that finding. Revisit if boundary diagnostics or AmP coverage expose gaps.
- Otherwise, adapt a generator to sample normalized `(gamma, k, nu_b)`, passing `(g=gamma, k, v_Hb=nu_b, f=1)` to the existing solver for compatibility. Specify ranges and boundary-focused sampling from the measured gaps.
- Keep new datasets separate and save solver version/settings, timeout, sampling seed, and diagnostic/label policy. Preserve a distinction between unresolved solver outcomes and demonstrated infeasibility.
- **Done when:** either the no-new-data decision is recorded, or a small pilot confirms the new generator and its provenance before the needed dataset is produced. If new data change the benchmark, version the protocol and repeat affected comparisons explicitly.

### T12 - Explore AmP species relative to the learned boundary

- [ ] **TODO** | Dependencies: T09; T07 for interpretation; access to suitable AmP data.
- Identify a versioned AmP source and retain species identity/model type. Check applicability of the standard embryo equations. Separate fitted species parameter sets from the repository's perturbed-parameter simulation dataset.
- Compute `eta = v_Hb/g^3` and plot species in `(eta, k)` space. At explicitly stated maternal food levels, compute `gamma`, `nu_b`, and signed margin `F(gamma,k) - log(nu_b)`.
- Find food-boundary intersections using `log(eta) + 3*log(gamma) = F(gamma,k)`, then `f_crit = g/gamma_crit`. Search an explicit admissible food interval, handle absent/multiple roots, and verify which side is feasible before calling a root the minimum viable food level.
- Plot margin and food thresholds with species metadata where available. Flag out-of-domain species and uncertain boundary estimates; check selected near-boundary cases numerically when possible.
- **Done when:** a reproducible analysis links species provenance, derived values, plots, root-handling decisions, and supported ecological observations without turning associations into causal claims.

### T13 - Consolidate the research outputs

- [ ] **TODO** | Dependencies: completed relevant modeling/analysis milestones.
- Update `README.md` with working commands, the new formulations, selected artifacts, and result locations. Keep the accepted PDF unchanged and document methodological corrections separately.
- Summarize which hypotheses were supported, which were not, remaining solver/coverage limitations, and the strongest findings for the next paper. Keep historical and new results distinguishable.
- Make this backlog reflect the actual remaining work; add follow-up tasks only when results or the user justify them.
- **Done when:** another session can reproduce the selected analyses using the recorded environment, data, configurations, and commands, and can identify the next unfinished research step.

## Progress and decisions

- **2026-09-06:** Repository orientation completed and `AGENTS.md` written. User specified conda `debbirth`, minimal test creation, priority for new formulations, and openness to a GP backend change if needed. Created this backlog. No environment verification, new model implementation, or training is claimed complete.
- **2026-09-06:** Added the requested structural improvements to T03/T05/T06/T08 and added T06A for shared inference/evaluation. Recorded source-inspection findings to address: ignored GP split settings, directory creation during config construction, NN inference mode/no-scaling handling, checkpoint selection, repeated prediction passes, and original-coordinate plotting assumptions. Added T08A as a proposed minimal training CLI. No refactor or CLI implementation has been performed.
- **2026-09-06:** Completed T01 and recorded commands/results in `docs/environment_baseline.md`. The conda environment and both archived model loaders work. Approved reads outside the sandbox confirmed the data are intact. Next: T02 normalization/coverage audit.
- **2026-09-06:** Completed T02 with the reproducible audit script, report, source hashes, row mapping, and plots. Normalization preserves 199,989 distinct rows without audited cross-split near-duplicate leakage. Retained all original labels and splits, documented solver-failure/timeout caveats, and deferred T11. Next: T03 shared formulation/data/config work.
