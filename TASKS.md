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

- [ ] **TODO** | Dependencies: T08 and T06A; use T08A for command-line execution once implemented.
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

- **2026-09-06:** Repository orientation completed and `AGENTS.md` written. User specified conda `debbirth`, minimal test creation, priority for new formulations, and openness to a GP backend change if needed. Created this backlog. No environment verification, new model implementation, or training is claimed complete.
- **2026-09-06:** Added the requested structural improvements to T03/T05/T06/T08 and added T06A for shared inference/evaluation. Recorded source-inspection findings to address: ignored GP split settings, directory creation during config construction, NN inference mode/no-scaling handling, checkpoint selection, repeated prediction passes, and original-coordinate plotting assumptions. Added T08A as a proposed minimal training CLI. No refactor or CLI implementation has been performed.
- **2026-09-06:** Completed T01 and recorded commands/results in `docs/environment_baseline.md`. The conda environment and both archived model loaders work. Approved reads outside the sandbox confirmed the data are intact. Next: T02 normalization/coverage audit.
- **2026-09-06:** Completed T02 with the reproducible audit script, report, source hashes, row mapping, and plots. Normalization preserves 199,989 distinct rows without audited cross-split near-duplicate leakage. Retained all original labels and splits, documented solver-failure/timeout caveats, and deferred T11. Next: T03 shared formulation/data/config work.
- **2026-09-06:** User clarified that solver timeouts/nonconvergence are infeasible for the intended practical screening task. Retained that policy without alternative-solver relabeling. Kept AmP evaluation in T12 and extra distribution-specific evaluations optional. Replaced T07's numerical-reference project with lightweight analytical-contract/implementation work and removed it as a dependency for training/tuning.
- **2026-09-06:** User approved T07 as an explicit mathematical-proof review: assumptions, scaling/integrals, attainable-maturity interval, critical endpoint, lambda_R existence/selection and any necessary uniqueness, and strict boundary treatment. Completion requires analytical justification, not numerical verification. Initial model work can proceed independently, but a claim of a complete proof depends on T07. This chat is now planning-only.
