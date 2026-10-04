# Working in this repository

## Purpose and sources

Develop accurate, fast, and interpretable surrogates for birth feasibility in standard Dynamic Energy Budget (DEB) models. Prioritize development of the new normalized and critical-boundary formulations. Keep compatibility with the existing formulations primarily for performance benchmarking and reproducibility of the accepted CONTROLO'26 work; avoid compatibility infrastructure that unnecessarily slows the new research.

Read these sources before making scientific or architectural changes:

- `README.md`: current implementation, data workflow, and archived model usage.
- `docs/Will it be born CONTROLO26 paper.pdf`: accepted paper and historical experimental methodology.
- `docs/birth_equations.md`: derivation and notation for the new representations.
- `docs/next_steps.md`: the research direction and proposed experiments.
- `TASKS.md`: the actionable backlog, dependencies, completion criteria, and current progress. Read it at the start of development work and update the relevant task and progress log before ending a work session. Follow the active user request rather than automatically executing the entire backlog.

The paper describes the published baseline; the two Markdown research notes describe subsequent work, not functionality already implemented. Check the source code and saved configurations for actual behavior. If a derivation, implementation, or document disagrees, explain the discrepancy rather than silently treating them as equivalent. Keep this guide current when interfaces or experimental conventions change.

## Scientific contract

- The target is `reached_birth`: `1` / `True` means feasible according to the recorded labels. `success`, timeouts, execution errors, and solver convergence are diagnostics, not interchangeable targets.
- The intended application is practical feasibility screening: the user explicitly treats get_lb2 timeouts/nonconvergence within the configured budget as infeasible. Retain these negative labels; do not launch alternative-solver retries or relabeling to recover mathematically feasible points by default. Preserve diagnostics to explain outcomes. Exact scaling and the critical surface describe the DEB equations, whereas a numerical solver's failure pattern need not obey those identities exactly.
- Archived model inputs are ordered `(g, k, v_Hb, f)`, regardless of the order used in prose or equations. Preserve that contract when loading historical artifacts.
- Work on finite positive `g`, `k`, `f`, and `v_Hb` for the logarithmic formulations. Handle invalid inputs explicitly; do not silently clip physical parameters into the domain. Do not assume `k <= 1`.
- `v_Hb` denotes the original scaled maturity at birth, not its food-normalized counterpart. Use explicit names for transformed quantities and document feature order.

The representations to compare are:

| Formulation | Learned function | Classification |
| --- | --- | --- |
| Full-parameter unconstrained (`full_par`) | Score from `(g, k, v_Hb, f)` | Sigmoid of score |
| Normalized unconstrained | Score from `(gamma, k, nu_b)` | Sigmoid of score |
| Critical boundary | `F(gamma, k)`, approximating `log(Psi(gamma, k))` | `sigmoid((F - log(nu_b)) / T)` |

Here `gamma = g/f`, `nu_b = v_Hb/f^3`, and `T > 0` is the temperature (`alpha = 1/T` in the derivation). Use natural logarithms consistently. `F` itself is the log critical maturity; the final classification logit is `(F - log(nu_b))/T`.

For the boundary formulation:

- Only `(gamma, k)` and deterministic functions of them enter the learned boundary. Maturity enters through the fixed subtraction of `log(nu_b)`; do not let the learner combine it arbitrarily with the other inputs.
- `Psi` is the normalized critical maturity. The threshold in original variables is `f^3 * Psi(g/f, k)`.
- Feasibility uses the strict inequality `nu_b < Psi(gamma, k)`. A zero margin is the boundary; document how ties and numerical tolerances are handled in classification.
- Positive temperature changes probability sharpness but leaves the zero-margin boundary unchanged. Tune temperature on validation data. A separately tuned probability threshold can move the decision boundary and must be reported separately.
- Prefer stable log-domain calculations such as `log(nu_b) = log(v_Hb) - 3*log(f)` and numerically stable loss/sigmoid implementations.

The derivation gives analytical constraints: `Psi(gamma, 1) = 1`; growth limits first for `k < 1`; maturation limits first for `k > 1`. For `k < 1`, the boundary uses `Phi(1; gamma, k)`. For `k > 1`, it uses the first admissible maturation-stationary terminal length `lambda_R < 1`, as defined in Eqs. (41)-(42). T07 (see the proof section of `docs/birth_equations.md`) proves `nu_b < Psi` in both directions for all `k > 0`, with strict inequality. For `k > 1`, `lambda_R` is the unique zero of the maturation rate at birth (Lemma C). Proven bounds are `1 < Psi < 1/k` for `k < 1` and `Psi < 1/k < 1` for `k > 1`, so `F = log(Psi)` has the sign of `1 - k` and `F <= -log(k)`. For `k > 1`, `Psi > lambda_low^3` with `lambda_low = x_b/(k - 1 + x_b)`, and `Psi -> min(1, k^-3)` as `gamma -> inf`; `Psi > k^-3` for every gamma is only conjectured. Numerical experiments are not required to establish a proven mathematical result and cannot replace a missing proof argument. If a mathematical step is implicit, identify and address it analytically when relevant. Use small implementation checks only where useful, for example to catch a sign or normalization error. Do not build a numerical Psi reference or delay training for solver comparisons unless a concrete later experiment needs one.

The accepted paper's Section II wording on `k*v_Hb < f^3` ("sufficient but not necessary") is a known typo for "necessary but not sufficient"; do not flag it.

## Research priorities

Follow `docs/next_steps.md`, implementing the part relevant to the user's current request:

1. Audit existing data after normalization: coverage, unique transformed points, equivalent parameter sets, and label disagreements. Reducing feature dimension does not automatically reduce row count or require new simulations. Merely setting `f=1` without transforming `g` and `v_Hb` is incorrect.
2. Add normalized and boundary variants for both GP and NN, retaining the original four-input formulation as a benchmark. The main new comparison is two model families by two new formulations; including the historical formulation gives six combinations.
3. Evaluate a new GP function set without `max`, square root, inversion, or negation, and consider the derived feature `x_b = g/(f+g) = gamma/(1+gamma)`. Keep historical primitives available for old models; treat the new set as an experiment, not an established performance improvement.
4. Train and tune, including boundary temperature, then compare predictive performance, complexity, inference cost, and sample efficiency. The proposed explanation for the NN advantage and the prospect of GP outperforming it remain hypotheses.
5. Explore AmP species through `eta = v_Hb/g^3`, the curve `nu_b = eta*gamma^3`, and margin to infeasibility. For food thresholds solve the boundary intersection before using `f_crit = g/gamma_crit`; check root existence, admissible food range, and any uniqueness assumption.

## Code map and implementation practices

- `src/debbirth/data/`: MATLAB simulation generators; Python schema, loading, splitting, and scaling. Centralize reusable transformations here rather than duplicating notebook formulas.
- `src/debbirth/formulations.py`: learned-output, boundary-margin, logit, and strict decision semantics. Formulation identifiers are `full_par`, `normalized`, and `boundary`.
- `src/debbirth/models/gp/`: configuration, custom primitives/constants, training, and symbolic/MATLAB export.
- `src/debbirth/tuning.py` and `experiments/tune_{gp,nn}.py`: JSON-based hyperparameter search for both families (T08B), with search spaces in the scripts. It replaced the former GP tuner `calibrate.py`. Hyperparameter search is not probability calibration.
- `src/debbirth/models/nn/`: PyTorch architecture, configuration, training, and loading.
- `src/debbirth/evaluate/` and `src/debbirth/plot/`: shared metrics, comparisons, and decision-boundary plots.
- `src/debbirth/utils/results.py`: run-directory and figure-output helpers.
- `notebooks/`: exploration and presentation of results. Put reusable logic in `src/debbirth/` and avoid unrelated notebook/output churn.
- `results/models/DEBBirthGP/` and `results/models/DEBBirthNet/`: historical artifacts. Write new experiments to distinct directories under `results/runs/` or `results/tune/`, which are ignored by Git.

Follow the existing dataclass configuration and module structure. Use explicit feature schemas and `pathlib` paths; avoid new machine-specific absolute paths. Preserve old loading and inference behavior when adding formulations. The archived GP uses unscaled inputs and the NN requires its saved log/standardization scaler. Saved configurations contain historical absolute paths; GP function-object strings are not sufficient to reconstruct training. Changes to protected GP primitives must also be reflected and numerically checked in symbolic and MATLAB exports: algebraic simplification must not silently discard protection semantics.

T03 shared preparation returns row-aligned `PreparedSplit` records; select/shuffle these records rather than independently indexing labels or offsets. CSV loading/preparation must not import model configs or Torch. Family adapters handle arrays/tensors and batching. Config construction/loading creates no directories; relative paths resolve against the repository root. Saved trainers return the resolved `train_config` and `outdir`, leaving an input config's `outdir=None` unchanged. GP configs serialize registered primitive and constant names; constant values are defined in `src/debbirth/models/gp/constants.py`. See `docs/formulations_and_data.md` for these interfaces and current limitations.

T05 implements both NN variants. `DEBBirthBoundaryNet.forward` returns unrestricted F; its probability/prediction methods require an explicit unscaled maturity offset. `NNPredictor` accepts original parameters and provides an explicit unscaled `PreparedSplit` path. Loading returns evaluation-mode models. `checkpoint_selection` defaults to historical `final_epoch`; `best_val_loss` restores the earliest minimum-validation-loss epoch and saves matching metrics. NN BCE is unweighted by default; with explicit `use_pos_weight`, it averages weighted terms over row count and retains training-derived positive weights on validation. See `docs/nn_formulations.md` for inference, strict decisions, artifact conventions and checks. T06A adds `GPPredictor` with the same interface through the shared `evaluate/predictor.py` base, one-pass GP validation, and coordinate-aware plotting (`slice_grid`, reference bounds, critical curves/surfaces); see `docs/gp_formulations.md` and `docs/formulations_and_data.md`.

T04 selected `gplearn` for all GP formulations; see `docs/gp_backend_decision.md`. The normalized classifier uses the existing wrapper. The separate `boundary_prototype.py` uses a SymbolicRegressor evolution engine with binary labels and dataset-bound fixed-offset BCE, preserving full-row/sample-mask alignment through an audited source contract. T06 integrates it as `boundary.py` (`fit_gp_boundary`, `GPBoundaryModel`) through `train_gp_classifier`; see `docs/gp_formulations.md`. Do not pass maturity or row IDs into the tree, encode offsets in labels, or reuse a fit-bound metric on reordered/new data. gplearn's parsimony affects tournaments, while final-program selection uses last-generation raw loss; document the selection policy and loss normalization explicitly. Preserve historical loading and protected primitives. Consider DEAP/PySR if concrete later needs justify additional control or constant optimization; migration is not required for the new formulations. Final-program selection stays at gplearn's last generation; the number of generations is the tuned quantity, analogous to final-epoch NN checkpoints.

T06C makes classifier losses unweighted by default for GP (`class_weights=None`) and NN (`use_pos_weight=False`); weighting is an explicit opt-in, with the old weighted configs kept as `experiments/*_balanced.json` / `*_pos_weight.json`. T06B adds `REVISED_FUNCTION_SET` (`"revised"`: `add, sub, mul, pdiv, plog, min, cbrt, square, cube, atan`) and `experiments/gp_normalized.json`/`gp_boundary.json` with `x_b`. The existing `symbolic.py` export drops protected semantics; see T06D before using exports for inference. See `docs/formulations_and_data.md`.

## Data and experimental reproducibility

- Preserve supplied raw data, splits, and archived models unless changing them is part of the requested work. The preprocessing notebook overwrites processed CSVs, including splits; do not execute it as a setup check.
- The current preprocessing source is `data/raw/sample4D_lhs_N_200000_g_1em3_1e2_k_m4_1_f_0p1_1_ddec_1_getlb2.csv`. The documented split is stratified 80/10/10 with seed 42, after filtering nonpositive execution times and missing/nonfinite inputs.
- Solver failures and six-second timeouts are intentionally treated as infeasible for the practical screening target, not as a default label-cleaning backlog. Keep that policy, diagnostics, and solver budget explicit. Do not claim the operational failure boundary is mathematically identical to the DEB critical surface.
- Keep `get_lb` and `get_lb2` dataset provenance distinct. MATLAB regeneration requires external DEBtool routines, relevant toolboxes, and AmP data where applicable. Record solver version, settings, timeout, sampling ranges, seed, and label policy for new data.
- Use matching split membership and training subsets across formulations. Fit scalers and training weights on training data only; tune hyperparameters, temperature, and operating thresholds on validation data; reserve test data for final evaluation.
- Check whether normalization or augmentation creates equivalent observations across splits. For species generalization, account for related perturbations from the same species when defining splits.
- Record formulation, feature order/transforms, data identity, seed, configuration, dependency versions, and code revision with results. Save scalers, temperature, thresholds, model state or expression, and metrics needed to reproduce inference.
- Report per-class errors, macro-F1, MCC, and ranking metrics as appropriate, not accuracy alone. Distinguish falsely rejecting a feasible set from falsely accepting an infeasible one. Include boundary slices across `k < 1`, `k = 1`, and `k > 1`, and identify extrapolation. Compare repeated seeds and matched sampling budgets for performance and sample-efficiency claims.
- Use the generated dataset for the main model comparison. Broad-domain versus boundary-focused evaluation is an optional diagnostic, not a prerequisite or a mandatory new benchmark. Keep evaluation on the strongly feasibility-skewed AmP population in T12; do not make an AmP-shaped training distribution a default requirement.

## Setup and validation

Use the existing conda environment `debbirth` for running repository code, notebooks, training, and any checks. Run commands from the repository root, for example:

```text
conda run -n debbirth python -c "import sys; print(sys.executable)"
```

An activated `debbirth` environment is equivalent. Do not silently substitute system Python, a bundled runtime, or a new environment for repository execution. Inspect the existing environment before installing dependencies; do not reinstall `requirements.txt` as a routine setup step. Hyperparameter tuning needs Ray Tune 2.58 and HyperOpt 0.3; both are installed in `debbirth` and listed in `requirements.txt`. Do not assume a dependency installation reproduces the paper's environment without checking versions.

The existing training examples are:

```text
conda run -n debbirth python -m src.debbirth.models.gp.train
conda run -n debbirth python -m src.debbirth.models.nn.train
```

These load `experiments/gp_full_par.json` and `experiments/nn_full_par.json`, train, validate, and save artifacts; use deliberately small configurations for smoke checks. Their example settings are not necessarily the archived paper settings. Both trainers require `train_val_test`; `train_test` is an explicit loader-only merge. NN and GP boundary training are implemented (`experiments/gp_normalized.json`, `experiments/gp_boundary.json`; small GP validation runs use `conda run -n debbirth python -m experiments.validate_gp_formulations`). Untuned NN examples are `experiments/nn_normalized.json` and `experiments/nn_boundary.json`; small validation runs use `conda run -n debbirth python -m experiments.validate_nn_formulations`. Notebook relative paths generally assume `notebooks/` as the working directory while imports need the repository root on the Python path.

The shared training CLI (T08A) is `conda run -n debbirth python -m src.debbirth.train --model {gp,nn} --formulation {full_par,normalized,boundary} [--config ...]`. It is a thin wrapper around `train_gp_classifier`/`train_net`. Keep training logic in those functions.
- **Overrides.** Only runtime and identity settings: seed, data directory, outdir, run name, workers, and NN device. Hyperparameters, including temperature, stay in config JSON.
- **Output directory.** A config's `outdir` is ignored unless `--outdir` is passed, so a saved `train_*_config.json` reproduces a run in a new directory.
- **No save.** `--no-save` runs in memory: progress is printed and nothing is written (no directory, `progress.csv`, model or invocation record).
- **Fail-fast checks.** Invalid family/config/formulation combinations fail before data loading.
- **Test data.** The CLI never evaluates the test split.

T08D: both trainers create the run directory before training and stream per-step validation to `progress.csv`, one row per GP generation or NN epoch. The format is shared, with `train_loss` (the optimized loss), an unweighted logit-BCE `val_bce` and the `BinaryMetrics` fields; the printout is an aligned table. Each GP row scores the generation's lowest raw-loss program through `GPPredictor`. GP uses a callback through gplearn's `_verbose_reporter` (`gp/generation_hook.py`), which reads the population and must not change evolution. `progress_every` (default 1; 0 off) is in both configs. See `docs/usage_guide.md` section 5.7.

NN configs, like GP configs, accept `run_name`. The NN default run directory suffix is `DEBBirthNet`, or `DEBBirthBoundaryNet` for boundary runs.

This is a research repository, not a software package. Tests are usually unnecessary: do not create test files, expand a test suite, or introduce testing infrastructure by default. Prefer a small direct calculation, an existing notebook, a short smoke run, or inspection of experimental results when validation is useful. Add an automated test only when it has clear value for a consequential, otherwise difficult-to-detect error, or when the user requests it.

Choose scientific checks relevant to the change rather than treating these as a mandatory checklist: scaling invariance, the `k=1` boundary, strict boundary behavior, finite extreme-range outputs, decreasing boundary-model feasibility with increasing maturity, or save/load prediction agreement. Verify exported GP expressions against runtime predictions when changing primitives or exports. Report what was actually checked and any limitations. Documentation-only edits need a content/diff check, not code execution or model training.

The existing symbolic-expression tests can be run with `conda run -n debbirth python tests/run_tests.py` when relevant; they are not a required step for every change or an end-to-end scientific validation suite.

Inspect the worktree before editing and preserve unrelated user changes. On Windows/OneDrive, inaccessible directories can appear as deleted in Git; verify accessibility before treating them as actual deletions or restoring files. If Git reports ownership mismatch, use a command-scoped `safe.directory` for this checkout rather than changing global configuration. Report environmental blockers without rewriting data or configuration to conceal them.
