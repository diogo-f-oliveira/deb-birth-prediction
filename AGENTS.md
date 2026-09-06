# Working in this repository

## Purpose and sources

Develop accurate, fast, and interpretable surrogates for birth feasibility in standard Dynamic Energy Budget (DEB) models. Prioritize development of the new normalized and critical-boundary formulations. Keep compatibility with the existing formulations primarily for performance benchmarking and reproducibility of the accepted CONTROLO'26 work; avoid compatibility infrastructure that unnecessarily slows the new research.

Read these sources before making scientific or architectural changes:

- `README.md`: current implementation, data workflow, and archived model usage.
- `docs/Will it be born CONTRLO26 paper.pdf`: accepted paper and historical experimental methodology; the filename is intentional.
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
| Original unconstrained | Score from `(g, k, v_Hb, f)` | Sigmoid of score |
| Normalized unconstrained | Score from `(gamma, k, nu_b)` | Sigmoid of score |
| Critical boundary | `F(gamma, k)`, approximating `log(Psi(gamma, k))` | `sigmoid((F - log(nu_b)) / T)` |

Here `gamma = g/f`, `nu_b = v_Hb/f^3`, and `T > 0` is the temperature (`alpha = 1/T` in the derivation). Use natural logarithms consistently. `F` itself is the log critical maturity; the final classification logit is `(F - log(nu_b))/T`.

For the boundary formulation:

- Only `(gamma, k)` and deterministic functions of them enter the learned boundary. Maturity enters through the fixed subtraction of `log(nu_b)`; do not let the learner combine it arbitrarily with the other inputs.
- `Psi` is the normalized critical maturity. The threshold in original variables is `f^3 * Psi(g/f, k)`.
- Feasibility uses the strict inequality `nu_b < Psi(gamma, k)`. A zero margin is the boundary; document how ties and numerical tolerances are handled in classification.
- Positive temperature changes probability sharpness but leaves the zero-margin boundary unchanged. Tune temperature on validation data. A separately tuned probability threshold can move the decision boundary and must be reported separately.
- Prefer stable log-domain calculations such as `log(nu_b) = log(v_Hb) - 3*log(f)` and numerically stable loss/sigmoid implementations.

The derivation gives analytical constraints: `Psi(gamma, 1) = 1`; growth limits first for `k < 1`; maturation limits first for `k > 1`. For `k < 1`, the boundary uses `Phi(1; gamma, k)`. For `k > 1`, it uses the first admissible maturation-stationary terminal length `lambda_R < 1`, as defined in Eqs. (41)-(42). Numerical experiments are not required to establish a proven mathematical result and cannot replace a missing proof argument. If a mathematical step is implicit, identify and address it analytically when relevant. Use small implementation checks only where useful, for example to catch a sign or normalization error. Do not build a numerical Psi reference or delay training for solver comparisons unless a concrete later experiment needs one.

The accepted paper calls `k*v_Hb < f^3` sufficient but not necessary in Section II. The displayed strict growth and maturation conditions imply this inequality, so it is a necessary screening condition, not by itself a sufficient feasibility test. Preserve the accepted PDF and flag this wording when extending the analysis.

## Research priorities

Follow `docs/next_steps.md`, implementing the part relevant to the user's current request:

1. Audit existing data after normalization: coverage, unique transformed points, equivalent parameter sets, and label disagreements. Reducing feature dimension does not automatically reduce row count or require new simulations. Merely setting `f=1` without transforming `g` and `v_Hb` is incorrect.
2. Add normalized and boundary variants for both GP and NN, retaining the original four-input formulation as a benchmark. The main new comparison is two model families by two new formulations; including the historical formulation gives six combinations.
3. Evaluate a new GP function set without `max`, square root, inversion, or negation, and consider the derived feature `x_b = g/(f+g) = gamma/(1+gamma)`. Keep historical primitives available for old models; treat the new set as an experiment, not an established performance improvement.
4. Train and tune, including boundary temperature, then compare predictive performance, complexity, inference cost, and sample efficiency. The proposed explanation for the NN advantage and the prospect of GP outperforming it remain hypotheses.
5. Explore AmP species through `eta = v_Hb/g^3`, the curve `nu_b = eta*gamma^3`, and margin to infeasibility. For food thresholds solve the boundary intersection before using `f_crit = g/gamma_crit`; check root existence, admissible food range, and any uniqueness assumption.

## Code map and implementation practices

- `src/debbirth/data/`: MATLAB simulation generators; Python schema, loading, splitting, and scaling. Centralize reusable transformations here rather than duplicating notebook formulas.
- `src/debbirth/models/gp/`: configuration, custom primitives/constants, training, tuning, and symbolic/MATLAB export. `calibrate.py` currently performs hyperparameter search; do not confuse it with probability calibration.
- `src/debbirth/models/nn/`: PyTorch architecture, configuration, training, and loading.
- `src/debbirth/evaluate/` and `src/debbirth/plot/`: shared metrics, comparisons, and decision-boundary plots.
- `src/debbirth/utils/results.py`: run-directory and figure-output helpers.
- `notebooks/`: exploration and presentation of results. Put reusable logic in `src/debbirth/` and avoid unrelated notebook/output churn.
- `results/models/DEBBirthGP/` and `results/models/DEBBirthNet/`: historical artifacts. Write new experiments to distinct directories under `results/runs/` or `results/tune/`, which are ignored by Git.

Follow the existing dataclass configuration and module structure. Use explicit feature schemas and `pathlib` paths; avoid new machine-specific absolute paths. Preserve old loading and inference behavior when adding formulations. The archived GP uses unscaled inputs and the NN requires its saved log/standardization scaler. Saved configurations contain historical absolute paths; GP function-object strings are not sufficient to reconstruct training. Changes to protected GP primitives must also be reflected and numerically checked in symbolic and MATLAB exports: algebraic simplification must not silently discard protection semantics.

The existing GP implementation uses `gplearn`. Before implementing the new GP formulations, assess whether its extension points support the required learned expression and fitness, especially the fixed observation-specific `-log(nu_b)` offset outside the evolved boundary tree and temperature handling. Do not assume either that gplearn must be retained or that migration is necessary. If it cannot support the new formulation cleanly, use a more customizable approach such as DEAP or another justified alternative. Base the choice on implementation simplicity, control over evolution/fitness, and research needs. Keep the existing gplearn baseline usable for comparisons without requiring the new models to share its backend.

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

An activated `debbirth` environment is equivalent. Do not silently substitute system Python, a bundled runtime, or a new environment for repository execution. Inspect the existing environment before installing dependencies; do not reinstall `requirements.txt` as a routine setup step. Inspect imports before running optional workflows: GP tuning imports Ray Tune and HyperOpt, which are not listed in the current requirements. Do not assume a dependency installation reproduces the paper's environment without checking versions.

The existing training examples are:

```text
conda run -n debbirth python -m src.debbirth.models.gp.train
conda run -n debbirth python -m src.debbirth.models.nn.train
```

These launch training and save artifacts; use deliberately small configurations for smoke checks. Their example settings are not necessarily the archived paper settings. Notebook relative paths generally assume `notebooks/` as the working directory while imports need the repository root on the Python path.

This is a research repository, not a software package. Tests are usually unnecessary: do not create test files, expand a test suite, or introduce testing infrastructure by default. Prefer a small direct calculation, an existing notebook, a short smoke run, or inspection of experimental results when validation is useful. Add an automated test only when it has clear value for a consequential, otherwise difficult-to-detect error, or when the user requests it.

Choose scientific checks relevant to the change rather than treating these as a mandatory checklist: scaling invariance, the `k=1` boundary, strict boundary behavior, finite extreme-range outputs, decreasing boundary-model feasibility with increasing maturity, or save/load prediction agreement. Verify exported GP expressions against runtime predictions when changing primitives or exports. Report what was actually checked and any limitations. Documentation-only edits need a content/diff check, not code execution or model training.

The existing symbolic-expression tests can be run with `conda run -n debbirth python tests/run_tests.py` when relevant; they are not a required step for every change or an end-to-end scientific validation suite.

Inspect the worktree before editing and preserve unrelated user changes. On Windows/OneDrive, inaccessible directories can appear as deleted in Git; verify accessibility before treating them as actual deletions or restoring files. If Git reports ownership mismatch, use a command-scoped `safe.directory` for this checkout rather than changing global configuration. Report environmental blockers without rewriting data or configuration to conceal them.
