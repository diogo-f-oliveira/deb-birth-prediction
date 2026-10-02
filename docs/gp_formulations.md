# GP formulations: interfaces and validation (T06)

All three GP formulations train through `train_gp_classifier` in `src/debbirth/models/gp/train.py`, using gplearn as decided in [T04](gp_backend_decision.md). The new experiment configs are `experiments/gp_normalized.json` and `experiments/gp_boundary.json`. Both use the revised function set, `x_b` and unweighted loss ([formulations and data](formulations_and_data.md)). Their evolution settings are untuned examples.

## Normalized and full-parameter classifiers

`full_par` and `normalized` use the existing `DEBBirthSymbolicClassifier` wrapper. It evolves a score S from the prepared features, which are unscaled, `(gamma, k, nu_b[, x_b])` for normalized, with named constants appended as constant columns. The probability is `sigmoid(S)`, the loss is gplearn's log loss, and the decision is gplearn's argmax, so `sigmoid(S) > 0.5` and ties are infeasible. Historical artifacts and import paths are unchanged.

## Boundary engine

`src/debbirth/models/gp/boundary.py` evolves `F(gamma, k[, x_b]) = log(Psi)` with gplearn's `SymbolicRegressor`, whose raw output is F. The fitness is the T04 fixed-offset BCE:

```text
L = sum_i w_i * BCE(y_i, sigmoid((F_i - log_nu_b_i) / T)) / sum_i w_i     over active rows
```

- **Inputs:** only `gamma`, `k` and optional `x_b` are tree terminals, in the configured order. `log_nu_b` is bound into the metric for this fit and never enters the tree or the labels. The T04 backend contract (gplearn 0.4.3 source fingerprints) is checked before every fit, and the bound metric rejects another split's labels.
- **Primitives and constants:** the configured registered primitives (the revised set in the shipped config) and the named constants are used, appended as constant columns exactly as in the classifier. `const_range=None`, so there are no ephemeral random constants, and the program text has no rounded numbers.
- **Weights:** `class_weights=None` gives unit weights, so the loss is the ordinary mean BCE. `"balanced"` or a class mapping are explicit opt-ins computed on the training rows. `max_samples` is 1, as in the classifier path.
- **Temperature:** `boundary_temperature` is the fixed, positive training temperature. The GP config must keep `metric="log loss"` and `transformer="sigmoid"`, which describe this fixed loss. Parsimony must be a finite nonnegative number; `"auto"` is undefined when invalid candidates have infinite loss.
- **Final program:** the lowest raw (unpenalized) training loss in the last generation, which is gplearn's own policy. Parsimony acts only in tournaments, and the number of generations is the tuned quantity. Alternative policies are T08C.

`fit_gp_boundary` returns a `GPBoundaryModel` and the engine. The engine carries the dataset-bound metric and history, is only for diagnostics of that fit, and is never saved. `GPBoundaryModel` holds the executable tree with its metric removed, the feature order, the constants and T. It exposes `predict_boundary(X)` (F), `predict_margin(X, log_nu_b)`, `predict_proba(X, log_nu_b)` with shape `(N,)`, and `predict(X, log_nu_b)` with the strict `margin > 0` rule and no tolerance. Inputs are prepared, unscaled, finite and positive. Nonfinite F raises. For original-parameter inference use `GPPredictor` (below).

## Original-parameter inference and evaluation (T06A)

`GPPredictor(model, data_spec)` in `src/debbirth/models/gp/predict.py` has the same interface as `NNPredictor`. Both share `ParameterPredictor` (`src/debbirth/evaluate/predictor.py`).

- **Inputs:** DataFrames or `(N, 4)` arrays ordered `(g, k, v_Hb, f)` go through shared preparation. `predict_prepared(split)` explicitly takes an unscaled `PreparedSplit` and never normalizes again.
- **Outputs:** `predict_details` returns `(N,)` vectors `learned_output`, `logit`, `probability`, `prediction`, plus `margin` for boundary models. `predict_margin` and `critical_maturity(normalized=..., log=...)` require a boundary model.
- **Thresholds:** `predict(..., probability_threshold=t)` and `evaluate_prepared(split, probability_threshold=t)` apply an explicit operating point, `probability >= t`, kept separate from the canonical decision.
- **Canonical decisions:** boundary models use the strict `margin > 0` in both families. Score models keep their historical rules. GP uses gplearn's argmax, so `probability > 0.5` and a tie is infeasible. NN uses `probability >= net_config.threshold`. These differ only at exactly 0.5.
- **Classifier outputs:** GP classifier outputs come from one program execution and reproduce gplearn's `predict_proba`/`predict` exactly. Scores may be infinite, but NaN scores raise.

`train_gp_classifier` and `load_gp_run` both return a predictor. The archived full-parameter GP's training config cannot be reconstructed, so its predictor uses the historical unscaled `(g, k, v_Hb, f)` schema. `train_gp_classifier` computes validation metrics through `GPPredictor.evaluate_prepared`: one prediction pass whose classifier metrics equal the historical two-call `evaluate_binary_classifier` used by the tuner.

## Saved runs

Saved runs keep the existing layout:

- `train_gp_config.json`, `run_metadata.json` (loss, decision rule, final-program policy, temperature policy, data identity) and `metrics/val_metrics.json`.
- `history.csv`.
- `model/gp_model.joblib`: the classifier, or the `GPBoundaryModel`.
- `model/best_program.txt`: the exact program.
- `model/expression.txt`.

`expression.txt` gives the exact gplearn program with feature and constant names, the definitions of the derived inputs, the constant values, and the runtime definition of every primitive in the function set. It also states the original-variable rule. No algebraic simplification is applied. The text documents the model; the joblib artifact is the inference object. Simplified SymPy/MATLAB exports drop protected semantics and are tracked as T06D. `load_gp_run` returns the saved model for either type and a `GPPredictor`.

Example (8 generations on 2,000 training rows, not tuned):

```text
F(gamma, k, x_b) = log(Psi) = sub(sqrt2, k)
Rule: reached_birth = 1 iff log(v_Hb) - 3*log(f) < F, i.e. nu_b < exp(F); in original variables v_Hb < f^3 * exp(sqrt(2) - k).
```

This tiny model ignores gamma and gives `F(gamma, 1) = sqrt(2) - 1`, not the analytical `F(gamma, 1) = 0`. It shows execution, not the learned surface.

## Validation (2026-10-02)

`conda run -n debbirth python -m experiments.validate_gp_formulations` ran on CPU. It used 2,000 training and 500 validation rows sampled with seed 42 from the existing splits, with matched membership across runs and no test data. The runs used population 200, 8 generations and tournament size 20, with the remaining settings from the shipped configs. There were three saved runs: normalized with `x_b`, boundary with `x_b` and T = 1, and boundary without `x_b` and T = 0.7. Artifacts: `results/runs/2026-10-02T16-55-47-145588_t06_gp_validation/`.

Checks passed:

- **Fitness audit:** raw and penalized fitness were recomputed independently for every retained program: 718 and 575 boundary records, plus 250 with explicit `"balanced"` weights. The final program is the last generation's lowest raw loss and is finite, and only configured primitives and permitted feature/constant terminals occur.
- **Parallelism:** serial and two-process evolution give identical programs and outputs.
- **Row alignment:** whole-record shuffles permute every output identically, and the bound metric rejects another split's labels.
- **Boundary semantics:** probability is nonincreasing in maturity at fixed inputs, zero margin is infeasible, and changing T leaves decisions unchanged.
- **Metrics and reloading:** saved validation metrics match recomputation from the saved model's predictions. The reloaded config is equal, and reloaded predictions agree exactly, both in-process and in a fresh process. `expression.txt` contains the exact program.
- **Preservation:** archived artifacts and source CSVs kept identical hashes.

A separate one-generation, population-16 smoke run trained `gp_full_par`, `gp_normalized` and `gp_boundary` through ordinary CSV loading of the full splits, without saving.

Validation metrics on these 500 rows (macro-F1/MCC: normalized 0.874/0.749, boundary 0.895/0.793) are execution diagnostics, not tuned or comparable results. Temperature, tuning budgets and the comparison protocol remain T08/T08B/T09.
