# Normalized and boundary neural networks (T05)

Implemented and checked on 2026-09-08. Both new NN formulations now train, save,
load and predict through `models/nn/train.py`. This implements the NN portion of
T06A needed for inference and evaluation. GP integration, shared plotting,
temperature calibration and full tuning remain later tasks.

## Representation and loss

The normalized network learns an unrestricted score from `(gamma, k, nu_b)`.
The boundary network learns unrestricted `F = log(Psi)` from `(gamma, k)`.
Both support the shared optional `x_b` feature and `none`, `standardize`, or
`log_standardize` preprocessing. Scalers fit only on training features.

`DEBBirthBoundaryNet.forward(x)` returns F. The last layer is linear, so F can
have either sign. Training uses the shared `output_to_logit` function outside
the learned network:

```text
log_nu_b = log(v_Hb) - 3*log(f)
margin = F - log_nu_b
logit = margin / boundary_temperature
probability = sigmoid(logit)
```

This is the representation in [the GP decision note](gp_backend_decision.md),
with gradient descent replacing symbolic evolution. The model fits binary
`reached_birth` labels directly; there are no generated Psi regression targets.
Dictionary batches keep labels, features, row indices and unscaled offsets
aligned through shuffling. Neither maturity nor row identity enters the network.

The loss is stable `BCEWithLogitsLoss(reduction="mean")`. Since T06C the
default and the shipped `experiments/nn_*.json` configs are unweighted
(`use_pos_weight=false`, no `pos_weight`), so training and validation report
ordinary mean BCE. Weighting is an explicit opt-in kept in
`experiments/nn_*_pos_weight.json`; a `pos_weight` without
`use_pos_weight=true` is rejected rather than silently ignored. With
`use_pos_weight=True`, positive terms receive `N_negative/N_positive`, computed
on the selected training rows unless explicitly supplied. Validation reuses
this training weight. The denominator is the number of rows, not the sum of
weights; this preserves historical NN loss semantics and differs from the GP
prototype's weighted mean. Both training and validation epoch loss reports
weight batch means by row count, including the last partial batch. The former
training report averaged batch means equally; this reporting correction does
not change optimization.

Temperature is fixed, finite and positive for each run. No calibration or
temperature search was performed. Changing training temperature can change
the learned F; adjusting temperature with fixed weights leaves the canonical
zero-margin decision unchanged.

## Training and checkpoint selection

The existing `train_net(cfg, save=True)` workflow loads the configured CSVs.
Examples for subsequent experiments are `experiments/nn_normalized.json` and
`experiments/nn_boundary.json`. They are untuned, full-data configurations,
not the small validation runs below. For example, from the repository root:

```python
from src.debbirth.models.nn.config import TrainDEBBirthNetConfig
from src.debbirth.models.nn.train import train_net

cfg = TrainDEBBirthNetConfig.load_json("experiments/nn_boundary.json")
result = train_net(cfg, save=True)
```

For matched subset experiments, `train_net(..., prepared=splits,
data_metadata=metadata)` accepts unscaled `PreparedSplit` records containing
`train` and `val`; `test` is optional and unused by training. Select entire
records with `.select(...)`. Callers supply source hashes and save selected
source-row identities, as demonstrated by the validation script. This explicit
path does not read configured CSVs. The ordinary loader still reads all three
configured CSVs but training does not evaluate the test split. Both paths
require `data_splits="train_val_test"`; `train_test` remains loader-only.

`checkpoint_selection` has two choices:

- `final_epoch` (default for historical configurations): save the final epoch.
- `best_val_loss`: retain the epoch with the lowest validation BCE, with earliest
  epoch winning exact ties; restore its weights after completing all epochs.

`history.csv` records every epoch at the end of the run; `progress.csv` records validation progress during it (see `docs/usage_guide.md`). `checkpoint.json` records policy, selection
metric, selected epoch, selected loss and completed epochs.
`metrics/val_metrics.json` and returned `val_metrics` describe the selected
weights. There is no early stopping. The standalone saver requires
`selected_epoch` explicitly for `best_val_loss` and expects the caller to
supply those weights. Config construction/loading creates no output directories;
resolved output paths and computed positive weights are returned in
`train_config`, without mutating the input config.

## Inference and artifacts

```python
from src.debbirth.models.nn.train import load_trained_nn

loaded = load_trained_nn(result["outdir"], device="cpu")
predictor = loaded["predictor"]
physical = [[1.0, 0.3, 0.1, 1.0]]  # g, k, v_Hb, f
p = predictor.predict_proba(physical)             # NumPy shape (N,)
decision = predictor.predict(physical)
margin = predictor.predict_margin(physical)       # boundary only
Psi = predictor.critical_maturity(physical)
v_Hb_crit = predictor.critical_maturity(physical, normalized=False)
log_Psi = predictor.critical_maturity(physical, log=True)
```

The lightweight `NNPredictor` bundles model, spec, saved scaler and temperature.
Its original-parameter, threshold, margin and critical-maturity methods are
shared with `GPPredictor` through `evaluate/predictor.py` (T06A); outputs are
unchanged (checked against the saved T05 probes).
Public methods accept original-parameter arrays ordered `(g, k, v_Hb, f)`, or
DataFrames with named original columns. `predict_details` returns aligned
vectors for `learned_output`, `logit`, `probability`, `prediction`, and boundary
`margin`. `predict_prepared(split)` explicitly accepts an **unscaled** prepared
record, checks its feature order and offsets, and skips normalization. It cannot
detect values that a caller has already standardized manually.

The normalized critical maturity is `exp(F)`; the original critical maturity
is evaluated as `exp(F + 3*log(f))`. Explicit threshold exponentiation raises
if its result overflows or underflows to zero; request `log=True` for extreme
thresholds. Probabilities do not need exponentiated thresholds. Invalid physical
inputs are rejected through shared preparation; NN float32 conversion rejects
unrepresentable features. Offsets and inference use float32, as in training.

Boundary `predict` uses strict `margin > 0`, with equality infeasible and no
tolerance. It never derives the canonical decision from a rounded probability.
Score formulations retain historical `probability >= net_config.threshold`.
Boundary training requires `net_config.threshold=0.5`. An explicit
`predict(..., probability_threshold=q)` uses `probability >= q` as a separate
operating rule, including its own equality convention; it is not a tuned or
saved calibration result and must be reported separately if used.

`load_trained_nn` retains `model`, `scaler` and `train_cfg`, adds `predictor`,
`net_cfg` and optional checkpoint metadata, and returns the model in evaluation
mode so dropout is inactive. Low-level full-parameter APIs and state-dict/scaler
formats remain intact. Low-level boundary probability/prediction calls require
an explicit `log_nu_b` argument, preventing accidental use of `sigmoid(F)`.

NN validation now collects loss and predictions in one batched forward pass.
`metrics_from_predictions(y, probabilities, y_pred=decisions)` separates metrics
from model execution. Metrics include per-class errors, macro-F1, MCC, ranking
scores, unweighted Brier score and unweighted probability log loss. Probability
log loss uses numerical clipping and may differ from stable raw-logit BCE for
saturated probabilities, even without class weighting. Old metric JSONs still
load, with unavailable new metrics represented by NaN. Generic GP execution and
coordinate-aware plotting have not been refactored by this task.

## Executed validation

Reproduce with `conda run -n debbirth python -m experiments.validate_nn_formulations`.
The successful run is under
[`results/runs/2026-09-08T16-27-53-838745_t05_nn_validation/`](../results/runs/2026-09-08T16-27-53-838745_t05_nn_validation/),
with `summary.json`, source hashes/row identities, configs, weights, scalers,
history, checkpoint metadata and fresh-process inference probes.

Both main runs used the same 512 training and 128 validation observations,
sampling/training seed 42, two 16-unit hidden layers, dropout 0.2, batches of
64, log standardization, positive class weighting and eight epochs. Boundary
temperature was 0.7. Both selected epoch 8. These are engineering diagnostics
from weighted pilot runs that predate the T06C unweighted default:

| Formulation | Validation weighted BCE | Macro-F1 | MCC |
| --- | ---: | ---: | ---: |
| Normalized | 0.587717 | 0.821862 | 0.659281 |
| Boundary | 0.542783 | 0.787375 | 0.582456 |

Additional two-epoch runs covered both variants without scaling and boundary
with standardization plus `x_b`. A synthetic CSV fixture exercised ordinary
loading/training/saving for all three formulations with final-epoch selection.
A deliberately conflicting train/validation fixture made validation loss rise
over three epochs: best-loss selection restored epoch 1, with weights exactly
equal to a separate one-epoch run and saved metrics agreeing with recomputation.

Checks passed for finite loss/probabilities, exact save/load predictions in the
same and a fresh process, loaded evaluation mode, no-scaler artifacts, aligned
shuffled offsets, training-only scaler statistics, independent weighted BCE,
one forward evaluation per validation row (including partial batches), maturity
monotonicity at k=0.3/1/3, scaling-equivalent inputs, original critical-maturity
scaling, strict ties, temperature-invariant canonical decisions (including
positive margins whose tempered logits underflow to zero), empty/invalid
inputs, extreme log offsets and finite gradients. The k=1 tie probe fixed F=0
manually; the learned model does not enforce the analytical k=1 identity.

Archived NN predictions matched the T01 probes. Source train/validation CSVs,
all archived artifacts, historical example configurations and GP tuning source
retained identical hashes. No supplied test data were read or evaluated. Sandbox
CSV access failed; approved execution outside the sandbox succeeded. CPU only
was checked; CUDA was unavailable. No dependencies, source datasets, archived
weights or tuning settings were changed, and no new test infrastructure was added.

The practical get_lb2 failure/timeout-negative label policy is unchanged. These
small runs establish working interfaces, not comparative performance, probability
calibration, an exact numerical critical surface or a mathematical proof.
