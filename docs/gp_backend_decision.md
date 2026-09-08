# T04: GP implementation decision

Completed 2026-09-08. **Keep gplearn for all three formulations.** The existing classifier supports normalized inputs; a small boundary adapter can use `SymbolicRegressor` as the evolution engine with a fixed-offset classification loss. Migration is unnecessary for these formulations. DEAP and PySR remain possible later experiments for different search capabilities.

This decision is supported by the installed gplearn 0.4.3 source and an executed prototype in conda `debbirth`. It does not complete the production integration in T06 or establish a predictive advantage.

The existing `full_par` model and its completed tuning can be reused as the historical benchmark. This backend decision does not require repeating either: the boundary adapter is separate and leaves the historical model's representation, loss and inference unchanged. A new experiment specifically designed to isolate formulation effects under matched training/tuning budgets would be a separate choice. When reusing the archived baseline, report differences in preprocessing, primitives, budgets and selection protocol rather than attributing every performance difference to the formulation alone.

## What gplearn supports

| Formulation | Implementation | Status |
| --- | --- | --- |
| `full_par` | Existing `DEBBirthSymbolicClassifier` on `(g, k, v_Hb, f)` | Historical path preserved |
| `normalized` | Same classifier on shared-prepared `(gamma, k, nu_b)` | Tiny training/reload run passed |
| `boundary` | `SymbolicRegressor` evolves `F(gamma, k)`; dataset-bound fitness applies the separate maturity offset | Tiny training, parallelism, subsampling and inference prototype passed |

The regressor name describes the engine's unrestricted real output. Its target remains binary `reached_birth`; it minimizes classification BCE, not regression error or distance to a solver-generated critical maturity. Its `predict` returns **F**, and its inherited R2 `score` is inappropriate. A boundary inference object supplies margin, probability and strict decisions.

The stock classifier normally transforms the tree output to a sigmoid before calling the metric. Recovering F by applying an inverse sigmoid would introduce saturation errors. The regressor supplies raw F directly. Custom fitness is supported by gplearn's [fitness interface](https://gplearn.readthedocs.io/en/stable/advanced.html#custom-fitness); the [evolution source](https://gplearn.readthedocs.io/en/stable/_modules/gplearn/genetic.html) establishes the masking and selection details below. The local installed source is the authority for the executed experiment.

## Deriving the boundary representation

### 1. Separate the unknown boundary from the observed maturity

For observation i, the dataset supplies positive original parameters `(g_i, k_i, v_Hb_i, f_i)` and the binary label `y_i = reached_birth_i`. Shared preparation computes

\[
\gamma_i=\frac{g_i}{f_i},\qquad
\nu_{b,i}=\frac{v_{Hb,i}}{f_i^3},\qquad
o_i=\log\nu_{b,i}=\log v_{Hb,i}-3\log f_i.
\]

Here `o_i` is a **known offset for observation i**. It is computed from the inputs, not fitted. The log-domain expression avoids forming `f_i^3` or exponentiating maturity during boundary training. All logarithms are natural.

The critical-boundary representation in `birth_equations.md` motivates the strict feasibility rule

\[
\nu_b<\Psi(\gamma,k).
\]

We use this as the model structure; T07 still owns the remaining analytical justification of the complete critical-surface characterization. The recorded solver labels, including timeout/nonconvergence negatives, remain the actual training target and need not match a mathematically exact critical surface.

The quantity to learn is the **threshold** as a function of gamma and k. We do not have observed values of Psi to use as regression targets. Instead, we learn a threshold that explains the existing binary outcomes across different maturities.

### 2. Evolve the logarithm of the threshold

Let theta denote a candidate GP tree, including its constants, and define its real-valued output by

\[
F_\theta(\gamma,k)\in\mathbb R,\qquad
\widehat\Psi_\theta(\gamma,k)=\exp(F_\theta(\gamma,k))>0.
\]

Thus F is an estimate of `log(Psi)`. F itself can be negative: for example, `F = log(0.5)` describes a positive critical maturity of 0.5. We impose positivity on the threshold through this representation, without constraining the tree output to be positive. The exponential is needed only when explicitly reporting the threshold; training and classification can stay in log space.

Because logarithm is strictly increasing, the model's feasibility rule has equivalent forms:

\[
\begin{aligned}
\nu_{b,i}<\widehat\Psi_\theta(\gamma_i,k_i)
&\iff \log\nu_{b,i}<F_\theta(\gamma_i,k_i)\\
&\iff m_i:=F_\theta(\gamma_i,k_i)-o_i>0.
\end{aligned}
\]

The **margin** therefore measures a log ratio:

\[
m_i=\log\frac{\widehat\Psi_\theta(\gamma_i,k_i)}{\nu_{b,i}}.
\]

A positive margin means the observed maturity lies below the predicted threshold; a negative margin means it lies above it. Equality is classified infeasible, with no tolerance. In original variables, the same decision is

\[
\boxed{v_{Hb,i}<f_i^3\exp\!\left[F_\theta(g_i/f_i,k_i)\right].}
\]

### 3. Turn the margin into a probability

The hard decision `m_i > 0` does not distinguish a slightly wrong candidate from a confidently wrong one. To obtain a graded training loss, choose a positive temperature T and set

\[
z_i=\frac{m_i}{T}
=\frac{F_\theta(\gamma_i,k_i)-o_i}{T},\qquad
p_i=\sigma(z_i)=\frac{1}{1+\exp(-z_i)}.
\]

These are three different quantities: **F is the log threshold, m is the margin, and z is the classification logit**. The sigmoid is a modeling choice for fitting the binary outcomes; it is not derived from the DEB dynamics. Probability calibration is a subsequent validation question, particularly with class weighting.

At fixed F, positive T changes sharpness while preserving `p_i > 0.5` if and only if `m_i > 0`. The coefficient of `o_i` in the margin is fixed at -1, and its coefficient in the logit is fixed at `-1/T`. Neither coefficient is independently evolved. Searching over training temperatures entails fitting candidate trees at those temperatures; post-training temperature adjustment holds the selected tree fixed. The prototype simply fixes T=0.7.

For an illustrative candidate with `F(gamma,k)=log(0.5)` at one chosen pair of coordinates and T=1:

| Observed normalized maturity nu_b | Tree output F | Margin m | Probability p | Strict prediction |
| --- | --- | --- | --- | --- |
| 0.25 | log(0.5) | log(2) | 2/3 | Feasible |
| 0.5 | log(0.5) | 0 | 1/2 | Infeasible (tie) |
| 1 | log(0.5) | -log(2) | 1/3 | Infeasible |

The tree returns the **same F for all three rows**. Their different maturities produce different probabilities through the fixed comparison outside the tree. No maturity information needs to enter the evolved expression.

## Deriving the classification loss

### 4. Score each candidate using the binary labels

A candidate tree assigns probability p_i to the observed outcome `y_i=1` and probability `1-p_i` to `y_i=0`. Its Bernoulli probability for the recorded outcome is

\[
P_\theta(y_i\mid\gamma_i,k_i,o_i)
=p_i^{y_i}(1-p_i)^{1-y_i}.
\]

Taking the negative logarithm gives binary cross-entropy:

\[
\ell_i=-y_i\log p_i-(1-y_i)\log(1-p_i).
\]

Write `softplus(a)=log(1+exp(a))`. Substituting the sigmoid identities

\[
\log\sigma(z)=z-\operatorname{softplus}(z),\qquad
\log(1-\sigma(z))=-\operatorname{softplus}(z)
\]

gives

\[
\begin{aligned}
\ell_i
&=-y_i[z_i-\operatorname{softplus}(z_i)]
  +(1-y_i)\operatorname{softplus}(z_i)\\
&=\operatorname{softplus}(z_i)-y_i z_i\\
&=\begin{cases}
\operatorname{softplus}(-z_i), & y_i=1,\\
\operatorname{softplus}(z_i), & y_i=0
\end{cases}\\
&=\boxed{\operatorname{softplus}\!\left[(1-2y_i)
\frac{F_\theta(\gamma_i,k_i)-o_i}{T}\right].}
\end{aligned}
\]

The last equality uses the fact that y_i is binary. It explains both the `1 - 2*y` factor and the sign in the implementation. The code evaluates this as `np.logaddexp(0.0, (1.0 - 2.0*y) * z)`, which avoids explicitly forming potentially overflowing exponentials or taking logarithms of rounded probabilities. It also avoids subtracting two large, almost equal numbers in `softplus(z) - y*z`.

For `y_i=1`, loss decreases when F increases relative to that observation's maturity: the proposed threshold moves above a recorded feasible point. For `y_i=0`, loss decreases when F decreases: the threshold moves below a recorded infeasible point. More formally,

\[
\frac{\partial\ell_i}{\partial F_i}=\frac{p_i-y_i}{T}.
\]

This derivative explains the loss's direction; gplearn does not use gradient descent here. It evaluates candidate trees, compares their losses and evolves them through selection, crossover and mutation. The binary labels constrain which side of the threshold each observation should occupy. They do not supply an exact numerical target for F, and finite or conflicting observations can require a compromise.

### 5. Combine class weighting, sample masks and parsimony

For the selected training subset of N rows, with N_0 negative and N_1 positive labels, the prototype computes balanced weights

\[
a_i=\frac{N}{2N_{y_i}}.
\]

Both classes must be present. Each class then contributes total weight N/2 before subsampling. Let s_i be 1 when a row participates in a particular program's fitness calculation and 0 otherwise. The weights passed by gplearn are `w_i=s_i*a_i`. For an active subset with positive total weight,

\[
\boxed{L(\theta)=\frac{\sum_i w_i\ell_i}{\sum_i w_i}.}
\]

Class weights are computed once from the selected training subset. They are not fitted on validation data or recomputed separately for each program's mask. The weighted-mean denominator follows the existing gplearn implementation. Equation (50) in `birth_equations.md` instead writes division by N: these coincide on the full subset with balanced weights, whose sum is N, but need not coincide after masking or with arbitrary weights. T08 must keep the loss normalization and relative parsimony strength explicit across comparisons.

The scalar fitness used to select parents is

\[
J(\theta)=L(\theta)+c\,|\mathcal T_\theta|,\qquad c\geq0,
\]

where `|T_theta|` counts nodes in the evolved tree for F. The fixed normalization, maturity subtraction and sigmoid are outside that tree and are not counted by this parsimony term. The custom metric returns L only; gplearn adds the tree penalty itself. Adding it inside the metric as well would count it twice. As detailed below, the stock final-program selection uses raw L rather than J.

## How the loss is connected to gplearn

For a given fit, preparation supplies three separate aligned objects:

| Object | Content | Role |
| --- | --- | --- |
| X, shape `(N, 2)` | gamma and k | The only observation columns the tree can access |
| y, shape `(N,)` | Recorded binary reached_birth | Labels supplied to `engine.fit(X, y, ...)` |
| o, shape `(N,)` | log_nu_b | Fixed context bound to the custom fitness |

gplearn calls a metric with the signature `metric(y, y_pred, sample_weight)`. With `SymbolicRegressor`, **the argument named `y_pred` is the raw tree output F**. That name does not require it to approximate y directly: the metric defines how those outputs should be scored. Replacing the default regression metric with the derived BCE makes this engine evolve a classifier boundary.

Binding an offset means creating a metric that remembers the offset array and T for this fit. Schematically, its calculation is:

```python
# o and T belong to this fit; they are not evolved inputs.
def metric(y, F, w):
    active = w > 0
    z = (F[active] - o[active]) / T
    losses = np.logaddexp(0.0, (1.0 - 2.0*y[active]) * z)
    return np.average(losses, weights=w[active])

# The regressor executes each candidate tree on X and calls the metric.
engine.fit(X, y, sample_weight=training_class_weights)
```

This excerpt shows the calculation, not a standalone replacement for the guarded prototype. In [`boundary_prototype.py`](../src/debbirth/models/gp/boundary_prototype.py), a picklable `partial(_bound_loss, expected_y=y, log_nu_b=offset, temperature=T)` binds this context, `_Fitness(..., greater_is_better=False)` tells gplearn to minimize it, and `SymbolicRegressor(metric=metric, ...)` installs it. The implementation adds input validation and the backend checks described below. Every new candidate produces a new F vector; all candidates in that fit are compared against the same observed labels and offsets, subject to their sample-weight masks.

There is no intermediate step that estimates Psi for each row, constructs continuous target values, or fits F to the binary values by squared error. New boundary observations at inference time need their own gamma/k and maturity offset, but no labels and no access to the training metric.

## Preserving row alignment

The adapter accepts one `PreparedSplit`, copies its features, labels and offset into one fit context, and makes those copies read-only. Only gamma and k enter X. It also accepts the shared optional x_b schema; that option was not exercised in this run. Labels remain 0/1, and weights remain weights. There is no maturity terminal, row-ID terminal, label encoding, global offset array, or feature-to-offset lookup. Repeated gamma/k pairs can have different maturities and labels.

The installed backend provides the necessary concrete row contract:

1. `BaseSymbolic.fit` validates the full X/y arrays without resampling rows.
2. `_parallel_evolve` evaluates each tree on full X. `max_samples` selects indices by setting other weights to zero; OOB evaluation uses the complementary mask.
3. `_Program.raw_fitness` passes full aligned y, F and weights to the metric.
4. Joblib distributes **programs** to workers, with the same full observation arrays and a serialized fit context. It does not partition rows.

This is an audited implementation dependency, not a public promise that every future backend will preserve order. The prototype checks version and source fingerprints of the four relevant methods before fitting, failing if they change. The label-array check also rejects obvious context reuse, but alone cannot detect same-class permutations. Source checks plus independent recomputation using each program's explicit sample indices cover the dependency. Reordering or selecting data requires a new `PreparedSplit.select(...)` and a fresh fit context. Warm starts are not exposed.

The small private dependency is constructing `_Fitness` directly with a picklable module-level function/partial. The public `make_fitness` factory invokes the loss on an unrelated two-row fixture, which cannot use the real dataset's offsets. Direct construction avoids special-casing that fixture. No installed gplearn code is modified, no global monkeypatch is used, and no evolution loop is copied.

Zero-weight rows are excluded before loss evaluation. Nonfinite candidate outputs on active rows receive infinite fitness; invalid input data/weights raise. The prototype checks finite selected outputs. It does not promise that arbitrary deep trees or extreme inputs will always produce finite expressions.

## Selection, prediction and saving

Parsimony affects parent tournaments, but gplearn's final returned program and reported generation best are selected by **raw fitness**. The final model is from the last generation; there is no classifier/regressor archive of the best validation model across generations. With subsampling, raw fitness is also measured on different per-program subsets. T06/T08 should explicitly choose and record the candidate-selection policy, using a common validation set for model selection; OOB scores are diagnostics. The prototype retains the stock final-program policy so this distinction is visible.

Prediction computes the offset from the inference observation, independently of the training context. `BoundaryExpression` exposes prepared-input F, margin, positive-class probability and `margin > 0`. Equality is infeasible without tolerance. T is positive and fixed at 0.7 for this engineering probe; no temperature or operating threshold was tuned. Changing T after fitting leaves the zero-margin boundary fixed; changing T during evolution can select a different F.

Both the fitted engine and an inference-only expression are saved. The engine contains its dataset-bound training metric. The inference artifact removes that metric and stores the executable tree, feature order and T; it needs no training labels or offsets. Joblib preserves full-precision constants. The displayed tree text rounds constants and is **not** a standalone exact export. Portable symbolic/MATLAB exports and the original-parameter prediction interface remain T06/T06A.

## Executed experiment

- Implementation: [`boundary_prototype.py`](../src/debbirth/models/gp/boundary_prototype.py).
- Reproduction: [`prototype_gp_boundary.py`](../experiments/prototype_gp_boundary.py).
- Artifacts: [`2026-09-08T15-45-43-866373_t04_gp_backend`](../results/runs/2026-09-08T15-45-43-866373_t04_gp_backend/), including `summary.json`, `run_metadata.json`, source-row CSVs, both boundary artifacts, and the normalized model/config.

```text
conda run -n debbirth python -m experiments.prototype_gp_boundary
```

The run used 512 training and 128 validation rows from the existing get_lb2 splits, with matched membership across formulations. Sampling/evolution seed was 42; population 48, three generations. Only training/validation source files were read. Sandbox data access failed; approved execution outside the sandbox succeeded. No dependencies or source datasets were changed.

Boundary runs covered `(max_samples, n_jobs)` of `(1,1)`, `(1,2)`, `(0.65,1)` and `(0.65,2)`. Independent recomputation verified raw fitness, OOB fitness when present, and penalized fitness for 350 retained program records across these runs. Another 84 records passed after whole-record shuffling, and 85 passed on a synthetic repeated-coordinate fixture with different offsets/labels. Ancestor pruning means these counts are the retained records, not every program ever evaluated. Serial/process-parallel final populations and predictions agreed exactly.

Additional checks passed for loss at logits +/-1000, zero weights/nonfinite candidates, context mismatch rejection, maturity monotonicity, strict ties, temperature-invariant decisions, and scaling-equivalent original inputs. A manually fixed F=0 checked the k=1 decision semantics; it is not evidence that an unconstrained learned model enforces F(gamma,1)=0. Fresh-process reload predictions agreed exactly for the normalized classifier, boundary engine and inference expression.

The boundary run's raw full-subset loss was 0.412410. Its tiny validation subset had macro-F1 0.834668 and MCC 0.673409; these are execution diagnostics, not tuned or publishable performance estimates. Timings include worker-startup effects and are not evidence for backend speed claims. The boundary probe uses stock gplearn add/sub/mul/div/log and ephemeral constants, while the normalized probe uses the existing registered primitives/constants. The stock protected div/log semantics differ from the repository's historical protected primitives. T06 must implement the intended new primitive/constant configuration and check exports; this probe does not choose that experiment's function set.

The accepted paper's practical timeout/error-negative label policy is preserved. Its Section II wording calls the screening inequality sufficient; the displayed constraints imply necessity instead. The PDF is unchanged. Nothing in this prototype establishes the remaining critical-surface proof steps tracked by T07 or equates the numerical solver's failure pattern with the analytical surface.

## What alternative backends would add

| Backend | Additional capabilities relevant here | Cost and decision |
| --- | --- | --- |
| **gplearn** | Existing vectorized tree evaluation, operators, parallel evolution and historical artifacts; all requested formulations are implementable | Small adapter with an explicit version/source dependency. Best choice for T06. |
| **DEAP** | Separate loss/complexity objectives with NSGA-II; hard tree-size/depth limits; typed primitives; direct control over selection, variation, archives and evaluation scheduling | More evolution, serialization and export code to own. Prefer if the research moves to custom evolutionary algorithms or constrained search. No inherent prediction/speed advantage is established. |
| **PySR / SymbolicRegression.jl** | Numerical constant optimization, multiple populations, complexity-aware expression search and operator/nesting constraints; custom objectives and expression templates | Additional Julia runtime and adaptation/verification of the fixed-offset objective, protected semantics and export. Consider if coefficient optimization or larger searches become the bottleneck. No DEB-specific advantage is established. |

DEAP documents [NSGA-II and tree limits](https://deap.readthedocs.io/en/master/api/tools.html), [typed GP](https://deap.readthedocs.io/en/master/tutorials/advanced/gp.html), and [multiprocessing integration](https://deap.readthedocs.io/en/master/tutorials/basic/part4.html). These tools would let us retain a frontier of accuracy/complexity tradeoffs and implement explicit row batches, validation archives or local constant fitting in our evaluation loop. Those latter policies require our own implementation; they are not automatic DEAP features. Typed GP alone also does not impose the DEB k=1 identity or a biological grammar.

PySR's [API](https://ai.damtp.cam.ac.uk/pysr/v2.0.0a2/api) describes constant optimization, templates and custom objectives; its [methods paper](https://arxiv.org/abs/2305.01582) describes the multi-population evolve/simplify/optimize search. The consulted API page is explicitly a 2.0.0 alpha reference, not a claim about a locally installed stable release. Neither DEAP nor PySR is installed in `debbirth`; the comparison is documentation-based, with no installation or benchmark performed.

One concrete limitation worth profiling later: gplearn's `max_samples < 1` masks the loss but still executes trees on all rows, and OOB evaluation executes them again. It does not provide proportional tree-evaluation savings. A DEAP loop could instead select whole aligned records and evaluate only that subset. Changing the search also changes experimental budgets and must be reported separately from the benefit of normalization/boundary structure.

**Recommendation:** implement T06 with gplearn, preserving the historical wrapper and using a small boundary adapter based on this experiment. Revisit DEAP when multiobjective or custom constrained evolution becomes an actual research requirement; consider PySR when constant optimization or measured search cost motivates a separate backend experiment. Backend migration should not delay the main formulation comparison.
