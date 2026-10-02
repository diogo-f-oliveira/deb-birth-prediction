"""Readable GP model text: exact program syntax plus the original-variable rule.

The program is gplearn's own syntax with feature and named-constant names; no
algebraic simplification is applied, so protected semantics are preserved by
stating each primitive's runtime definition. This text documents a model; the
saved joblib artifact remains the inference object (simplified SymPy/MATLAB
exports are T06D).
"""
from .boundary import GPBoundaryModel
from .functions import primitive_identifier

# Runtime definitions (EPS = 1e-12 for repository primitives; 0.001 for gplearn stock).
PRIMITIVE_SEMANTICS = {
    "add": "add(a, b) = a + b",
    "sub": "sub(a, b) = a - b",
    "mul": "mul(a, b) = a * b",
    "div": "div(a, b) = a / b if |b| > 0.001 else 1  (gplearn protected)",
    "log": "log(a) = log(|a|) if |a| > 0.001 else 0  (gplearn protected)",
    "sqrt": "sqrt(a) = sqrt(|a|)  (gplearn protected)",
    "inv": "inv(a) = 1 / a if |a| > 0.001 else 0  (gplearn protected)",
    "neg": "neg(a) = -a",
    "abs": "abs(a) = |a|",
    "max": "max(a, b) = elementwise maximum",
    "min": "min(a, b) = elementwise minimum",
    "sin": "sin(a)", "cos": "cos(a)", "tan": "tan(a)",
    "pdiv": "pdiv(a, b) = (a + 1e-12) / b', where b' = copysign(1e-12, b) if |b| < 1e-12 (+0 gives +1e-12) else b",
    "plog": "plog(a) = log(max(a, 1e-12))",
    "pinv": "pinv(a) = 1 / a if |a| >= 1e-12 else 0",
    "cbrt": "cbrt(a) = real cube root (negative for negative a)",
    "square": "square(a) = a^2",
    "cube": "cube(a) = a^3",
    "atan": "atan(a) = arctan(a)",
    "beta43_0": "beta43_0(a) = B_x(4/3, 0) with x clipped to [1e-12, 1 - 1e-12]",
}

FEATURE_DEFINITIONS = {
    "g": "g", "k": "k", "v_Hb": "v_Hb", "f": "f",
    "gamma": "gamma = g / f",
    "nu_b": "nu_b = v_Hb / f^3",
    "x_b": "x_b = g / (f + g) = gamma / (1 + gamma)",
}


def gp_program_text(model) -> str:
    """Exact program text for either GP model type (keeps private access here)."""
    if isinstance(model, GPBoundaryModel):
        return model.expression
    return str(model._program)


def gp_model_text(model, cfg) -> str:
    formulation = cfg.data_spec.formulation
    features = list(cfg.data_spec.feature_cols)
    args = ", ".join(features)
    lines = [f"Formulation: {formulation}",
             "Program: exact gplearn syntax with named features and constants (no simplification).",
             "Inputs (original parameters g, k, v_Hb, f; natural logarithms):"]
    lines += [f"  {FEATURE_DEFINITIONS[name]}" for name in features]
    constants = cfg.gp.constants
    if constants:
        lines.append("Constants: " + ", ".join(f"{c.name} = {c.value!r}" for c in constants))
    if formulation == "boundary":
        lines += [
            f"F({args}) = log(Psi) = {gp_program_text(model)}",
            "Critical maturity: Psi = exp(F); in original variables v_Hb,crit = f^3 * exp(F).",
            "Rule: reached_birth = 1 iff log(v_Hb) - 3*log(f) < F, i.e. nu_b < exp(F). "
            "Equality is infeasible; no tolerance or tuned threshold.",
            f"Probability: sigmoid((F - log(nu_b)) / T), training temperature T = {model.temperature!r}.",
        ]
    else:
        lines += [
            f"S({args}) = {gp_program_text(model)}",
            "Probability: P(reached_birth = 1) = sigmoid(S).",
            "Rule: reached_birth = 1 iff P > 0.5 (gplearn argmax; ties infeasible), i.e. S > 0 "
            "up to floating-point rounding of the sigmoid.",
        ]
    lines.append("Primitive definitions:")
    lines += [f"  {PRIMITIVE_SEMANTICS[primitive_identifier(p)]}" for p in cfg.gp.function_set]
    return "\n".join(lines) + "\n"
