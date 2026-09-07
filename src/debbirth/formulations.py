"""Output semantics independent of array backend: score, F, margin, and logit.

Use a stable backend sigmoid (scipy.special.expit or torch.sigmoid) for
probability. The boundary learner produces F = log(Psi), not the final logit.
"""
from math import isfinite

FORMULATIONS = ("full_par", "normalized", "boundary")


def validate_temperature(temperature: float) -> None:
    if not isfinite(temperature) or temperature <= 0:
        raise ValueError("Boundary temperature must be finite and strictly positive.")


def boundary_margin(learned_boundary, log_nu_b):
    """Return F - log(nu_b), rejecting accidental broadcasting across rows."""
    if getattr(learned_boundary, "shape", ()) != getattr(log_nu_b, "shape", ()):
        raise ValueError("Boundary output and log_nu_b must have identical shapes.")
    return learned_boundary - log_nu_b


def output_to_logit(learned_output, *, formulation="full_par", log_nu_b=None, temperature=1.0):
    if formulation not in FORMULATIONS:
        raise ValueError(f"Unknown formulation {formulation!r}; expected {FORMULATIONS}.")
    validate_temperature(temperature)
    if formulation == "boundary":
        if log_nu_b is None:
            raise ValueError("Boundary logits require the separately supplied log_nu_b.")
        return boundary_margin(learned_output, log_nu_b) / temperature
    if log_nu_b is not None or temperature != 1.0:
        raise ValueError("Maturity offsets and temperature apply only to the boundary formulation.")
    return learned_output


def output_to_probability(learned_output, *, sigmoid, **kwargs):
    """Apply the caller's stable sigmoid, preserving NumPy/Torch and gradients."""
    return sigmoid(output_to_logit(learned_output, **kwargs))


def boundary_is_feasible(margin):
    """Strict canonical decision: equality is infeasible; no tolerance."""
    return margin > 0
