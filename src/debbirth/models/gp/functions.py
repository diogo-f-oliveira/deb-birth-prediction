from typing import Callable, Tuple, Union, Any

import numpy as np
from gplearn.functions import make_function




# Small epsilon used for numerical protection
EPS = 1e-12


def _cbrt(x: np.ndarray) -> np.ndarray:
    """Elementwise cubic root (real-valued, supports negatives)."""
    x = np.asarray(x)
    return np.cbrt(x)


CBRT = make_function(function=_cbrt, name="cbrt", arity=1, wrap=True)


def _square(x: np.ndarray) -> np.ndarray:
    """Elementwise square: x^2."""
    x = np.asarray(x)
    return np.square(x)


SQUARE = make_function(function=_square, name="square", arity=1, wrap=True)


def _cube(x: np.ndarray) -> np.ndarray:
    """Elementwise cube: x^3."""
    x = np.asarray(x)
    return np.power(x, 3)


CUBE = make_function(function=_cube, name="cube", arity=1, wrap=True)


def _atan(x: np.ndarray) -> np.ndarray:
    """Elementwise arctangent."""
    x = np.asarray(x)
    return np.arctan(x)


ATAN = make_function(function=_atan, name="atan", arity=1, wrap=True)


def _pdiv(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Protected division.

    Adds a small EPS to the numerator and guards tiny denominators by substituting
    a non-zero epsilon with the sign of y (zero maps to +EPS), avoiding zero denominators.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # Use copysign to ensure zero -> +EPS and preserve sign for non-zero tiny values
    denom_safe = np.where(np.abs(y) < EPS, np.copysign(EPS, y), y)
    return (x + EPS) / denom_safe


PDIV = make_function(function=_pdiv, name="pdiv", arity=2, wrap=True)


def _pinv(x: np.ndarray) -> np.ndarray:
    """Protected inversion: returns 0 where |x| < EPS, else 1/x.

    Use np.divide with the 'where' argument to avoid performing division
    on zero/near-zero entries and eliminate divide-by-zero warnings.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    mask = np.abs(x) >= EPS
    # Perform division only where mask is True; other entries remain zero.
    np.divide(1.0, x, out=out, where=mask)
    return out


PINV = make_function(function=_pinv, name="pinv", arity=1, wrap=True)


def _plog(x: np.ndarray) -> np.ndarray:
    """Protected log: floor inputs at EPS before taking log (assumes non-negative inputs)."""
    x = np.asarray(x, dtype=float)
    safe_x = np.maximum(x, EPS)
    return np.log(safe_x)


PLOG = make_function(function=_plog, name="plog", arity=1, wrap=True)

def _beta_43_0(x: np.ndarray) -> np.ndarray:
    """
    Protected version of the DEB-book special case B_x(4/3, 0):
        B_x(4/3, 0) = ∫_0^x t^(1/3) / (1 - t) dt

    Closed form via substitution u = x^(1/3):
        = -3u - ln(1-u) + 0.5 ln(u^2+u+1)
          + sqrt(3) * arctan((2u+1)/sqrt(3)) - pi/(2*sqrt(3))

    Protection: clip x to [eps, 1-eps] to avoid singularity at x=1.
    """
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, EPS, 1.0 - EPS)

    u = np.cbrt(x)  # u = x^(1/3), stable for arrays
    q = u*u + u + 1.0

    sqrt3 = np.sqrt(3.0)
    out = (
        -3.0 * u
        - np.log1p(-u)                      # -ln(1-u)
        + 0.5 * np.log(q)                   # +0.5 ln(u^2+u+1)
        + sqrt3 * np.arctan((2.0*u + 1.0) / sqrt3)
        - (np.pi / (2.0 * sqrt3))
    )

    # Extra safety: replace any non-finite values (shouldn't occur with clipping)
    out = np.where(np.isfinite(out), out, 0.0)
    return out

BETA_43_0 = make_function(function=_beta_43_0, name="beta43_0", arity=1)

# Type aliases
GPFunction = Union[str, Callable[..., Any]]
GPFunctionSet = Tuple[GPFunction, ...]

# Predefined function sets

# Replace "div" with protected PDIV
ARITHMETIC_FUNCTION_SET: GPFunctionSet = (
    "add",
    "sub",
    "mul",
    PDIV
)

# DEFAULT_FUNCTION_SET: replace "div", "log", "inv" with PDIV, PLOG, PINV respectively
DEFAULT_FUNCTION_SET: GPFunctionSet = (
    "add",
    "sub",
    "mul",
    PDIV,
    "sqrt",
    PLOG,
    PINV,
    "neg",
    "min",
    "max",
)

# EXTENDED now builds on DEFAULT_FUNCTION_SET (inherits protected primitives)
EXTENDED_FUNCTION_SET: GPFunctionSet = DEFAULT_FUNCTION_SET + (
    CBRT,
    SQUARE,
    ATAN
)
