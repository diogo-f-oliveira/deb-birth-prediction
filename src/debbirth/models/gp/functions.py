from typing import Callable, Tuple, Union, Any

import numpy as np
from gplearn.functions import make_function


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


# Type aliases
GPFunction = Union[str, Callable[..., Any]]
GPFunctionSet = Tuple[GPFunction, ...]

# Predefined function sets

ARITHMETIC_FUNCTION_SET: GPFunctionSet = ("add", "sub", "mul", "div")

DEFAULT_FUNCTION_SET: GPFunctionSet = ("add", "sub", "mul", "div", "sqrt", "log", "inv", "neg")

EXTENDED_FUNCTION_SET: GPFunctionSet = DEFAULT_FUNCTION_SET + (CBRT, SQUARE, ATAN)
