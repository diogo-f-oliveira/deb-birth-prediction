from dataclasses import dataclass
from math import isfinite
from typing import Tuple


@dataclass(frozen=True)
class GPConstant:
    """A named constant terminal.

    The `name` is used as the constant feature/terminal name in the GP expression.
    The `value` is appended as a constant-valued feature column.
    """

    name: str
    value: float


GPConstantSet = Tuple[GPConstant, ...]

# Configuration files select these names; numerical definitions live only here.
# Keep existing names/values stable so saved experiment settings remain meaningful.
CONSTANT_REGISTRY: dict[str, float] = {
    "c0": 0.0,
    "c1": 1.0,
    "c2": 2.0,
    "c3": 3.0,
    "c1_2": 0.5,
    "c1_3": 1.0 / 3.0,
    "sqrt2": 2 ** 0.5,
    "sqrt3": 3 ** 0.5,
}


def resolve_constant(constant) -> GPConstant:
    """Resolve a name; accept old pairs only when they match the registry.

    GPConstant remains the internal terminal representation and retains its
    historical import path for archived joblib models.
    """
    if isinstance(constant, dict):
        constant = GPConstant(**constant)
    name = constant.name if isinstance(constant, GPConstant) else constant
    if not isinstance(name, str) or name not in CONSTANT_REGISTRY:
        raise ValueError(f"Unknown GP constant {name!r}; choose from {tuple(CONSTANT_REGISTRY)}.")
    value = CONSTANT_REGISTRY[name]
    if not isfinite(value):
        raise ValueError(f"Registered GP constant {name!r} must be finite.")
    if isinstance(constant, GPConstant) and constant.value != value:
        raise ValueError(f"GP constant {name!r} conflicts with its registered value {value}.")
    return GPConstant(name=name, value=value)


NO_CONSTANT_SET: GPConstantSet = ()

DEFAULT_CONSTANT_SET: GPConstantSet = tuple(resolve_constant(name) for name in ("c1", "c2", "c3"))

EXTENDED_CONSTANT_SET: GPConstantSet = DEFAULT_CONSTANT_SET + tuple(
    resolve_constant(name) for name in ("c1_2", "c1_3", "sqrt2", "sqrt3")
)
