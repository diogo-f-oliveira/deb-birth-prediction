from dataclasses import dataclass
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

NO_CONSTANT_SET: GPConstantSet = ()

DEFAULT_CONSTANT_SET: GPConstantSet = (
    GPConstant(name="c1", value=1.0),
    GPConstant(name="c2", value=2.0),
    GPConstant(name="c3", value=3.0),
)

EXTENDED_CONSTANT_SET: GPConstantSet = DEFAULT_CONSTANT_SET + (
    GPConstant(name="c1_2", value=0.5),
    GPConstant(name="c1_3", value=1.0 / 3.0),
    GPConstant(name="sqrt2", value=2 ** 0.5),
    GPConstant(name="sqrt3", value=3 ** 0.5),
)
