"""Attaches Tensor's operator dunders to their backing ops.

See the comment above the `Tensor` class definition in tensor.py for why this lives in its own
module: it breaks the circular dependency between tensor.py and the ops modules. This module
is imported once (for its side effect) from tmlc/__init__.py, before any Tensor is used.
"""

from tmlc.tensor import Tensor
from tmlc.ops.ops_arithmetic import add, div, mm, mul, negate, power
from tmlc.ops.ops_basic import constant
from tmlc.ops.ops_shape import transpose


def _coerce(other: Tensor | float | int, reciprocal: bool = False) -> Tensor:
    if isinstance(other, (int, float)):
        value = 1 / other if reciprocal else other
        return constant(value, label=str(value))
    return other


def _add(self: Tensor, other: Tensor | float | int) -> Tensor:
    return add(self, _coerce(other))


def _mul(self: Tensor, other: Tensor | float | int) -> Tensor:
    return mul(self, _coerce(other))


def _truediv(self: Tensor, other: Tensor | float | int) -> Tensor:
    return div(self, _coerce(other, reciprocal=True))


def _sub(self: Tensor, other: Tensor | float | int) -> Tensor:
    return add(self, negate(_coerce(other)))


def _rsub(self: Tensor, other: Tensor | float | int) -> Tensor:
    return add(_coerce(other), negate(self))


def _neg(self: Tensor) -> Tensor:
    return negate(self)


def _pow(self: Tensor, other: Tensor | float | int) -> Tensor:
    return power(self, _coerce(other))


def _matmul(self: Tensor, other: Tensor) -> Tensor:
    return mm(self, other)


Tensor.__add__ = _add
Tensor.__radd__ = _add
Tensor.__mul__ = _mul
Tensor.__rmul__ = _mul
Tensor.__truediv__ = _truediv
Tensor.__sub__ = _sub
Tensor.__rsub__ = _rsub
Tensor.__neg__ = _neg
Tensor.__pow__ = _pow
Tensor.__matmul__ = _matmul
Tensor.T = property(transpose)  # pyright: ignore[reportAttributeAccessIssue]
