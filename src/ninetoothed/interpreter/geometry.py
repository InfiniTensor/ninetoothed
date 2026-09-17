"""Conservative integer-plan proofs for execution-local extraction geometry."""

import math
import operator
from functools import partial

import numpy as np

from ninetoothed.ir import AccessMap, IndexExpr, LayoutLevel, TensorLayout, TensorSpec

from . import expressions

_MIN = -(2**63)
_MAX = 2**63 - 1
_CONSTANT = expressions._constant
_SYMBOL = expressions._symbol
_BINARY = expressions._binary
_UNARY = expressions._unary
_COMPARISONS = (
    operator.eq,
    operator.ne,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
)
_LOGICAL = (np.logical_and, np.logical_or)
_INTEGER = (operator.add, operator.sub, operator.mul, np.floor_divide, np.remainder)
_BITS = (np.bitwise_and, np.bitwise_or, np.bitwise_xor)


def _bounded(low, high, boolean=False):
    return (low, high, boolean) if _MIN <= low <= high <= _MAX else None


def _literal(value):
    if type(value) is bool:
        return (int(value), int(value), True)

    if type(value) is int:
        return _bounded(value, value)
    return None


def _trusted(expression, depth=0):
    if (
        type(expression) is not IndexExpr
        or depth > 128
        or type(expression.op) is not str
    ):
        return False

    if expression.op == "constant":
        return not expression.operands and _literal(expression.value) is not None

    if expression.op == "symbol":
        return not expression.operands and type(expression.value) is str

    count = (
        2
        if expression.op in expressions.BINARY
        else 1
        if expression.op in expressions.UNARY
        else 0
    )

    return (
        count != 0
        and expression.value is None
        and len(expression.operands) == count
        and all(_trusted(value, depth + 1) for value in expression.operands)
    )


def _bounds(plan, symbols, memo):
    if id(plan) in memo:
        return memo[id(plan)]

    result = _plan_bounds(plan, symbols, memo)
    memo[id(plan)] = result

    return result


def _plan_bounds(plan, symbols, memo):
    if type(plan) is not partial or plan.keywords:
        return None

    if plan.func is _CONSTANT and len(plan.args) == 1:
        return _literal(plan.args[0])

    if plan.func is _SYMBOL and len(plan.args) == 1:
        return symbols.get(plan.args[0])

    if plan.func is _UNARY and len(plan.args) == 2:
        function, child = plan.args
        value = _bounds(child, symbols, memo)

        if value is None:
            return None

        low, high, boolean = value

        if function is np.logical_not:
            return (0, 1, True)

        if function is np.invert:
            return (0, 1, True) if boolean else _bounded(-high - 1, -low - 1)

        if not boolean and function is operator.pos:
            return value

        if not boolean and function is operator.neg:
            return _bounded(-high, -low)
        return None

    if plan.func is not _BINARY or len(plan.args) != 3:
        return None

    function, left, right = plan.args
    a, b = _bounds(left, symbols, memo), _bounds(right, symbols, memo)

    if a is None or b is None:
        return None

    if any(function is item for item in (*_COMPARISONS, *_LOGICAL)):
        return (0, 1, True)

    if any(function is item for item in _BITS):
        if a[2] and b[2]:
            return (0, 1, True)

        if a[2] or b[2]:
            return None

        if a[0] >= 0 and b[0] >= 0:
            high = (
                min(a[1], b[1])
                if function is np.bitwise_and
                else (1 << max(a[1].bit_length(), b[1].bit_length())) - 1
            )

            return _bounded(0, high)
        return (_MIN, _MAX, False)

    if a[2] or b[2] or not any(function is item for item in _INTEGER):
        return None

    if function is operator.add:
        return _bounded(a[0] + b[0], a[1] + b[1])

    if function is operator.sub:
        return _bounded(a[0] - b[1], a[1] - b[0])

    if function is operator.mul:
        products = [x * y for x in a[:2] for y in b[:2]]

        return _bounded(min(products), max(products))

    if b[0] <= 0:
        return None

    if function is np.remainder:
        return _bounded(0, b[1] - 1)

    quotients = [x // y for x in a[:2] for y in b[:2]]

    return _bounded(min(quotients), max(quotients))


class ExtractionGeometry:
    """Keep one proven address map; array contents are always read separately."""

    def __init__(self):
        self._schema = None
        self._roots = None
        self._key = None
        self._value = None

    def access(self, ref):
        if (
            type(ref.array) is not np.ndarray
            or type(ref.spec) is not TensorSpec
            or type(ref.symbols) is not dict
            or np.dtype(int).itemsize != 8
        ):
            return ref._access()

        layout = ref.spec.layout

        if (
            type(layout) is not TensorLayout
            or type(ref.level) is not int
            or not layout.levels
            or ref.level != len(layout.levels) - 1
            or not layout.value_accesses
        ):
            return ref._access()

        schema = (id(layout), ref.level)

        if self._schema != schema:
            self._schema, self._roots, self._key, self._value = schema, None, None, None
            level, access = layout.levels[ref.level], layout.value_accesses[-1]

            if type(level) is LayoutLevel and type(access) is AccessMap:
                roots = (*level.shape, access.predicate, *access.source_indices)

                if all(_trusted(root) for root in roots):
                    self._roots = (len(level.shape), roots)

        if self._roots is None or type(ref.outer_index) is not int:
            return ref._access()

        bindings = []
        ranges = {}

        for name, value in ref.symbols.items():
            interval = _literal(value)

            if type(name) is not str or interval is None:
                return ref._access()

            bindings.append((name, type(value), value))
            ranges[name] = interval

        outer_range = _literal(ref.outer_index)

        if outer_range is None or type(ref.extracted) is not tuple:
            return ref._access()

        for coordinates in ref.extracted:
            if type(coordinates) is not tuple or any(
                type(value) is not int or _literal(value) is None
                for value in coordinates
            ):
                return ref._access()

        rank, roots = self._roots
        plans = tuple(
            expressions._compiled_expression(expressions._ExpressionIdentity(root))
            for root in roots
        )
        key = (
            schema,
            ref.outer_index,
            ref.extracted,
            ref.array.shape,
            ref.array.strides,
            ref.array.dtype.str,
            tuple(sorted(bindings)),
            plans,
        )

        if key == self._key:
            return self._value

        self._key, self._value = None, None
        shape = []

        for plan in plans[:rank]:
            value = _bounds(plan, ranges, {})

            if value is None or value[2] or value[0] != value[1] or value[0] <= 0:
                return ref._access()

            shape.append(value[0])

        if math.prod(shape) > 65536:
            return ref._access()

        # TensorRef.shape evaluates the original symbols. Only the access map
        # receives these coordinate overrides, so keep their proofs separate.
        ranges["outer_index"] = outer_range

        for level, coordinates in enumerate(ref.extracted):
            for dimension, value in enumerate(coordinates):
                ranges[f"extract_{level}_{dimension}"] = _literal(value)

        for dimension, size in enumerate(shape):
            ranges[f"value_{dimension}"] = (0, size - 1, False)

        values = [_bounds(plan, ranges, {}) for plan in plans[rank:]]

        if any(value is None for value in values) or any(
            value[2] for value in values[1:]
        ):
            return ref._access()

        result = ref._access()
        self._key, self._value = key, result

        return result
