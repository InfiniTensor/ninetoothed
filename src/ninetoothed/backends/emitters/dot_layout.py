"""Shared proofs for replacing complete dense dot tiles with native primitives."""

import re
from dataclasses import dataclass
from typing import Mapping


def integer_interval(expression, domains):
    """Conservative bounds, without assuming unknown scalar symbols positive."""
    import sympy

    if expression.is_number:
        return expression, expression

    if expression in domains:
        return domains[expression]

    if isinstance(expression, sympy.Add):
        intervals = [integer_interval(item, domains) for item in expression.args]

        if any(item is None for item in intervals):
            return None
        return sum(item[0] for item in intervals), sum(item[1] for item in intervals)

    if isinstance(expression, sympy.Mul):
        result = (sympy.Integer(1), sympy.Integer(1))

        for item in expression.args:
            interval = integer_interval(item, domains)

            if interval is None:
                return None

            products = [a * b for a in result for b in interval]
            result = min(products), max(products)
        return result

    if expression.func == sympy.floor:
        interval = integer_interval(expression.args[0], domains)

        if interval is not None:
            return tuple(sympy.floor(value) for value in interval)

    if expression.func == sympy.Mod:
        divisor = expression.args[1]

        if divisor.is_Integer and divisor > 0:
            interval = integer_interval(expression.args[0], domains)

            if interval is not None and interval[0] >= 0:
                return sympy.Integer(0), min(interval[1], divisor - 1)
    return None


def predicate_always_true(predicate, domains):
    """Prove every original predicate; unknown or interior masks stay indexed."""
    import sympy

    if predicate == sympy.true:
        return True

    if isinstance(predicate, sympy.And):
        return all(predicate_always_true(item, domains) for item in predicate.args)

    if not getattr(predicate, "is_Relational", False):
        return False

    interval = integer_interval(predicate.lhs - predicate.rhs, domains)

    if interval is None:
        return False

    low, high = interval

    return bool(
        (isinstance(predicate, sympy.StrictLessThan) and high < 0)
        or (isinstance(predicate, sympy.LessThan) and high <= 0)
        or (isinstance(predicate, sympy.StrictGreaterThan) and low > 0)
        or (isinstance(predicate, sympy.GreaterThan) and low >= 0)
        or (isinstance(predicate, sympy.Equality) and low == high == 0)
    )


@dataclass(frozen=True)
class DenseDotAccess:
    offsets: tuple[str, str]
    strides: tuple[str, str]
    source_shape: tuple[int, int]

    def base_offsets(self, replacements: Mapping[str, str]):
        result = []

        for offset in self.offsets:
            for old, new in {"value_0": "0", "value_1": "0", **replacements}.items():
                offset = re.sub(rf"\b{re.escape(old)}\b", f"({new})", offset)

            result.append(offset)
        return tuple(result)


def prove_dense_dot_access(attrs, *, shape, tasks, stages, expected_offsets):
    """Prove unit-coordinate maps and complete masks over a finite tiled domain.

    Expected offsets bind operands to the original store grid and induction
    variable. This additionally proves exclusive output ownership. No names,
    source text, application identity, or runtime data values select a plan.
    """
    import sympy

    templates = attrs.get("access_templates", ())

    if (
        len(templates) != 1
        or attrs.get("source_ndim") != 2
        or attrs.get("jagged_dim") is not None
        or tasks <= 0
        or stages <= 0
    ):
        return None

    template = templates[0]

    if template.get("level", 0) != attrs.get("dtype_level", 0):
        return None

    offsets = tuple(map(str, template.get("offsets", ())))
    strides = tuple(map(str, attrs.get("source_strides", ())))

    if (
        len(offsets) != 2
        or len(strides) != 2
        or tuple(map(str, template.get("shape", ()))) != tuple(map(str, shape))
        or template.get("jagged")
    ):
        return None

    text = " ".join(
        (*offsets, *map(str, expected_offsets), str(template.get("mask", True)))
    )
    symbols = {
        name: sympy.Symbol(name, integer=True)
        for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)
        if name not in {"floor", "Mod", "True", "False"}
    }

    for name in ("outer_index", "value_0", "value_1", "extract_0_0"):
        symbols.setdefault(name, sympy.Symbol(name, integer=True))

    domains = {
        symbols["outer_index"]: (0, tasks - 1),
        symbols["value_0"]: (0, shape[0] - 1),
        symbols["value_1"]: (0, shape[1] - 1),
        symbols["extract_0_0"]: (0, stages - 1),
    }

    try:
        source_shape = tuple(map(int, attrs["source_shape"]))

        if len(source_shape) != 2 or any(dim <= 0 for dim in source_shape):
            return None

        parsed = tuple(sympy.sympify(offset, locals=symbols) for offset in offsets)
        expected = tuple(
            sympy.sympify(offset, locals=symbols) for offset in expected_offsets
        )

        if any(sympy.simplify(a - b) != 0 for a, b in zip(parsed, expected)):
            return None

        for expression, extent in zip(parsed, source_shape):
            bounds = integer_interval(expression, domains)

            if bounds is None or bounds[0] < 0 or bounds[1] >= extent:
                return None

        mask = sympy.sympify(str(template.get("mask", True)), locals=symbols)

        if not predicate_always_true(mask, domains):
            return None
    except (TypeError, ValueError, KeyError, AttributeError, SyntaxError):
        return None
    return DenseDotAccess(offsets, strides, source_shape)


def prove_full_matrix_access(info, shape):
    """Prove that one original task addresses the complete physical matrix."""
    attrs = dict(info.attrs or {})

    if tuple(map(str, attrs.get("source_shape", ()))) != tuple(map(str, shape)):
        return None
    return prove_dense_dot_access(
        attrs,
        shape=shape,
        tasks=1,
        stages=1,
        expected_offsets=(
            f"outer_index * {shape[0]} + value_0",
            f"Mod(outer_index, 1) * {shape[1]} + value_1",
        ),
    )


def dense_tile_attributes(info, offsets, strides, source_shape, *, tile_shape=(32, 32)):
    """Describe a complete tile after the caller proves its bounds and ownership."""
    attrs = dict(info.attrs or {})
    attrs.update(
        dtype_level=0,
        application_shape=tuple(map(str, tile_shape)),
        application_ndim=2,
        source_shape=tuple(map(str, source_shape)),
        source_strides=tuple(strides),
        dtype_shapes=(tuple(map(str, tile_shape)),),
        dtype_target_dims=(("0", "1"),),
        target_dims=("0", "1"),
        access_templates=(
            {
                "level": 0,
                "shape": tuple(map(str, tile_shape)),
                "offsets": offsets,
                "mask": True,
                "linear_offset": f"({offsets[0]})*({strides[0]})+({offsets[1]})*({strides[1]})",
            },
        ),
    )

    return attrs


__all__ = [
    "DenseDotAccess",
    "dense_tile_attributes",
    "integer_interval",
    "predicate_always_true",
    "prove_dense_dot_access",
    "prove_full_matrix_access",
]
