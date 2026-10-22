"""Prove row-local input maps, including singleton broadcast dimensions."""

import re
from dataclasses import dataclass

from ninetoothed.backends.emitters.dot_layout import (
    predicate_always_true,
    prove_dense_dot_access,
)


@dataclass(frozen=True)
class RowInputLayout:
    mode: str
    row_index: str
    source_rows: int
    source_columns: int
    strides: tuple[str, str]
    guard: str

    def base_offset(self, row_expression):
        """Bind the caller's task coordinate only after proving its row map."""
        return (
            f"({row_expression}) * ({self.strides[0]})"
            if self.row_index == "row"
            else "0"
        )


def prove_row_input(context, name, rows, columns):
    """Prove complete static R×K/R×1/1×K/1×1 inputs for a 1×K domain.

    This proves reads only. The caller still proves its pure def-use graph,
    output maps, row task ownership, dtype support and device resource limits.
    Scalar broadcasts read one original value per row; vector broadcasts
    read their original row zero. No source predicate is discarded.
    """
    import sympy

    if (
        not isinstance(rows, int)
        or not isinstance(columns, int)
        or rows <= 0
        or columns <= 0
        or name not in context.variables
        or tuple(map(str, context.axes)) != ("1", str(columns))
        or tuple(map(str, context.outer_axes)) != (str(rows), "1")
    ):
        return None

    info = context.tensors.get(name)
    attrs = dict(info.attrs or {}) if info is not None else {}

    if (
        info is None
        or info.ndim != 2
        or attrs.get("source_ndim") != 2
        or attrs.get("dtype_level", 0) != 0
        or attrs.get("jagged_dim") is not None
        or attrs.get("jagged_offsets_param")
        or tuple(map(str, attrs.get("target_dims", ()))) != ("0", "1")
        or tuple(map(str, attrs.get("view_shape", ()))) != (str(rows), "1")
    ):
        return None

    try:
        source_rows, source_columns = map(int, attrs["source_shape"])
    except (KeyError, TypeError, ValueError):
        return None

    shape = ("1", str(source_columns))
    value_type = context.value_types.get(name)

    if (
        source_rows not in {1, rows}
        or source_columns not in {1, columns}
        or tuple(map(str, attrs.get("application_shape", ()))) != shape
        or value_type is None
        or tuple(map(str, value_type.shape or ())) != shape
    ):
        return None

    row_index = "0" if source_rows == 1 else "row"
    expected = (
        "value_0" if row_index == "0" else "outer_index + value_0",
        "value_1",
    )
    access = prove_dense_dot_access(
        attrs,
        shape=(1, source_columns),
        tasks=rows,
        stages=1,
        expected_offsets=expected,
    )

    if access is None:
        return None

    if any(
        not re.fullmatch(r"[A-Za-z_]\w*|[1-9]\d*", stride) for stride in access.strides
    ):
        return None

    template = attrs["access_templates"][0]
    text = " ".join(
        (
            *access.offsets,
            *access.strides,
            str(template.get("linear_offset", "")),
            str(attrs.get("view_linear_offset", "")),
            str(attrs.get("view_mask", True)),
            *map(str, attrs.get("view_offsets", ())),
        )
    )
    symbols = {
        item: sympy.Symbol(item, integer=True)
        for item in re.findall(r"[A-Za-z_]\w*", text)
        if item not in {"floor", "Mod", "True", "False"}
    }

    for item in ("outer_index", "value_0", "value_1"):
        symbols.setdefault(item, sympy.Symbol(item, integer=True))

    def parse(expression):
        return sympy.sympify(str(expression), locals=symbols)

    try:
        linear = sum(
            parse(offset) * parse(stride)
            for offset, stride in zip(access.offsets, access.strides)
        )

        if sympy.simplify(parse(template["linear_offset"]) - linear) != 0:
            return None

        if sympy.simplify(parse(attrs["view_linear_offset"])) != 0:
            return None

        if len(attrs.get("view_offsets", ())) != 2 or any(
            sympy.simplify(parse(offset)) != 0 for offset in attrs["view_offsets"]
        ):
            return None

        view_mask = re.sub(
            r"\bindex\b", "outer_index", str(attrs.get("view_mask", True))
        )
        domains = {
            symbols["outer_index"]: (0, rows - 1),
            symbols["value_0"]: (0, 0),
            symbols["value_1"]: (0, source_columns - 1),
        }

        if not predicate_always_true(parse(view_mask), domains):
            return None
    except (KeyError, TypeError, ValueError, AttributeError, SyntaxError):
        return None

    row_stride, column_stride = access.strides
    guard = f"(({row_stride}) == {source_columns}) && (({column_stride}) == 1)"

    return RowInputLayout(
        "scalar" if source_columns == 1 else "vector",
        row_index,
        source_rows,
        source_columns,
        access.strides,
        guard,
    )
