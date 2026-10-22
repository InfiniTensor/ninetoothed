"""Prove a complete physical matrix transpose from input/output access maps."""

import re
from dataclasses import dataclass

from ninetoothed.backends.emitters.dot_layout import (
    predicate_always_true,
    prove_dense_dot_access,
)
from ninetoothed.dtype import normalize_dtype


@dataclass(frozen=True)
class TransposeTransfer:
    source: str
    output: str
    rows: int
    columns: int
    source_strides: tuple[str, str]
    output_strides: tuple[str, str]
    guard: str

    @property
    def byte_span(self):
        return self.rows * self.columns * 4


def plan_transpose_transfer(context):
    """Prove the same valid reads and writes before choosing a tiled schedule.

    Only one direct float32 store is accepted. No operator name or generated
    source pattern selects the plan. The caller proves disjoint source/output
    buffers, target DMA ownership, primitive legality and private resources.
    """
    import sympy

    program = context.kernel.ssa

    if (
        program is None
        or len(program.blocks) != 1
        or len(program.blocks[0].operations) != 1
        or len(context.outputs) != 1
        or len(context.stores) != 1
    ):
        return None

    store = program.blocks[0].operations[0]

    if (
        store is not context.stores[0]
        or store.opcode != "mem.store"
        or store.regions
        or store.results
        or len(store.operands) != 2
        or set(store.attrs) - {"target", "target_shape"}
    ):
        return None

    source, output = store.operands

    if (
        source not in context.variables
        or output != context.outputs[0]
        or source == output
        or store.attrs.get("target", output) != output
    ):
        return None

    infos = (context.tensors.get(source), context.tensors.get(output))

    if any(
        info is None or info.ndim != 2 or normalize_dtype(info.dtype) != "float32"
        for info in infos
    ):
        return None

    try:
        rows, columns = map(int, infos[0].attrs["source_shape"])
    except (KeyError, TypeError, ValueError):
        return None

    if (
        rows <= 0
        or columns <= 0
        or rows * columns > ((1 << 63) - 1) // 4
        or tuple(map(str, context.axes)) != ("1", str(rows))
        or tuple(map(str, context.outer_axes)) != (str(columns), "1")
        or tuple(map(str, store.attrs.get("target_shape", ()))) != ("1", str(rows))
    ):
        return None

    accesses = []

    for info, shape, offsets in (
        (infos[0], (rows, columns), ("value_1", "outer_index + value_0")),
        (infos[1], (columns, rows), ("outer_index + value_0", "value_1")),
    ):
        attrs = info.attrs or {}

        if (
            tuple(map(str, attrs.get("source_shape", ()))) != tuple(map(str, shape))
            or tuple(map(str, attrs.get("application_shape", ()))) != ("1", str(rows))
            or tuple(map(str, attrs.get("view_shape", ()))) != (str(columns), "1")
            or attrs.get("dtype_level", 0) != 0
            or attrs.get("jagged_offsets_param")
        ):
            return None

        access = prove_dense_dot_access(
            attrs, shape=(1, rows), tasks=columns, stages=1, expected_offsets=offsets
        )

        if access is None or any(
            not re.fullmatch(r"[A-Za-z_]\w*|[1-9]\d*", stride)
            for stride in access.strides
        ):
            return None

        template = attrs["access_templates"][0]
        view_mask = re.sub(
            r"\bindex\b", "outer_index", str(attrs.get("view_mask", True))
        )
        text = " ".join(
            (
                *access.offsets,
                *access.strides,
                str(template.get("linear_offset", "")),
                view_mask,
            )
        )
        symbols = {
            name: sympy.Symbol(name, integer=True)
            for name in re.findall(r"[A-Za-z_]\w*", text)
            if name not in {"Mod", "floor", "True", "False"}
        }

        def parse(expression):
            return sympy.sympify(str(expression), locals=symbols)

        try:
            expected_linear = sum(
                parse(offset) * parse(stride)
                for offset, stride in zip(access.offsets, access.strides)
            )

            if sympy.simplify(parse(template["linear_offset"]) - expected_linear) != 0:
                return None

            domains = {
                symbols["outer_index"]: (0, columns - 1),
                symbols["value_0"]: (0, 0),
                symbols["value_1"]: (0, rows - 1),
            }

            if not predicate_always_true(parse(view_mask), domains):
                return None
        except (KeyError, TypeError, ValueError, AttributeError, SyntaxError):
            return None

        accesses.append(access)

    src, dst = accesses
    guard = " && ".join(
        f"(({stride}) == {expected})"
        for strides, widths in ((src.strides, (columns, 1)), (dst.strides, (rows, 1)))
        for stride, expected in zip(strides, widths)
    )

    return TransposeTransfer(
        source, output, rows, columns, src.strides, dst.strides, guard
    )
