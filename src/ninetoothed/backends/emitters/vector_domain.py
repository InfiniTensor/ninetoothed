"""Prove a common flat source domain before replacing indexed accesses."""

import re

from ninetoothed.backends.emitters import ssa as common
from ninetoothed.compiler.effects import pure_expression
from ninetoothed.dtype import normalize_dtype


def prove_flat_domain(context):
    """Accept canonical 1-D tiles only, preserving their exact valid-lane mask."""
    import sympy

    program = context.kernel.ssa

    if (
        program is None
        or len(program.blocks) != 1
        or len(context.outputs) != 1
        or len(context.stores) != 1
        or len(context.axes) != 1
        or any(
            name.startswith("nt_")
            for name in (*context.variables, *context.outputs, *context.shape_params)
        )
    ):
        return None

    store = context.stores[0]

    if (
        store.opcode != "mem.store"
        or len(store.operands) != 2
        or store.operands[1] != context.outputs[0]
        or set(store.attrs) - {"target", "target_shape"}
    ):
        return None

    for operation in program.blocks[0].operations:
        if operation.regions or (
            operation is not store and not pure_expression(operation)
        ):
            return None

    extent = None

    for name in (*context.variables, *context.outputs):
        info = context.tensors[name]

        if info.ndim == 0:
            continue

        attrs = info.attrs or {}
        value_type = context.value_types.get(name)
        template = common._access_template(info, 0)

        if (
            info.ndim != 1
            or value_type is None
            or value_type.attrs.get("dtype_level", 0) != 0
            or tuple(map(str, value_type.shape)) != context.axes
            or normalize_dtype(value_type.dtype) != normalize_dtype(info.dtype)
            or len(info.source_shape) != 1
            or len(info.source_strides) != 1
            or attrs.get("source_ndim") != 1
            or tuple(map(str, attrs.get("source_shape", ()))) != info.source_shape
            or tuple(map(str, attrs.get("source_strides", ()))) != info.source_strides
            or attrs.get("dtype_level", 0) != 0
            or attrs.get("jagged_dim") is not None
            or attrs.get("jagged_offsets_param")
            or template is None
            or len(attrs.get("access_templates", ())) != 1
            or tuple(map(str, template.get("shape", ()))) != context.axes
            or template.get("jagged")
            or len(template.get("offsets", ())) != 1
        ):
            return None

        current = info.source_shape[0]

        if extent is not None and current != extent:
            return None

        extent = current
        texts = (
            current,
            context.axes[0],
            context.grid_total,
            *info.shape,
            str(template["offsets"][0]),
            str(template.get("mask", True)),
        )
        symbols = {
            n: sympy.Symbol(n, integer=True)
            for n in re.findall(r"[A-Za-z_]\w*", " ".join(texts))
            if n not in {"True", "False", "floor", "ceiling", "Mod"}
        }
        symbols.setdefault("outer_index", sympy.Symbol("outer_index", integer=True))
        symbols.setdefault("value_0", sympy.Symbol("value_0", integer=True))

        try:

            def parse(text):
                return sympy.sympify(str(text), locals=symbols)

            offset = parse(template["offsets"][0])
            expected = (
                symbols["outer_index"] * parse(context.axes[0]) + symbols["value_0"]
            )

            if sympy.simplify(offset - expected) != 0:
                return None

            tiles = sympy.ceiling(parse(current) / parse(context.axes[0]))
            outer, lane = symbols["outer_index"], symbols["value_0"]
            mask = parse(template.get("mask", True))
            terms = mask.args if isinstance(mask, sympy.And) else (mask,)
            covered = {
                sympy.true,
                outer >= 0,
                outer < tiles,
                lane >= 0,
                lane < parse(context.axes[0]),
                expected >= 0,
                expected < parse(current),
            }

            if any(term not in covered for term in terms):
                return None

            # The common scalar emitter uses this outer grid. Prove it covers
            # exactly the expected number of tiles, including the final tail.
            if (
                sympy.simplify(
                    parse(context.grid_total) - tiles * parse(context.axes[0])
                )
                != 0
            ):
                return None

            if (
                len(info.shape) != 1
                or sympy.simplify(parse(info.shape[0]) - tiles) != 0
            ):
                return None
        except (TypeError, ValueError, SyntaxError, sympy.SympifyError):
            return None

    return extent
