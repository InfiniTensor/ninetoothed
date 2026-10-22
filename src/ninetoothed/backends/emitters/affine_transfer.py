"""Prove complete one-dimensional copy domains with a fixed source offset."""

import re
from dataclasses import dataclass

from ninetoothed.backends.emitters.expressions import normalize_dtype
from ninetoothed.backends.emitters.staging import canonical_flat_view


@dataclass(frozen=True)
class AffineTransfer:
    source: str
    output: str
    offset: int
    extent: int
    source_stride: str
    output_stride: str


def affine_transfer_memory_guard(plan, *, max_source_stride):
    """Keep the entire DMA read span separate from the dense output domain."""
    max_elements = ((1 << 63) - 1) // 4

    if (
        plan.offset < 0
        or plan.extent <= 0
        or plan.extent > max_elements
        or max_source_stride <= 0
    ):
        return "false"

    last_element = plan.offset + plan.extent - 1
    safe_stride = (
        min(max_source_stride, (max_elements - 1) // last_element)
        if last_element
        else max_source_stride
    )

    if safe_stride <= 0:
        return "false"

    stride = f"((uintptr_t)({plan.source_stride}))"
    source = f"((uintptr_t){plan.source})"
    output = f"((uintptr_t){plan.output})"
    first_offset = f"({plan.offset * 4}ULL * {stride})"
    start = f"({source} + {first_offset})"
    read_bytes = f"(({plan.extent - 1}ULL * {stride} + 1ULL) * 4ULL)"
    output_bytes = f"{plan.extent * 4}ULL"
    maximum = "~((uintptr_t)0)"

    return " && ".join(
        (
            f"(({plan.source_stride}) > 0)",
            f"(({plan.source_stride}) <= {safe_stride})",
            f"({source} <= {maximum} - {first_offset})",
            f"({start} <= {maximum} - ({read_bytes} - 1ULL))",
            f"({output} <= {maximum} - ({output_bytes} - 1ULL))",
            f"(({output} >= {start} && {output} - {start} >= {read_bytes}) || "
            f"({start} >= {output} && {start} - {output} >= {output_bytes}))",
        )
    )


def plan_affine_transfer(context):
    """Preserve every source predicate and prove all valid output lanes exist."""
    import sympy

    blocks = context.kernel.ssa.blocks

    if len(blocks) != 1 or len(blocks[0].operations) != 1 or len(context.outputs) != 1:
        return None

    store = blocks[0].operations[0]

    if (
        store.opcode != "mem.store"
        or store.regions
        or store.results
        or len(store.operands) != 2
        or store.operands[0] not in context.variables
        or store.operands[1] != context.outputs[0]
        or set(store.attrs) - {"target", "target_shape"}
    ):
        return None

    source, output = store.operands
    src, dst = context.tensors[source], context.tensors[output]
    attrs = src.attrs or {}
    templates = attrs.get("access_templates", ())

    if (
        normalize_dtype(src.dtype) != "float32"
        or normalize_dtype(dst.dtype) != "float32"
        or src.ndim != 1
        or not canonical_flat_view(dst)
        or len(templates) != 1
        or attrs.get("dtype_level", 0) != 0
        or attrs.get("jagged_dim") is not None
        or attrs.get("jagged_offsets_param")
    ):
        return None

    template = templates[0]
    strides = src.source_strides or attrs.get("source_strides", ())
    output_strides = dst.source_strides or (dst.attrs or {}).get("source_strides", ())
    offsets = template.get("offsets", ())

    if (
        len(strides) != 1
        or len(output_strides) != 1
        or len(offsets) != 1
        or template.get("level", 0) != 0
    ):
        return None

    try:
        (source_extent,) = map(int, attrs.get("source_shape") or src.source_shape)
        (extent,) = map(
            int, (dst.attrs or {}).get("source_shape") or dst.source_shape or dst.shape
        )
        (block,) = map(int, template.get("shape", ()))
        (view_count,) = map(int, attrs.get("view_shape", ()))
    except (ValueError, TypeError):
        return None

    if extent <= 0 or block <= 0 or view_count != (extent + block - 1) // block:
        return None

    output_shape = tuple(
        map(str, (dst.attrs or {}).get("application_shape", dst.shape))
    )

    if (
        output_shape != (str(block),)
        or ("target" in store.attrs and store.attrs["target"] != output)
        or (
            "target_shape" in store.attrs
            and tuple(map(str, store.attrs["target_shape"])) != output_shape
        )
    ):
        return None

    text = " ".join(
        (
            str(offsets[0]),
            str(template.get("linear_offset", "")),
            str(template.get("mask", True)),
            str(src.view_mask or True),
            str(strides[0]),
        )
    )
    symbols = {
        name: sympy.Symbol(name, integer=True)
        for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)
        if name not in {"floor", "Mod", "True", "False"}
    }
    outer = symbols.setdefault("outer_index", sympy.Symbol("outer_index", integer=True))
    inner = symbols.setdefault("value_0", sympy.Symbol("value_0", integer=True))
    flat = outer * block + inner

    try:

        def parse(value):
            return sympy.sympify(str(value), locals=symbols)

        offset = sympy.simplify(parse(offsets[0]) - flat)

        if not offset.is_Integer or offset < 0 or offset + extent > source_extent:
            return None

        stride = parse(strides[0])

        if (
            sympy.simplify(
                parse(template.get("linear_offset", "")) - (flat + offset) * stride
            )
            != 0
        ):
            return None

        allowed = (
            outer >= 0,
            outer < view_count,
            inner >= 0,
            inner < block,
            flat >= 0,
            flat < extent,
            flat + offset >= 0,
            flat + offset < source_extent,
        )
        mask = parse(template.get("mask", True))

        if src.view_mask:
            mask &= parse(re.sub(r"\bindex\b", "outer_index", src.view_mask))

        terms = mask.args if isinstance(mask, sympy.And) else (mask,)

        if not all(
            term == sympy.true
            or any(
                getattr(term, "rel_op", None) == getattr(bound, "rel_op", None)
                and sympy.simplify(term.lhs - bound.lhs) == 0
                and sympy.simplify(term.rhs - bound.rhs) == 0
                for bound in allowed
            )
            for term in terms
        ):
            return None
    except (TypeError, ValueError, SyntaxError, AttributeError, sympy.SympifyError):
        return None
    return AffineTransfer(
        source, output, int(offset), extent, str(strides[0]), str(output_strides[0])
    )
