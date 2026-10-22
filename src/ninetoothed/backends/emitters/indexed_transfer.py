"""Prove a raw pointer indexed load with a single canonical output domain."""

from dataclasses import dataclass

from ninetoothed.backends.emitters.expressions import normalize_dtype
from ninetoothed.backends.emitters.staging import canonical_flat_view


@dataclass(frozen=True)
class IndexedTransfer:
    source: str
    indices: str
    output: str
    extent: str
    guard: str


def plan_indexed_transfer(context):
    """Reject masks, effects, computed indices and noncanonical output views.

    The pointer is a raw data pointer, so its addressing deliberately does not
    inherit the source tensor's arrangement or element stride. Index and output
    views do inherit those properties and require a complete physical prefix.
    """
    blocks = context.kernel.ssa.blocks

    if len(blocks) != 1 or len(context.outputs) != 1:
        return None

    operations = blocks[0].operations

    if len(operations) != 4 or any(op.regions for op in operations):
        return None

    pointer, add, load, store = operations

    if (
        pointer.opcode != "mem.data_ptr"
        or len(pointer.operands) != 1
        or len(pointer.results) != 1
        or pointer.attrs
        or add.opcode != "arith.add"
        or len(add.operands) != 2
        or len(add.results) != 1
        or add.attrs
        or load.opcode != "mem.load"
        or load.operands != (add.results[0].name,)
        or len(load.results) != 1
        or load.attrs
        or store.opcode != "mem.store"
        or store.operands != (load.results[0].name, context.outputs[0])
        or store.results
        or set(store.attrs) - {"target", "target_shape"}
    ):
        return None

    source = pointer.operands[0]
    pointer_name = pointer.results[0].name

    if source not in context.variables or pointer_name not in add.operands:
        return None

    index_name = add.operands[1] if add.operands[0] == pointer_name else add.operands[0]
    output = context.outputs[0]

    if index_name not in context.variables:
        return None

    for result in (pointer.results[0], add.results[0]):
        if (
            result.type.kind != "pointer"
            or normalize_dtype(result.type.dtype) != "float32"
        ):
            return None

    if normalize_dtype(load.results[0].type.dtype) != "float32":
        return None

    if normalize_dtype(context.tensors[source].dtype) != "float32":
        return None

    index_info = context.tensors[index_name]
    output_info = context.tensors[output]
    index_shape = tuple(
        map(str, (index_info.attrs or {}).get("application_shape", index_info.shape))
    )
    output_shape = tuple(
        map(str, (output_info.attrs or {}).get("application_shape", output_info.shape))
    )

    if (
        pointer.results[0].type.shape
        or index_shape != output_shape
        or tuple(map(str, add.results[0].type.shape)) != index_shape
        or tuple(map(str, load.results[0].type.shape)) != output_shape
        or ("target" in store.attrs and store.attrs["target"] != output)
        or (
            "target_shape" in store.attrs
            and tuple(map(str, store.attrs["target_shape"])) != output_shape
        )
    ):
        return None

    predicates = []
    extents = []

    for name, dtype in ((index_name, "int32"), (output, "float32")):
        info = context.tensors[name]

        if normalize_dtype(info.dtype) != dtype or not canonical_flat_view(info):
            return None

        attrs = info.attrs or {}

        if attrs.get("dtype_level", 0) != 0:
            return None

        shape = attrs.get("source_shape") or info.source_shape or info.shape
        strides = info.source_strides or attrs.get("source_strides", ())

        if len(shape) != 1 or len(strides) != 1:
            return None

        extents.append(str(shape[0]))
        predicates.append(f"({strides[0]}) == 1")

    predicates.extend(
        (f"({extents[0]}) == ({extents[1]})", f"({context.total}) >= ({extents[1]})")
    )

    return IndexedTransfer(
        source, index_name, output, extents[1], " && ".join(predicates)
    )
