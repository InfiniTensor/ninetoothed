"""Prove a stored value is independent of tensor lanes and memory reads."""

from dataclasses import dataclass

from ninetoothed.dtype import normalize_dtype

_PURE = {
    "arith.constant",
    "arith.add",
    "arith.sub",
    "arith.mul",
    "arith.div",
    "arith.neg",
    "arith.pow",
    "arith.maximum",
    "arith.minimum",
    "arith.mod",
    "arith.floordiv",
    "arith.lt",
    "arith.le",
    "arith.gt",
    "arith.ge",
    "arith.eq",
    "arith.ne",
    "cmp.lt",
    "cmp.le",
    "cmp.gt",
    "cmp.ge",
    "cmp.eq",
    "cmp.ne",
    "select.where",
    "tensor.cast",
    "tensor.full",
    "tensor.zeros",
} | {
    "math." + name
    for name in (
        "abs",
        "sqrt",
        "rsqrt",
        "reciprocal",
        "exp",
        "exp2",
        "expm1",
        "log",
        "log2",
        "log1p",
        "sin",
        "cos",
        "tan",
        "tanh",
        "erf",
        "ceil",
        "floor",
        "sigmoid",
    )
}


@dataclass(frozen=True)
class UniformStore:
    value: str
    output: str
    dtype: str


def uniform_store(program):
    """Return one terminal same-domain float32 store with pure uniform data.

    Geometry and physical stride/alignment remain target obligations. Unknown
    operations, regions, index values, tensor loads and extra stores reject the
    entire graph; even unused effects cannot be elided by this proof.
    """
    if program is None or len(program.blocks) != 1:
        return None

    block = program.blocks[0]

    if block.args or not block.operations:
        return None

    store = block.operations[-1]

    if (
        store.opcode != "mem.store"
        or len(store.operands) != 2
        or store.regions
        or store.attrs.get("indices")
        or store.attrs.get("source")
        or store.attrs.get("target_level", 0) != 0
    ):
        return None

    inputs = {value.name: value for value in program.inputs}
    target = inputs.get(store.operands[1])

    if (
        target is None
        or normalize_dtype(target.type.dtype) != "float32"
        or target.type.kind != "tensor"
        or tuple(map(str, target.type.shape))
        != tuple(map(str, store.attrs.get("target_shape", ())))
    ):
        return None

    uniform = {
        name
        for name, value in inputs.items()
        if value.type.kind == "scalar"
        and not value.type.shape
        and normalize_dtype(value.type.dtype) in {"float32", "float16", "int32", "bool"}
    }
    types = {name: value.type for name, value in inputs.items()}

    for operation in block.operations[:-1]:
        if (
            operation.opcode not in _PURE
            or operation.regions
            or len(operation.results) != 1
            or any(name not in uniform for name in operation.operands)
        ):
            return None

        result = operation.results[0]

        if result.type.kind not in {"scalar", "tensor"}:
            return None

        uniform.add(result.name)
        types[result.name] = result.type

    value = store.operands[0]

    if value not in uniform or normalize_dtype(types[value].dtype) != "float32":
        return None
    return UniformStore(value, target.name, "float32")
