"""Effect boundary shared by legality analyses and optional transforms."""

_POINTWISE_CALLS = frozenset(
    {
        "abs",
        "ceil",
        "cos",
        "erf",
        "exp",
        "exp2",
        "expm1",
        "floor",
        "log",
        "log2",
        "log1p",
        "pow",
        "reciprocal",
        "round",
        "rsqrt",
        "sigmoid",
        "sin",
        "sqrt",
        "tanh",
        "trunc",
    }
)


def broadcast_view_parts(operation):
    """Recognize full slices and inserted singleton axes without reindexing."""
    if operation.opcode != "tensor.view":
        return None

    parts = tuple(
        part.strip()
        for part in str(operation.attrs.get("subscript", "")).strip("()").split(",")
        if part.strip()
    )

    return parts if parts and all(part in {":", "None"} for part in parts) else None


def is_pointwise_expression(operation):
    """Identify value operations whose coordinates follow ordinary broadcasting."""
    return not operation.regions and (
        operation.opcode.startswith(("arith.", "cmp."))
        or (operation.opcode.startswith("math.") and operation.opcode != "math.rand")
        or operation.opcode in {"tensor.cast", "select.where"}
        or operation.opcode.startswith("call.")
        and operation.opcode[5:] in _POINTWISE_CALLS
    )


def pure_expression(operation):
    """Whitelist deterministic expressions; unknown operations are barriers."""
    return not operation.regions and (
        operation.opcode.startswith(("arith.", "cmp.", "reduce."))
        or (operation.opcode.startswith("math.") and operation.opcode != "math.rand")
        or operation.opcode
        in {
            "dtype.constant",
            "dtype.of",
            "dtype.select",
            "select.where",
            "shape.dim",
            "tensor.stride",
            "tensor.cast",
            "tensor.extract",
            "tensor.view",
            "tensor.zeros",
            "tensor.full",
            "index.offset",
        }
    )
