"""Effect boundary shared by legality analyses and optional transforms."""


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
