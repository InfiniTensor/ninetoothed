"""Shared numeric legality for native pointwise vector plans."""

FLOAT_DTYPES = frozenset({"float16", "float32"})
BINARY_OPERATORS = {"arith.add": "+", "arith.sub": "-", "arith.mul": "*"}
COMPARISON_OPERATORS = {
    "cmp.lt": "<",
    "cmp.le": "<=",
    "cmp.gt": ">",
    "cmp.ge": ">=",
    "cmp.eq": "==",
    "cmp.ne": "!=",
}


def classify_pointwise(operation, operands, dtype):
    code = operation.opcode

    if code in BINARY_OPERATORS and dtype in FLOAT_DTYPES and len(operands) == 2:
        if any(v.dtype not in FLOAT_DTYPES and v.literal is None for v in operands):
            return None

        if any(v.dtype != dtype and v.literal is None for v in operands):
            return "scalar" if dtype == "float32" else None

        if all(not v.scalar for v in operands):
            return "binary"

        if code in {"arith.add", "arith.mul"} and dtype == "float32":
            return "scalar_binary"

        return "scalar"

    if code in COMPARISON_OPERATORS and dtype == "bool" and len(operands) == 2:
        if all(v.dtype == "float32" or v.literal is not None for v in operands):
            return "scalar"

    if code == "select.where" and dtype == "float32" and len(operands) == 3:
        if operands[0].dtype == "bool" and all(
            v.dtype == dtype or v.literal is not None for v in operands[1:]
        ):
            return "scalar"

    if code == "tensor.cast" and len(operands) == 1:
        if (
            dtype in FLOAT_DTYPES
            and operands[0].dtype in FLOAT_DTYPES
            and not operands[0].scalar
        ):
            return "cast"

    return None
