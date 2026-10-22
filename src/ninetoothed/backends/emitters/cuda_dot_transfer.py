"""Prove complete dot tiles for guarded direct half memory staging."""

from dataclasses import dataclass

from ninetoothed.backends.emitters.dot_layout import prove_dense_dot_access
from ninetoothed.dtype import normalize_dtype


@dataclass(frozen=True)
class DotTransfers:
    left: str
    right: str
    left_width: int
    right_width: int
    left_base: tuple[str, str]
    right_base: tuple[str, str]
    guard: str


def _integer_bound(name, context):
    operation = context.operations.get(name)

    if operation is None:
        return None

    if operation.opcode == "arith.constant":
        value = operation.attrs.get("value")

        return value if isinstance(value, int) and not isinstance(value, bool) else None

    if (
        operation.opcode == "shape.dim"
        and len(operation.operands) == 1
        and not operation.attrs.get("source")
    ):
        try:
            return int(
                context.value_types[operation.operands[0]].shape[
                    int(operation.attrs["dim"])
                ]
            )
        except (KeyError, IndexError, TypeError, ValueError):
            return None
    return None


def prove_dot_transfers(loop, dot, context, *, stage, outer):
    """Require full original maps and masks; the caller retains region semantics."""
    if tuple(map(str, context.output_axes)) != ("32", "32") or len(loop.operands) != 4:
        return None

    lower, stages, step = (_integer_bound(name, context) for name in loop.operands[:3])

    if lower != 0 or step != 1 or stages is None or not 0 < stages <= 4096:
        return None

    output = context.tensor_infos.get(context.output)

    if output is None:
        return None

    try:
        rows, columns = map(int, output.source_shape)
    except (TypeError, ValueError):
        return None

    if rows <= 0 or columns <= 0 or rows % 32 or columns % 32:
        return None

    tasks = (rows // 32) * (columns // 32)
    row = f"outer_index // {columns // 32} * 32 + value_0"
    column = f"outer_index % {columns // 32} * 32 + value_1"
    output_proof = prove_dense_dot_access(
        output.attrs or {},
        shape=(32, 32),
        tasks=tasks,
        stages=stages,
        expected_offsets=(row, column),
    )

    if output_proof is None or output_proof.source_shape != (rows, columns):
        return None

    input_names = {value.name for value in context.program.inputs}
    accesses = []
    names = []

    for operand, expected, physical in (
        (dot.operands[0], (row, "extract_0_0 * 32 + value_1"), (rows, stages * 32)),
        (
            dot.operands[1],
            ("extract_0_0 * 32 + value_0", column),
            (stages * 32, columns),
        ),
    ):
        extract = context.operations.get(operand)

        if (
            extract is None
            or extract.opcode != "tensor.extract"
            or extract.operands[1:] != (str(loop.attrs.get("induction")),)
            or extract.operands[0] not in input_names
            or normalize_dtype(extract.results[0].type.dtype) != "float16"
        ):
            return None

        info = context.tensor_infos.get(extract.operands[0])

        if info is None or normalize_dtype(info.dtype) != "float16":
            return None

        proof = prove_dense_dot_access(
            extract.results[0].type.attrs,
            shape=(32, 32),
            tasks=tasks,
            stages=stages,
            expected_offsets=expected,
        )

        if proof is None or proof.source_shape != physical:
            return None

        accesses.append(proof)
        names.append(extract.operands[0])

    guard = " && ".join(
        f"({access.strides[0]}) == {access.source_shape[1]} && ({access.strides[1]}) == 1"
        for access in accesses
    )
    bases = tuple(
        access.base_offsets({"outer_index": outer, "extract_0_0": stage})
        for access in accesses
    )

    return DotTransfers(*names, stages * 32, columns, *bases, guard)


__all__ = ["DotTransfers", "prove_dot_transfers"]
