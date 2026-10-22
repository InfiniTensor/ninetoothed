"""Row value provenance and resource accounting for a proved vector domain."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RowBatchPlan:
    row_plan: object
    batch_rows: int
    columns: int
    rows: int
    varying_scalars: tuple[str, ...]
    varying_inputs: tuple[str, ...]
    extra_scratch: bool
    nram_bytes: int

    @property
    def tasks(self):
        return (self.rows + self.batch_rows - 1) // self.batch_rows


def plan_row_batch(row_plan, varying_inputs=(), *, batch_rows=8, budget=192 * 1024):
    """Keep row statistics separate from uniform shape and scalar arguments.

    Access maps, row independence, effects and store ownership are proved by
    the caller. This plan tracks row-varying scalar def-use without reassociation
    and accounts for a private transpose buffer and padded statistic arrays.
    The target proves its channel packing and reduction primitive contracts.
    """
    try:
        rows, columns = int(row_plan.rows), int(row_plan.cols)
    except (TypeError, ValueError):
        return None

    dag = row_plan.dag
    varying = set(varying_inputs)
    scalar_inputs = {item.value.name for item in dag.inputs if item.value.scalar}

    if (
        not isinstance(batch_rows, int)
        or isinstance(batch_rows, bool)
        or batch_rows not in {8, 16}
        or rows < batch_rows
        or columns < 128
        or columns % 128 != 0
        or not varying <= scalar_inputs
        or any(
            dtype != "float32" for dtype in (*dag.buffer_dtypes, *dag.scratch_dtypes)
        )
        or any(store.value.dtype != "float32" for store in dag.stores)
        or any(
            item.value.dtype != "float32" and item.value.name in varying
            for item in dag.inputs
        )
    ):
        return None

    for step in dag.steps:
        node = step.node

        if (
            node.lowering.mode in {"where", "compare", "cast"}
            or node.result.dtype == "bool"
        ):
            return None

        if node.lowering.mode == "reduce":
            if node.operation.opcode not in {"reduce.sum", "reduce.max", "reduce.min"}:
                return None

            varying.add(node.result.name)
        elif node.result.scalar and any(
            value.name in varying for value in node.operands
        ):
            varying.add(node.result.name)

        if node.result.name in varying and node.result.dtype != "float32":
            return None

        if (
            not node.result.scalar
            and sum(value.name in varying for value in node.operands) > 1
        ):
            return None

    extra_scratch = not dag.scratch_dtypes
    capacity = batch_rows * columns
    nram_bytes = capacity * (dag.bytes_per_element + 4 + (4 if extra_scratch else 0))
    nram_bytes += (len(varying) + 2) * 128 * 4

    if nram_bytes > budget:
        return None

    return RowBatchPlan(
        row_plan,
        batch_rows,
        columns,
        rows,
        tuple(sorted(varying)),
        tuple(sorted(varying_inputs)),
        extra_scratch,
        nram_bytes,
    )
