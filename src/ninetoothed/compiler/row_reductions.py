"""Backend-neutral grouping of already arranged independent reduction rows."""

from dataclasses import replace
from functools import cache

from ninetoothed.compiler.passes import HARDWARE_INDEPENDENT, Pass


class GroupReductionRows(Pass):
    """Partition arranged independent rows into physical execution groups.

    Insert after ``ssa.select_schedule`` and before backend optimization. The
    backend selects a legal implementation or retains its original schedule.
    The optional ``rows`` divides the logical row tile without changing its
    shape or arithmetic. By default, one group covers the entire logical tile.
    """

    name = "ssa.group_reduction_rows"
    category = HARDWARE_INDEPENDENT
    phase = "schedule_intent"
    default_enabled = False

    def run(self, program, context):
        options = dict(context.pass_options.get(self.name, {}))

        if options.keys() - {"rows"}:
            raise ValueError("Row grouping only accepts a physical row count.")

        schedule = dict(program.metadata.get("schedule", {}))
        reduction = schedule.get("reduction", {})
        shape = reduction.get("parallel_shape", ())
        operations = program.blocks[0].operations
        stores = tuple(op for op in operations if op.opcode == "mem.store")
        reason = None

        if not (
            reduction.get("mode") == "row-vector"
            and len(shape) == 1
            and str(shape[0]).isdigit()
            and int(str(shape[0])) > 1
        ):
            reason = "requires multiple arranged independent reduction rows"
        elif (
            any(
                op.regions
                or not (
                    op.opcode.startswith(("arith.", "cmp.", "math.", "dtype."))
                    or op.opcode
                    in {
                        "reduce.sum",
                        "reduce.max",
                        "reduce.min",
                        "tensor.cast",
                        "tensor.view",
                        "shape.dim",
                        "select.where",
                        "mem.store",
                    }
                )
                for op in operations
            )
            or len(stores) != 1
        ):
            reason = "requires a straight-line reduction graph with one store"
        elif any(
            tensor.jagged_dim is not None
            or tensor.layout is None
            or len(tensor.layout.levels) != 1
            or tensor.attrs.get("layout_operations") is None
            or set(tensor.attrs["layout_operations"])
            - {
                "tile",
                "expand",
                "unsqueeze",
                "squeeze",
            }
            or tensor.attrs.get("custom_tiling", True)
            for tensor in context.tensors
            if tensor.attrs.get("source_ndim", tensor.ndim)
        ):
            reason = (
                "requires a single-level layout with regular tiling and broadcasting"
            )
        elif any(
            op.opcode == "tensor.view"
            and (
                op.attrs.get("source")
                or any(
                    part.strip() not in {":", "None"}
                    for part in str(op.attrs.get("subscript", ""))
                    .strip("()")
                    .split(",")
                )
            )
            for op in operations
        ):
            reason = "only broadcast-axis views are grouped"
        elif stores[0] is not operations[-1] or set(stores[0].attrs) - {
            "target",
            "target_shape",
        }:
            reason = "requires one terminal whole-output store"
        else:
            values = {value.name: value for value in program.inputs}
            values.update((v.name, v) for op in operations for v in op.results)
            output = stores[0].operands[1]

            target_shape = stores[0].attrs.get("target_shape")

            if target_shape is not None and tuple(map(str, target_shape)) != tuple(
                map(str, values[output].type.shape)
            ):
                reason = "requires a whole-output store domain"
            elif any(output in op.operands for op in operations[:-1]):
                reason = "output is also read"
            elif tuple(map(str, values[output].type.shape)) != tuple(
                reduction.get("value_shape", ())
            ):
                reason = "requires output to match the reduction input domain"
            elif not _matching_reduction_domains(operations, values, context.tensors):
                reason = (
                    "requires matching reduction input domains without broadcasting"
                )

        schedule["row_group"] = {
            "enabled": reason is None,
            "rows": int(str(shape[0])) if reason is None else 1,
            "reason": reason or "independent arranged reduction rows",
        }
        group = schedule["row_group"]

        if group["enabled"]:
            rows = options.get("rows", group["rows"])

            if (
                isinstance(rows, bool)
                or not isinstance(rows, int)
                or rows < 1
                or group["rows"] % rows
            ):
                raise ValueError("Physical rows must divide the logical row tile.")

            group["groups_per_tile"] = group["rows"] // rows
            group["rows"] = rows

        return replace(
            program, metadata=dict(program.metadata) | {"schedule": schedule}
        )


def _matching_reduction_domains(operations, values, tensors):
    """Prove mask agreement without introducing broadcast-domain analysis."""
    inputs = {tensor.name: tensor for tensor in tensors}
    producers = {value.name: op for op in operations for value in op.results}

    @cache
    def domains(name):
        if name in inputs:
            tensor = inputs[name]

            if tensor.attrs.get("source_ndim", tensor.ndim) == 0:
                return frozenset()

            layout = tensor.layout

            if layout is None or len(layout.value_accesses) != 1:
                return None

            return frozenset(
                {
                    (
                        tuple(dim.render() for dim in layout.application_shape),
                        layout.value_accesses[0].predicate.render(),
                    )
                }
            )

        op = producers.get(name)

        if op is None:
            return None if name.startswith("%") else frozenset()

        if op.opcode.startswith(("dtype.", "shape.", "reduce.")):
            return frozenset()

        result = frozenset()

        for operand in op.operands:
            dependency = domains(operand)

            if dependency is None or (
                dependency and values[operand].type.shape != values[name].type.shape
            ):
                return None

            result |= dependency

        return result if len(result) <= 1 else None

    return all(
        domains(op.operands[0]) is not None
        for op in operations
        if op.opcode.startswith("reduce.")
    )
