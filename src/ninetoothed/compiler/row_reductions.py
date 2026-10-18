"""Backend-neutral grouping of already arranged independent reduction rows."""

from dataclasses import replace

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
            any(op.regions or op.opcode == "mem.atomic_add" for op in operations)
            or len(stores) != 1
        ):
            reason = "requires a straight-line graph with one store"
        elif stores[0] is not operations[-1] or set(stores[0].attrs) - {"target"}:
            reason = "requires one terminal whole-output store"
        else:
            values = {value.name: value for value in program.inputs}
            values.update((v.name, v) for op in operations for v in op.results)
            output = stores[0].operands[1]

            if any(output in op.operands for op in operations[:-1]):
                reason = "output is also read"
            elif tuple(map(str, values[output].type.shape)) != tuple(
                reduction.get("value_shape", ())
            ):
                reason = "requires output to match the reduction input domain"

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

            if group["groups_per_tile"] > 1 and any(
                op.opcode.startswith("linalg.") for op in operations
            ):
                group.update(
                    enabled=False, reason="row slicing requires elementwise consumers"
                )

        return replace(
            program, metadata=dict(program.metadata) | {"schedule": schedule}
        )
