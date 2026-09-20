"""Opt-in block scheduling for small, straight-line reduction graphs.

Insert ``ssa.triton.block_reductions`` after the default schedule passes and
before target validation through ``CompileRequest.pipeline``. The default
pipeline is unchanged. Rejected graphs retain their original schedule, with
the reason recorded in ``schedule.block_reductions``.

``CompileRequest.pass_options`` can set ``max_block_elements`` (default 4096)
and ``max_total_elements`` (default 32768) for this pass. The latter bounds the
sum of SSA value sizes, conservatively estimating register pressure rather than
predicting occupancy. The block size also has a hard limit of 65536 elements.
These limits are candidate controls, not an automatic tuning policy.

Only FP32 reductions with explicit axes are scheduled. Parallel summation can
change rounding relative to the serial fallback; it is not bitwise equivalent.
"""

from dataclasses import replace
from math import prod

from ninetoothed.backends.core import Target
from ninetoothed.compiler.passes import LANGUAGE_SPECIFIC, Pass
from ninetoothed.dtype import normalize_dtype

_OPERATIONS = frozenset(
    {
        "arith.constant",
        "arith.add",
        "arith.sub",
        "arith.mul",
        "arith.div",
        "arith.neg",
        "arith.maximum",
        "arith.minimum",
        "math.exp",
        "math.exp2",
        "math.log",
        "math.log2",
        "math.sqrt",
        "math.rsqrt",
        "tensor.view",
        "linalg.transpose",
        "shape.dim",
        "tensor.cast",
        "reduce.sum",
        "reduce.max",
        "reduce.min",
        "mem.store",
    }
)


class TritonBlockReductions(Pass):
    """Evaluate bounded reduction graphs once per arranged output block."""

    name = "ssa.triton.block_reductions"
    category = LANGUAGE_SPECIFIC
    phase = "optimization"
    supported_backends = (Target.TRITON,)
    default_enabled = False

    def run(self, program, context):
        options = dict(context.pass_options.get(self.name, {}))
        unknown = options.keys() - {"max_block_elements", "max_total_elements"}

        if unknown:
            raise ValueError(f"Unknown block reduction options: {sorted(unknown)}.")

        limits = {"max_block_elements": 4096, "max_total_elements": 32768} | options

        for name, value in limits.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Block reduction option `{name}` must be positive.")

        schedule = dict(program.metadata.get("schedule", {}))
        reason = _rejection(program, context, schedule, **limits)
        schedule["block_reductions"] = {
            "enabled": reason is None,
            "reason": reason or "bounded straight-line reduction graph",
            **limits,
        }

        return replace(
            program, metadata=dict(program.metadata) | {"schedule": schedule}
        )


def _rejection(program, context, schedule, *, max_block_elements, max_total_elements):
    if context.resolved_target.backend != Target.TRITON:
        return "requires the Triton backend"

    if schedule.get("reduction", {}).get("mode") == "row-vector":
        return "preserve the existing row-vector schedule"

    operations = program.blocks[0].operations
    reductions = tuple(op for op in operations if op.opcode.startswith("reduce."))

    if not reductions:
        return "no reductions"

    if any(op.regions or op.opcode not in _OPERATIONS for op in operations):
        return "unsupported operation or region"

    stores = tuple(op for op in operations if op.opcode == "mem.store")

    if len(stores) != 1 or stores[0] is not operations[-1]:
        return "requires one terminal store"

    store = stores[0]

    if set(store.attrs) - {"target"}:
        return "requires a whole arranged output store"

    output = store.operands[1]

    if any(output in op.operands for op in operations[:-1]):
        return "output is also read"

    values = {value.name: value for value in program.inputs}
    values.update((value.name, value) for op in operations for value in op.results)
    total_elements = 0

    for value in values.values():
        shape = _static_shape(value.type.shape)

        if shape is None or len(shape) > 2:
            return "requires static rank-zero, rank-one, or rank-two values"

        if any(extent & (extent - 1) for extent in shape):
            return "requires power-of-two block dimensions"

        elements = prod(shape)

        if elements > min(max_block_elements, 65536):
            return "block element budget exceeded"

        total_elements += elements

    if total_elements > max_total_elements:
        return "total value element budget exceeded"

    for op in operations:
        if op.opcode == "tensor.view":
            parts = str(op.attrs.get("subscript", "")).strip("()").split(",")

            if op.attrs.get("source") or any(
                part.strip() not in {":", "None"} for part in parts
            ):
                return "only broadcast-axis views are supported"

    for op in reductions:
        shape = values[op.operands[0]].type.shape
        axis = op.attrs.get("axis")

        if normalize_dtype(op.results[0].type.dtype) != "float32":
            return "requires FP32 reduction results"

        if isinstance(axis, bool) or not isinstance(axis, int):
            return "requires an explicit reduction axis"

        if not shape or not -len(shape) <= axis < len(shape):
            return "requires a ranked reduction operand"

    tensors = {tensor.name: tensor for tensor in context.tensors}
    outer_sizes = set()

    for value in program.inputs:
        if value.type.kind != "tensor":
            continue

        tensor = tensors.get(value.name)
        layout = None if tensor is None else tensor.layout

        if layout is None or len(layout.levels) != 1 or tensor.jagged_dim is not None:
            return "requires a single-level dense arranged layout"

        outer = _static_shape(tuple(axis.render() for axis in layout.view_shape))

        if outer is None:
            return "requires a static outer program domain"

        outer_sizes.add(prod(outer))

    if len(outer_sizes) != 1:
        return "input and output program domains differ"

    if output not in tensors or not values[output].type.shape:
        return "requires a ranked arranged output"

    if values[store.operands[0]].type.shape != values[output].type.shape:
        return "store value must match the arranged output shape"

    return None


def _static_shape(shape):
    try:
        dimensions = tuple(int(str(axis)) for axis in shape)
    except (TypeError, ValueError):
        return None

    return dimensions if all(axis > 0 for axis in dimensions) else None
