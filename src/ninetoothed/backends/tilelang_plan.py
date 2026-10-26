"""Pure SSA eligibility and resource planning for TileLang source emission.

Scheduling classifies dataflow before symbolic tile specialization. Concrete
shape, layout and resource checks run once afterwards, before source rendering.
"""

from dataclasses import asdict, dataclass
from math import prod

from ninetoothed.backends.emitters.analysis import dot_accumulation_plan


@dataclass(frozen=True)
class TilePlan:
    """Serializable choices consumed by the renderer and launch metadata."""

    requested: str | None
    actual: str
    reason: str | None
    threads: int
    grid: str
    shared_memory_bytes: int = 0
    loop: str | None = None
    dot: str | None = None

    def as_metadata(self):
        return asdict(self)


INLINE_OPS = {
    "arith.constant",
    "dtype.of",
    "dtype.constant",
    "shape.dim",
    "index.offset",
    "tensor.zeros",
    "tensor.full",
    "tensor.extract",
    "tensor.view",
    "linalg.transpose",
}


def static_shape(type_):
    try:
        return tuple(int(dim) for dim in type_.shape)
    except (TypeError, ValueError):
        return None


def _materialized_reason(
    program, tensors, value_types, operations, stores, threads, shared_limit
):
    """Prove the bounded, effect-free subset before generating any source."""
    if threads < 32 or threads > 1024 or threads % 32:
        return "unsupported-thread-count", 0

    if len(stores) != 1 or stores[0] not in program.blocks[0].operations:
        return "requires-single-top-level-store", 0

    if not value_types[stores[0].operands[1]].shape:
        return "scalar-output", 0

    for info in tensors.values():
        if not info.ndim:
            continue

        attrs = info.attrs or {}

        if (
            attrs.get("jagged_offsets_param") is not None
            or attrs.get("layout_operations") is None
            or set(attrs["layout_operations"])
            - {"tile", "expand", "squeeze", "unsqueeze"}
            or attrs.get("custom_tiling", True)
        ):
            return f"unsupported-layout:{info.name}", 0

    shared = 0
    dot_count = 0

    def uniform(name):
        if name in tensors:
            return tensors[name].ndim == 0

        op = operations.get(name)

        return op is not None and (
            op.opcode in {"arith.constant", "shape.dim"}
            or op.opcode.startswith(("arith.", "cmp."))
            and all(uniform(operand) for operand in op.operands)
        )

    def visit(block):
        nonlocal shared, dot_count

        for op in block.operations:
            if op.opcode == "mem.store":
                if op is not stores[0]:
                    return "nested-store"

                if (
                    value_types[op.operands[0]].shape
                    != value_types[op.operands[1]].shape
                ):
                    return "store-shape-mismatch"

                continue

            allowed = op.opcode in INLINE_OPS | {
                "tensor.cast",
                "select.where",
                "linalg.dot",
                "linalg.matmul",
                "reduce.sum",
                "reduce.max",
                "reduce.min",
                "scf.for",
                "scf.if",
                "scf.yield",
            } or op.opcode.startswith(("arith.", "cmp.", "math."))

            if not allowed:
                return f"unsupported-operation:{op.opcode}"

            if op.opcode == "tensor.view" and any(
                part.strip() not in {"", ":", "None"}
                for part in str(op.attrs.get("subscript", "")).strip("()").split(",")
            ):
                return "unsupported-view"

            if op.opcode == "tensor.extract" and (
                len(op.operands) != 2
                or op.operands[0] not in tensors
                or op.attrs.get("source")
                or len(op.results[0].type.shape) != 2
                or not op.results[0].type.attrs.get("dtype_level", 0)
            ):
                return "unsupported-extract"

            for value in op.results:
                shape = static_shape(value.type)

                if shape is None or len(shape) > 2 or any(dim <= 0 for dim in shape):
                    return f"unsupported-shape:{value.name}"

                if shape and value.type.dtype not in {
                    "float16",
                    "bfloat16",
                    "float32",
                    "int32",
                    "int64",
                    "index",
                    "bool",
                }:
                    return f"unsupported-dtype:{value.name}"

            if op.opcode in {"linalg.dot", "linalg.matmul"}:
                a, b = (value_types[name] for name in op.operands[:2])

                if (
                    a.dtype not in {"float16", "bfloat16"}
                    or a.dtype != b.dtype
                    or op.results[0].type.dtype != "float32"
                ):
                    return "unsupported-dot-dtype"

                a_shape, b_shape = static_shape(a), static_shape(b)

                if (
                    not a_shape
                    or not b_shape
                    or len(a_shape) != 2
                    or len(b_shape) != 2
                    or a_shape[1] != b_shape[0]
                    or static_shape(op.results[0].type) != (a_shape[0], b_shape[1])
                    or any(dim <= 0 or dim % 16 for dim in (*a_shape, *b_shape))
                ):
                    return "unsupported-dot-shape"

                warps = threads // 32

                if int(a.shape[0]) % (16 * warps):
                    return "unsupported-gemm-warp-layout"

                shared += 2 * (prod(static_shape(a)) + prod(static_shape(b)))
                dot_count += 1

            if op.opcode.startswith("reduce."):
                shape = static_shape(value_types[op.operands[0]])
                axis = op.attrs.get("axis")

                if (
                    not shape
                    or axis is None
                    or int(axis) % len(shape) != len(shape) - 1
                ):
                    return "unsupported-reduction-axis"

                if (
                    value_types[op.operands[0]].dtype != "float32"
                    or op.results[0].type.dtype != "float32"
                ):
                    return "unsupported-reduction-dtype"

            if op.opcode == "scf.if" and (
                len(op.regions) != 2 or not uniform(op.operands[0])
            ):
                return "nonuniform-or-unsupported-if"

            if op.opcode in {"scf.if", "scf.for"} and any(
                value.type.dtype in {None, "bool"} for value in op.results
            ):
                return "unsupported-region-result-dtype"

            if op.opcode == "scf.for" and (len(op.regions) != 1 or not op.results):
                return "unsupported-loop"

            if op.opcode == "scf.for" and not all(
                uniform(bound) for bound in op.operands[:3]
            ):
                return "nonuniform-loop-bound"

            for region in op.regions:
                rejected = visit(region)

                if rejected:
                    return rejected
        return None

    reason = visit(program.blocks[0])
    shared += threads * 4

    if reason or not dot_count:
        return reason or "no-dot", shared
    # Sum every staging allocation; no assumed lifetime reuse.
    return ("shared-memory-limit" if shared > shared_limit else None), shared


def plan_tiles(
    *,
    program,
    tensors,
    value_types,
    operations,
    stores,
    schedule,
    options,
    threads,
    outer_grid,
    generic_grid,
):
    """Resolve a candidate to exactly one supported emission route."""
    requested = program.metadata.get("selected_schedule_candidate")
    limit = int(options.get("max_shared_memory_bytes", 48 * 1024))

    if schedule.get("tilelang_materialize_tiles"):
        reason, shared = _materialized_reason(
            program, tensors, value_types, operations, stores, threads, limit
        )

        if reason is None:
            return TilePlan(
                requested, "materialized-tiles", None, threads, outer_grid, shared
            )
        return TilePlan(requested, "generic", reason, threads, generic_grid, shared)

    structural = dot_accumulation_plan(program)

    if structural is not None:
        dot, loop = structural.dot, structural.loop
        a, b = (value_types[name] for name in dot.operands[:2])
        shapes = [static_shape(t) for t in (a, b, dot.results[0].type)]

        def constant(name):
            op = operations.get(name)

            return (
                op.attrs.get("value")
                if op is not None and op.opcode == "arith.constant"
                else None
            )

        dtypes = {
            "float16": 2,
            "bfloat16": 2,
            "float32": 4,
            "float8_e4m3fn": 1,
            "float8_e5m2": 1,
        }
        carry = loop.attrs["iter_args"][0]
        accumulation = [
            value_types[name]
            for name in (
                carry["initial"],
                carry["block_arg"],
                loop.results[0].name,
                dot.results[0].name,
            )
        ]
        valid = (
            all(shape and len(shape) == 2 for shape in shapes)
            and all(dim > 0 and dim % 16 == 0 for shape in shapes for dim in shape)
            and shapes[0][1] == shapes[1][0]
            and shapes[2] == (shapes[0][0], shapes[1][1])
            and all(
                static_shape(value_types[name]) == shapes[2]
                for name in structural.store.operands
            )
            and a.dtype in dtypes
            and a.dtype == b.dtype
            and all(
                t.dtype == "float32" and static_shape(t) == shapes[2]
                for t in accumulation
            )
            and not value_types[loop.operands[1]].shape
            and constant(loop.operands[0]) == 0
            and constant(loop.operands[2]) == 1
            and 32 <= threads <= 1024
            and threads % 32 == 0
        )
        shared = 0

        if valid:
            stages = int(schedule.get("num_stages", 2))
            shared = (
                prod(shapes[0]) * dtypes[a.dtype] + prod(shapes[1]) * dtypes[b.dtype]
            ) * max(stages, 1)
            valid = shared <= limit

        if valid:
            return TilePlan(
                requested,
                "cooperative-gemm",
                None,
                threads,
                outer_grid,
                shared,
                loop.results[0].name,
                dot.results[0].name,
            )
        return TilePlan(
            requested,
            "generic",
            "unsupported-gemm-shape-dtype-or-resources",
            threads,
            generic_grid,
            shared,
        )
    return TilePlan(
        requested, "generic", "no-complete-dot-accumulation", threads, generic_grid
    )
