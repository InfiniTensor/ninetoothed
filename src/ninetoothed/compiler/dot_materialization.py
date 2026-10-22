"""Materialize each explicit dot group before its original ordered carry add."""

from dataclasses import dataclass, replace

from ninetoothed.backends.emitters.dot_layout import prove_dense_dot_access
from ninetoothed.compiler.effects import pure_expression
from ninetoothed.compiler.layout_runtime import memory_spans_overlap, tensor_memory_span
from ninetoothed.compiler.materialization import MaterializationStage, ScratchTensor
from ninetoothed.dtype import normalize_dtype
from ninetoothed.ir import TensorSpec, ssa


@dataclass(frozen=True)
class OrderedDotPlan:
    stages: tuple[MaterializationStage, ...]
    scratch: tuple[ScratchTensor, ...]
    scratch_specs: tuple[TensorSpec, ...]
    dimensions: tuple[int, int, int]
    groups: int
    source_names: tuple[str, str, str]

    def metadata(self):
        return {
            "role": "explicit-dot-groups-and-ordered-add",
            "group_width": 32,
            "groups": self.groups,
            "dimensions": dict(zip(("rows", "columns", "reduction"), self.dimensions)),
            "scratch_bytes": self.scratch[0].nbytes,
            "scratch_budget_bytes": 32 * 1024 * 1024,
            "partial_dtype": "float32",
            "carry_dtype": "float32",
            "reassociation": False,
            "lifetime": "per-call-same-stream",
            "stages": tuple(
                {
                    "kind": stage.program.kind,
                    "role": stage.program.metadata["ordered_dot_stage"],
                }
                for stage in self.stages
            ),
        }

    def runtime_eligible(self, arguments):
        """Require dense sources whose reads cannot alias the original output."""
        m, n, k = self.dimensions

        for name, shape in zip(self.source_names, ((m, k), (k, n), (m, n))):
            value = arguments[name]

            if (
                tuple(value.shape) != shape
                or tuple(value.stride()) != (shape[1], 1)
                or value.data_ptr() % 32
            ):
                return False

        if k > 65535 or n > 65535:
            return False

        # Computing all partials first moves reads before the original stores.
        # Unknown spans fail closed; read/read aliases do not change that order.
        output_span = tensor_memory_span(arguments[self.source_names[-1]])

        return not any(
            memory_spans_overlap(tensor_memory_span(arguments[name]), output_span)
            for name in self.source_names[:2]
        )


def _literal(name, definitions, types):
    operation = definitions.get(name)

    if operation is None:
        return None

    if operation.opcode == "arith.constant":
        value = operation.attrs.get("value")

        return value if isinstance(value, int) and not isinstance(value, bool) else None

    if operation.opcode == "shape.dim" and not operation.attrs.get("source"):
        try:
            return int(types[operation.operands[0]].shape[int(operation.attrs["dim"])])
        except (KeyError, ValueError, IndexError, TypeError):
            return None
    return None


def _scratch_spec(name, groups, m, n):
    strides = (f"{name}_stride_0", f"{name}_stride_1")
    offsets = (
        f"extract_0_0 * {m} + outer_index // {n // 32} * 32 + value_0",
        f"outer_index % {n // 32} * 32 + value_1",
    )
    attrs = {
        "ndim": 2,
        "source_ndim": 2,
        "source_name": name,
        "source_shape": (str(groups * m), str(n)),
        "source_dtype": "float32",
        "source_strides": strides,
        "dtype_level": 0,
        "other": 0,
        "view_shape": (str(m // 32), str(n // 32)),
        "view_ndim": 2,
        "application_shape": (str(groups),),
        "application_ndim": 1,
        "dtype_shapes": ((str(groups),), ("32", "32")),
        "dtype_target_dims": (("0",), ("0", "1")),
        "target_dims": ("0", "1"),
        "access_templates": (
            {
                "level": 1,
                "shape": ("32", "32"),
                "offsets": offsets,
                "linear_offset": f"({offsets[0]}) * {strides[0]} + ({offsets[1]}) * {strides[1]}",
                "mask": f"(extract_0_0 >= 0) & (extract_0_0 < {groups}) & (outer_index >= 0) & (outer_index < {(m // 32) * (n // 32)}) & (value_0 >= 0) & (value_0 < 32) & (value_1 >= 0) & (value_1 < 32)",
            },
        ),
    }

    return TensorSpec(
        name=name, ndim=2, shape=attrs["view_shape"], dtype="float32", attrs=attrs
    )


def plan_ordered_dot_materialization(
    program, tensors, *, max_scratch_bytes=32 * 1024 * 1024
):
    """Prove the full original K32 dot/add region and its terminal pure epilogue.

    Partial matrices are stacked in an ordinary two-dimensional allocation.
    Each original dot result is stored as FP32 once; the second stage retains
    the original carry initializer, add operand order, iteration order and casts.
    The independent budget does not change dependent-dot materialization limits.
    """
    from ninetoothed.compiler.canonicalize import canonicalize_dtype_expressions

    program = canonicalize_dtype_expressions(program)

    if len(program.blocks) != 1 or len(program.outputs) != 1:
        return None

    top = program.blocks[0].operations
    loops = [op for op in top if op.opcode == "scf.for"]
    stores = [op for op in top if op.opcode == "mem.store"]

    if len(loops) != 1 or len(stores) != 1 or top[-1] is not stores[0]:
        return None

    loop, store = loops[0], stores[0]
    specs = {spec.name: spec for spec in tensors}
    inputs = {value.name: value for value in program.inputs}
    definitions = {value.name: op for op in top for value in op.results}
    types = {
        **{name: value.type for name, value in inputs.items()},
        **{name: op.results[0].type for name, op in definitions.items()},
    }

    if (
        len(loop.results) != 1
        or len(loop.regions) != 1
        or len(loop.operands) != 4
        or len(loop.attrs.get("iter_args", ())) != 1
        or len(store.operands) != 2
        or set(store.attrs) - {"target", "target_shape"}
        or program.outputs[0].name != store.operands[1]
        or tuple(map(str, store.attrs.get("target_shape", ()))) != ("32", "32")
    ):
        return None

    body = loop.regions[0]

    if (
        len(body.args) != 2
        or tuple(op.opcode for op in body.operations)
        not in {
            (
                "tensor.extract",
                "tensor.extract",
                "linalg.dot",
                "arith.add",
                "scf.yield",
            ),
            (
                "tensor.extract",
                "tensor.extract",
                "linalg.matmul",
                "arith.add",
                "scf.yield",
            ),
        }
        or any(op.regions for op in body.operations)
        or any(len(op.results) != 1 for op in body.operations[:-1])
    ):
        return None

    left, right, dot, addition, yielded = body.operations
    carry = loop.attrs["iter_args"][0]
    initializer = definitions.get(carry["initial"])

    if (
        dot.operands != (left.results[0].name, right.results[0].name)
        or addition.operands
        not in {
            (carry["block_arg"], dot.results[0].name),
            (dot.results[0].name, carry["block_arg"]),
        }
        or yielded.operands != (addition.results[0].name,)
        or body.args[1].name != carry["block_arg"]
        or body.args[0].name != loop.attrs.get("induction")
        or loop.operands[-1] != carry["initial"]
        or initializer is None
        or initializer.opcode not in {"tensor.zeros", "tensor.full"}
        or len(initializer.results) != 1
        or initializer.operands
        or initializer.attrs.get("value", 0) != 0
    ):
        return None

    carry_types = (
        initializer.results[0].type,
        body.args[1].type,
        dot.results[0].type,
        addition.results[0].type,
        loop.results[0].type,
    )

    if any(
        type_.kind != "tensor"
        or tuple(map(str, type_.shape)) != ("32", "32")
        or normalize_dtype(type_.dtype) != "float32"
        for type_ in carry_types
    ):
        return None

    lower, groups, step = (
        _literal(name, definitions, types) for name in loop.operands[:3]
    )

    if lower != 0 or step != 1 or groups is None or not 0 < groups <= 2047:
        return None

    output = specs.get(store.operands[1])

    if output is None or normalize_dtype(output.dtype) not in {"float16", "float32"}:
        return None

    try:
        m, n = map(int, output.attrs["source_shape"])
    except (KeyError, TypeError, ValueError):
        return None

    if (
        m <= 0
        or n <= 0
        or m % 32
        or n % 32
        or tuple(map(str, output.shape)) != (str(m // 32), str(n // 32))
    ):
        return None

    k = groups * 32
    scratch_bytes = groups * m * n * 4

    if scratch_bytes > min(max_scratch_bytes, 32 * 1024 * 1024):
        return None

    row = f"outer_index // {n // 32} * 32 + value_0"
    column = f"outer_index % {n // 32} * 32 + value_1"
    tasks = (m // 32) * (n // 32)
    names = []

    for extract, expected, physical in (
        (left, (row, "extract_0_0 * 32 + value_1"), (m, k)),
        (right, ("extract_0_0 * 32 + value_0", column), (k, n)),
    ):
        if (
            len(extract.operands) != 2
            or extract.operands[1] != body.args[0].name
            or extract.operands[0] not in inputs
            or normalize_dtype(extract.results[0].type.dtype) != "float16"
        ):
            return None

        info = specs.get(extract.operands[0])

        if info is None or normalize_dtype(info.dtype) != "float16":
            return None

        proof = prove_dense_dot_access(
            extract.results[0].type.attrs,
            shape=(32, 32),
            tasks=tasks,
            stages=groups,
            expected_offsets=expected,
        )

        if proof is None or proof.source_shape != physical:
            return None

        names.append(extract.operands[0])

    proof = prove_dense_dot_access(
        output.attrs,
        shape=(32, 32),
        tasks=tasks,
        stages=groups,
        expected_offsets=(row, column),
    )

    if (
        proof is None
        or proof.source_shape != (m, n)
        or names[0] == names[1]
        or set(inputs) != {*names, store.operands[1]}
    ):
        return None
    # The top block is replayed only where required. Freeze shape queries rather
    # than accidentally asking a scratch view for an original source shape.

    frozen = []

    for operation in top:
        if operation in (loop, store) or operation is initializer:
            frozen.append(operation)
            continue

        if operation.regions or not pure_expression(operation):
            return None

        if operation.opcode == "shape.dim":
            value = _literal(operation.results[0].name, definitions, types)

            if value is None:
                return None

            operation = replace(
                operation, opcode="arith.constant", operands=(), attrs={"value": value}
            )
        elif operation.opcode not in {
            "arith.constant",
            "tensor.cast",
        } and not operation.opcode.startswith(("arith.", "math.", "cmp.")):
            return None

        if any(operand in names for operand in operation.operands):
            return None

        if (
            operation.results
            and operation.results[0].type.shape
            and tuple(map(str, operation.results[0].type.shape)) != ("32", "32")
        ):
            return None

        frozen.append(operation)

    occupied = (
        set(inputs)
        | {
            str(stride)
            for spec in tensors
            for stride in spec.attrs.get("source_strides", ())
        }
        | set(program.metadata.get("symbols", ()))
    )
    name = "nt_dot_partials"

    while name in occupied or any(
        f"{name}_stride_{dim}" in occupied for dim in range(2)
    ):
        name += "_"

    scratch = ScratchTensor(
        value=dot.results[0].name,
        name=name,
        shape=(str(groups * m), str(n)),
        dtype="float32",
    )
    scratch_spec = _scratch_spec(name, groups, m, n)
    scratch_value = ssa.Value(
        name=name,
        type=ssa.Type(
            kind="tensor",
            shape=(str(groups),),
            dtype="float32",
            attrs=scratch_spec.attrs,
        ),
    )
    partial_store = ssa.Operation(
        opcode="mem.store",
        operands=(dot.results[0].name, name),
        attrs={
            "target": name,
            "indices": (body.args[0].name,),
            "target_shape": ("32", "32"),
            "target_dtype_level": 1,
            "base_dtype_level": 0,
        },
    )
    partial_loop = replace(
        loop,
        operands=loop.operands[:3],
        results=(),
        attrs={"induction": body.args[0].name, "iter_args": ()},
        regions=(
            replace(
                body,
                args=body.args[:1],
                operations=(
                    left,
                    right,
                    dot,
                    partial_store,
                    ssa.Operation(opcode="scf.yield"),
                ),
            ),
        ),
    )
    bounds = {name for name in loop.operands[:3]}
    partial_top = tuple(
        op for op in frozen if any(value.name in bounds for value in op.results)
    ) + (partial_loop,)
    partial_program = replace(
        program,
        kind=program.kind + "_dot_partials",
        inputs=tuple(inputs[name] for name in names) + (scratch_value,),
        outputs=(scratch_value,),
        blocks=(ssa.Block(operations=partial_top),),
        metadata={
            "ordered_dot_stage": "partials",
            "source_kind": program.kind,
            "groups": groups,
            "dimensions": (m, n, k),
            "reassociation": False,
        },
    )
    partial_type = ssa.Type(
        kind="tensor",
        shape=("32", "32"),
        dtype="float32",
        attrs={**scratch_spec.attrs, "dtype_level": 1},
    )
    read_partial = ssa.Operation(
        opcode="tensor.extract",
        operands=(name, body.args[0].name),
        results=(replace(dot.results[0], type=partial_type),),
        attrs={"subscript": "i", "source": False},
    )
    ordered_loop = replace(
        loop, regions=(replace(body, operations=(read_partial, addition, yielded)),)
    )
    ordered_top = tuple(ordered_loop if op is loop else op for op in frozen)
    ordered_program = replace(
        program,
        kind=program.kind + "_ordered_add",
        inputs=(scratch_value, inputs[store.operands[1]]),
        blocks=(ssa.Block(operations=ordered_top),),
        metadata={
            "ordered_dot_stage": "ordered-add",
            "source_kind": program.kind,
            "groups": groups,
            "dimensions": (m, n, k),
            "reassociation": False,
        },
    )
    ssa.verify_program(partial_program)
    ssa.verify_program(ordered_program)

    return OrderedDotPlan(
        stages=(
            MaterializationStage(
                program=partial_program, scratch_inputs=(), scratch_output=scratch
            ),
            MaterializationStage(
                program=ordered_program, scratch_inputs=(scratch,), scratch_output=None
            ),
        ),
        scratch=(scratch,),
        scratch_specs=(scratch_spec,),
        dimensions=(m, n, k),
        groups=groups,
        source_names=(*names, store.operands[1]),
    )


__all__ = ["OrderedDotPlan", "plan_ordered_dot_materialization"]
