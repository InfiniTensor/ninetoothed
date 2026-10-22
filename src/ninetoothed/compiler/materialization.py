"""Partition pure SSA graphs at reused expensive tensor producers."""

import math
from collections import Counter
from dataclasses import dataclass, replace
from functools import cache

from ninetoothed.compiler.effects import pure_expression
from ninetoothed.dtype import normalize_dtype
from ninetoothed.ir import TensorSpec, ssa


@dataclass(frozen=True)
class ScratchTensor:
    value: str
    name: str
    shape: tuple[str, str]
    dtype: str

    @property
    def nbytes(self):
        return (
            math.prod(map(int, self.shape)) * {"float16": 2, "float32": 4}[self.dtype]
        )


@dataclass(frozen=True)
class MaterializationStage:
    program: ssa.Program
    scratch_inputs: tuple[ScratchTensor, ...]
    scratch_output: ScratchTensor | None


@dataclass(frozen=True)
class MaterializationPlan:
    stages: tuple[MaterializationStage, ...]
    scratch: tuple[ScratchTensor, ...]
    input_snapshot: tuple = ()
    bound_snapshot: tuple = ()

    def bind(self, program):
        return replace(
            self, bound_snapshot=(program.inputs, program.outputs, program.blocks)
        )

    def metadata(self):
        return {
            "cuts": tuple(tensor.value for tensor in self.scratch),
            "scratch_bytes": sum(tensor.nbytes for tensor in self.scratch),
            "stages": tuple(
                {
                    "kind": stage.program.kind,
                    "inputs": tuple(value.name for value in stage.program.inputs),
                    "output": stage.program.outputs[0].name,
                    "operations": tuple(
                        op.opcode for op in stage.program.blocks[0].operations
                    ),
                }
                for stage in self.stages
            ),
            "lifetime": "per-call-same-stream",
            "reassociation": False,
        }


def retain_bound_plan(program):
    """Reject stale logical stages before specialization or emission can use them."""
    plan = program.plans.get("materialization")

    if plan is None or plan.bound_snapshot == (
        program.inputs,
        program.outputs,
        program.blocks,
    ):
        return program

    diagnostic = program.metadata.get("materialization", {})

    if diagnostic.get("explicitly_enabled"):
        raise ValueError(
            "Explicit materialization plan no longer matches its SSA input."
        )
    return replace(
        program,
        plans={},
        metadata={
            **program.metadata,
            "materialization": {
                "enabled": False,
                "selected": False,
                "reason": "stale SSA stage plan",
            },
        },
    )


def dense_scratch_spec(tensor):
    """Represent an internal buffer through the existing tensor/access ABI."""
    rows, columns = tensor.shape
    strides = (f"{tensor.name}_stride_0", f"{tensor.name}_stride_1")
    offsets = (
        f"outer_index * {rows} + value_0",
        f"outer_index % 1 * {columns} + value_1",
    )
    attrs = {
        "ndim": 2,
        "source_ndim": 2,
        "source_name": tensor.name,
        "source_shape": tensor.shape,
        "source_dtype": tensor.dtype,
        "source_strides": strides,
        "dtype_level": 0,
        "view_shape": ("1", "1"),
        "view_ndim": 2,
        "application_shape": tensor.shape,
        "application_ndim": 2,
        "dtype_shapes": (tensor.shape,),
        "dtype_target_dims": (("0", "1"),),
        "target_dims": ("0", "1"),
        "other": 0,
        "access_templates": (
            {
                "level": 0,
                "shape": tensor.shape,
                "offsets": offsets,
                "linear_offset": f"({offsets[0]}) * {strides[0]} + ({offsets[1]}) * {strides[1]}",
                "mask": f"(outer_index >= 0) & (outer_index < 1) & (value_0 >= 0) & (value_0 < {rows}) & (value_1 >= 0) & (value_1 < {columns})",
            },
        ),
    }

    return TensorSpec(
        name=tensor.name, ndim=2, shape=("1", "1"), dtype=tensor.dtype, attrs=attrs
    )


def _static_matrix(type_):
    if type_.kind != "tensor" or len(type_.shape) != 2:
        return False

    try:
        return all(int(dim) > 0 for dim in type_.shape)
    except (TypeError, ValueError):
        return False


def plan_materialization(program, *, max_scratch_bytes=8 * 1024 * 1024):
    """Select static full-domain, effect-free graphs containing dependent dots.

    Cuts depend on def-use and expensive producer ancestry. The pass preserves
    every operation's arithmetic order, dtype, source access attributes and
    original store. It introduces only explicit storage boundaries.
    """
    from ninetoothed.compiler.canonicalize import canonicalize_dtype_expressions

    program = canonicalize_dtype_expressions(program)

    if len(program.blocks) != 1:
        return None

    ops = program.blocks[0].operations
    stores = [op for op in ops if op.opcode == "mem.store"]

    if len(stores) != 1 or ops[-1] is not stores[0] or len(stores[0].operands) != 2:
        return None

    store = stores[0]

    if set(store.attrs) - {"target", "target_shape"}:
        return None

    if any(
        op.regions
        or (
            op is not store
            and not pure_expression(op)
            and op.opcode not in {"linalg.dot", "linalg.matmul", "linalg.transpose"}
        )
        for op in ops
    ):
        return None

    if any(len(op.results) != 1 for op in ops if op is not store):
        return None

    inputs = {value.name: value for value in program.inputs}
    defs = {op.results[0].name: op for op in ops if op.results}
    types = {
        **{name: value.type for name, value in inputs.items()},
        **{name: op.results[0].type for name, op in defs.items()},
    }

    if any(
        not _static_matrix(value.type)
        or normalize_dtype(value.type.dtype) not in {"float16", "float32"}
        or value.type.attrs.get("jagged_dim") is not None
        for value in program.inputs
    ):
        return None

    output = inputs.get(store.operands[1])

    if output is None or tuple(map(str, output.type.shape)) != tuple(
        map(str, store.attrs.get("target_shape", ()))
    ):
        return None

    physical = tuple(map(str, output.type.attrs.get("source_shape", output.type.shape)))

    if (
        physical != tuple(map(str, output.type.shape))
        or tuple(map(str, types[store.operands[0]].shape)) != physical
    ):
        return None

    dots = [op for op in ops if op.opcode in {"linalg.dot", "linalg.matmul"}]

    if len(dots) < 2 or any(not _static_matrix(op.results[0].type) for op in dots):
        return None

    uses = Counter(name for op in ops for name in op.operands)

    @cache
    def ancestors(name):
        op = defs.get(name)

        return (
            frozenset()
            if op is None
            else frozenset((name,)).union(
                *(ancestors(operand) for operand in op.operands)
            )
        )

    dot_values = {op.results[0].name for op in dots}
    reduction_values = {
        name for name, op in defs.items() if op.opcode.startswith("reduce.")
    }
    wanted = set()

    for dot in dots:
        for operand in dot.operands:
            dependency = ancestors(operand)

            if dependency & dot_values:
                wanted.update(
                    name for name in dependency & dot_values if uses[name] > 1
                )

                if dependency & reduction_values and _static_matrix(types[operand]):
                    wanted.add(operand)

    if not wanted:
        return None

    ordered = [
        op.results[0].name for op in ops if op.results and op.results[0].name in wanted
    ]
    used_names = (
        set(inputs)
        | set(defs)
        | {
            str(stride)
            for value in program.inputs
            for stride in value.type.attrs.get("source_strides", ())
        }
        | set(program.metadata.get("symbols", ()))
    )
    scratch = []

    for index, value in enumerate(ordered):
        type_ = types[value]
        dtype = normalize_dtype(type_.dtype)

        if not _static_matrix(type_) or dtype not in {"float16", "float32"}:
            return None

        name = f"nt_materialized_{index}"

        if name in used_names or any(
            f"{name}_stride_{axis}" in used_names for axis in range(2)
        ):
            return None

        scratch.append(ScratchTensor(value, name, tuple(map(str, type_.shape)), dtype))

    if sum(tensor.nbytes for tensor in scratch) > max_scratch_bytes:
        return None

    available = {}
    stages = []

    def stage(result_name, scratch_output):
        needed = set()

        def visit(name):
            if name in needed:
                return

            needed.add(name)

            if name in available:
                return

            op = defs.get(name)

            if op is not None:
                for operand in op.operands:
                    visit(operand)

                for key in ("dtype_ref", "dtype_source", "dtype_operand"):
                    reference = op.attrs.get(key)

                    if reference in inputs or reference in defs:
                        visit(reference)

        visit(result_name)
        mapping = {
            name: tensor.name for name, tensor in available.items() if name in needed
        }

        def rewrite(value):
            if isinstance(value, str):
                return mapping.get(value, value)

            if hasattr(value, "items"):
                return {key: rewrite(item) for key, item in value.items()}

            if isinstance(value, (tuple, list)):
                return tuple(rewrite(item) for item in value)
            return value

        selected = [
            replace(
                op,
                operands=tuple(rewrite(name) for name in op.operands),
                attrs=rewrite(op.attrs),
            )
            for op in ops
            if op.results
            and op.results[0].name in needed
            and op.results[0].name not in available
        ]
        stage_inputs = [value for name, value in inputs.items() if name in needed]
        scratch_inputs = tuple(
            tensor for name, tensor in available.items() if name in needed
        )

        for tensor in scratch_inputs:
            spec = dense_scratch_spec(tensor)
            stage_inputs.append(
                ssa.Value(
                    name=tensor.name,
                    type=ssa.Type(
                        kind="tensor",
                        shape=tensor.shape,
                        dtype=spec.dtype,
                        attrs=spec.attrs,
                    ),
                )
            )

        if scratch_output is not None:
            spec = dense_scratch_spec(scratch_output)
            out_value = ssa.Value(
                name=spec.name,
                type=ssa.Type(
                    kind="tensor",
                    shape=scratch_output.shape,
                    dtype=spec.dtype,
                    attrs=spec.attrs,
                ),
            )
            stage_inputs.append(out_value)
            terminal = ssa.Operation(
                opcode="mem.store",
                operands=(rewrite(result_name), spec.name),
                attrs={"target": spec.name, "target_shape": scratch_output.shape},
            )
        else:
            out_value = output

            if output.name not in {value.name for value in stage_inputs}:
                stage_inputs.append(output)

            terminal = replace(store, operands=(rewrite(result_name), output.name))

        selected.append(terminal)
        result = ssa.Program(
            kind=f"{program.kind}_materialization_{len(stages)}",
            inputs=tuple(stage_inputs),
            outputs=(out_value,),
            blocks=(ssa.Block(operations=tuple(selected)),),
            metadata={
                "source": "ssa-materialization",
                "parent": program.kind,
                "stage": len(stages),
            },
        )
        ssa.verify_program(result)

        return MaterializationStage(result, scratch_inputs, scratch_output)

    for tensor in scratch:
        stages.append(stage(tensor.value, tensor))
        available[tensor.value] = tensor

    stages.append(stage(store.operands[0], None))

    snapshot = (program.inputs, program.outputs, program.blocks)

    return MaterializationPlan(tuple(stages), tuple(scratch), snapshot, snapshot)


__all__ = [
    "MaterializationPlan",
    "MaterializationStage",
    "ScratchTensor",
    "dense_scratch_spec",
    "plan_materialization",
]
