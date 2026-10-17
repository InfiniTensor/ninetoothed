"""Opt-in singleton dot decomposition, using the bounded block schedule.

Insert this pass immediately before ``ssa.decompose_linalg``. It replaces
static rank-two dots with M=1 or N=1 by FP32 products and reductions. The
A04 block legality/resource checks must accept the entire rewritten graph;
otherwise the original program is returned. The default pipeline is unchanged.
Floating-point association differs from tensor-core dot accumulation.
"""

from dataclasses import replace

from ninetoothed.backends.core import Target
from ninetoothed.backends.triton_reductions import analyze_block_reductions
from ninetoothed.compiler.passes import LANGUAGE_SPECIFIC, Pass
from ninetoothed.dtype import normalize_dtype
from ninetoothed.ir import ssa


class TritonSingletonDot(Pass):
    """Lower bounded singleton dots to ordinary multiply/reduce SSA."""

    name = "ssa.triton.singleton_dot"
    category = LANGUAGE_SPECIFIC
    phase = "optimization"
    supported_backends = (Target.TRITON,)
    default_enabled = False

    def run(self, program, context):
        if context.pass_options.get(self.name):
            raise ValueError("Singleton dot has no pass-specific options.")

        block = program.blocks[0]

        def finish(candidate, analysis):
            schedule = dict(program.metadata.get("schedule", {}))
            schedule["singleton_dot"] = analysis

            if analysis["enabled"]:
                schedule["block_reductions"] = analysis

            return replace(
                candidate, metadata=dict(program.metadata) | {"schedule": schedule}
            )

        if any(op.regions for op in block.operations):
            return finish(program, {"enabled": False, "reason": "unsupported region"})

        values = {value.name: value for value in program.inputs}
        values.update((v.name, v) for op in block.operations for v in op.results)
        names = set(values)
        operations = []
        changed = False

        def emit(opcode, operands, shape, **attrs):
            name = f"%singleton_{len(names)}"

            while name in names:
                name += "_"

            names.add(name)
            value = ssa.Value(
                name=name, type=ssa.Type(kind="tensor", shape=shape, dtype="float32")
            )
            operations.append(
                ssa.Operation(
                    opcode=opcode,
                    operands=tuple(operands),
                    results=(value,),
                    attrs=attrs,
                )
            )

            return value

        for op in block.operations:
            if singleton_dot_shape(op, values) is None:
                operations.append(op)
                continue

            lhs, rhs = (values[name] for name in op.operands)
            m, k = lhs.type.shape
            _, n = rhs.type.shape
            a = emit("tensor.cast", (lhs.name,), (m, k), dtype="float32")
            b = emit("tensor.cast", (rhs.name,), (k, n), dtype="float32")

            if str(m) == "1":
                a = emit("linalg.transpose", (a.name,), (k, m))
                shape, axis, subscript = (k, n), 0, "None, :"
            else:
                b = emit("linalg.transpose", (b.name,), (n, k))
                shape, axis, subscript = (m, k), 1, ":, None"

            product = emit("arith.mul", (a.name, b.name), shape)
            reduced = emit(
                "reduce.sum", (product.name,), (n,) if axis == 0 else (m,), axis=axis
            )
            operations.append(
                ssa.Operation(
                    opcode="tensor.view",
                    operands=(reduced.name,),
                    results=op.results,
                    attrs={"subscript": subscript},
                )
            )
            changed = True

        if not changed:
            return finish(
                program, {"enabled": False, "reason": "no supported singleton dot"}
            )

        candidate = replace(
            program, blocks=(replace(block, operations=tuple(operations)),)
        )
        analysis = analyze_block_reductions(candidate, context)

        return finish(candidate if analysis["enabled"] else program, analysis)


def singleton_dot_shape(op, values):
    """Match the shared supported domain and return the static (M, K, N)."""
    if (
        op.opcode not in {"linalg.dot", "linalg.matmul"}
        or len(op.operands) != 2
        or len(op.results) != 1
        or op.attrs
    ):
        return None

    lhs, rhs = (values[name].type for name in op.operands)
    result = op.results[0].type

    if any(
        type_.kind != "tensor" or len(type_.shape) != 2 for type_ in (lhs, rhs, result)
    ):
        return None

    if any(
        normalize_dtype(type_.dtype) not in {"float16", "bfloat16", "float32"}
        for type_ in (lhs, rhs)
    ):
        return None

    m, k = lhs.shape
    rk, n = rhs.shape

    if normalize_dtype(result.dtype) != "float32" or k != rk or result.shape != (m, n):
        return None

    try:
        shape = tuple(int(str(axis)) for axis in (m, k, n))
    except (TypeError, ValueError):
        return None

    return shape if min(shape) > 0 and (shape[0] == 1 or shape[2] == 1) else None
