"""Opt-in singleton dot decomposition, using the bounded block schedule.

Insert this pass immediately before ``ssa.decompose_linalg``. It replaces
static rank-two dots with M=1 or N=1 by FP32 products and reductions. The
A04 block legality/resource checks must accept the entire rewritten graph;
otherwise the original program is returned. The default pipeline is unchanged.
Floating-point association differs from tensor-core dot accumulation.
"""

from dataclasses import replace

from ninetoothed.backends.core import Target
from ninetoothed.backends.triton_reductions import TritonBlockReductions
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

        if any(op.regions for op in block.operations):
            return program

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
            if not _supported_dot(op, values):
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
            return program

        candidate = replace(
            program, blocks=(replace(block, operations=tuple(operations)),)
        )
        scheduled = TritonBlockReductions().run(candidate, context)

        if not scheduled.metadata["schedule"]["block_reductions"]["enabled"]:
            return program
        return scheduled


def _supported_dot(op, values):
    if (
        op.opcode not in {"linalg.dot", "linalg.matmul"}
        or len(op.operands) != 2
        or len(op.results) != 1
        or op.attrs
    ):
        return False

    lhs, rhs = (values[name].type for name in op.operands)
    result = op.results[0].type

    if any(
        type_.kind != "tensor" or len(type_.shape) != 2 for type_ in (lhs, rhs, result)
    ):
        return False

    if any(
        normalize_dtype(type_.dtype) not in {"float16", "bfloat16", "float32"}
        for type_ in (lhs, rhs)
    ):
        return False

    m, k = lhs.shape
    rk, n = rhs.shape

    return (
        normalize_dtype(result.dtype) == "float32"
        and k == rk
        and result.shape == (m, n)
        and (str(m) == "1" or str(n) == "1")
    )
