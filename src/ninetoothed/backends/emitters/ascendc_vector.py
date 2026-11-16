"""AscendC UB primitives consuming the common native vector plan."""

from dataclasses import dataclass, replace

from ninetoothed.backends.emitters.native_vector import plan_native_vector
from ninetoothed.backends.emitters.vector_ops import (
    BINARY_OPERATORS,
    COMPARISON_OPERATORS,
    classify_pointwise,
)

_TYPES = {"float32": "float", "float16": "half"}
_WIDTHS = {"float32": 4, "float16": 2}
_BINARY = {"arith.add": "Add", "arith.sub": "Sub", "arith.mul": "Mul"}


@dataclass(frozen=True)
class AscendCVectorPlan:
    vectors: object
    block_limit: int = 24
    aivs_per_block: int = 2

    @property
    def blocks(self):
        tasks = self.vectors.tasks
        groups = f"(({tasks}) / {self.aivs_per_block} + (({tasks}) % {self.aivs_per_block} != 0))"

        return f"(({groups}) < {self.block_limit} ? ({groups}) : {self.block_limit})"

    @property
    def task_stride(self):
        return f"{self.aivs_per_block}LL * ::AscendC::GetBlockNum()"


def plan_vector(context, elements):
    # The fixed UB budget and AIV-only grouping are validated for c220.
    arch = context.kernel.compiler_options.get("backend_options", {}).get(
        "arch", "native"
    )

    if arch not in {"native", "dav-c220"}:
        return None

    vector = plan_native_vector(
        context,
        classify_pointwise,
        elements=elements,
        dtype_bytes=_WIDTHS,
        budget=192 * 1024 - 256,
        alignment=32,
    )

    if vector is None:
        return None

    output = context.outputs[0]
    # Adjacent parallel packets must not share a partial GM cache line.
    guard = f"({vector.guard}) && (((uintptr_t){output} % 32ULL) == 0)"

    return AscendCVectorPlan(replace(vector, guard=guard))


def render_support():
    return """template<::AscendC::HardEvent event>
__aicore__ inline void nt_vec_fence(::AscendC::TPipe &pipe) {
    auto id = pipe.FetchEventID(event);
    ::AscendC::SetFlag<event>(id);
    ::AscendC::WaitFlag<event>(id);
    pipe.ReleaseEventID<event>(id);
}
__aicore__ inline void nt_vec_barrier() {
    ::AscendC::PipeBarrier<PIPE_V>();
}"""


def render_vector(plan, target):
    """Emit only the frozen DAG, storage, ownership and stream-independent code."""
    vector = plan.vectors
    dag, chunk = vector.dag, vector.storage.elements
    lines = ["::AscendC::TPipe nt_vec_pipe;"]

    for i, (dtype, size) in enumerate(
        zip(dag.buffer_dtypes, vector.storage.allocation_bytes)
    ):
        lines.extend(
            (
                f"::AscendC::TBuf<::AscendC::TPosition::VECCALC> nt_vec_storage_{i};",
                f"nt_vec_pipe.InitBuffer(nt_vec_storage_{i}, {size});",
                f"auto nt_vec_b{i} = nt_vec_storage_{i}.Get<{_TYPES[dtype]}>();",
            )
        )

    lines.append(
        f"for (int64_t nt_vec_task = ::AscendC::GetBlockIdx(); nt_vec_task < ({vector.tasks}); nt_vec_task += {plan.task_stride}) {{"
    )
    body = [
        f"const int64_t nt_vec_start = nt_vec_task * {chunk};",
        f"::uint32_t nt_vec_count = (::uint32_t)((({vector.extent}) - nt_vec_start) < {chunk} ? (({vector.extent}) - nt_vec_start) : {chunk});",
        "::uint32_t nt_vec_aligned = (nt_vec_count + 127) / 128 * 128;",
    ]

    def fence(event):
        body.append(f"::nt_vec_fence<::AscendC::HardEvent::{event}>(nt_vec_pipe);")

    for value, buffer in dag.inputs:
        body.append(
            f"::AscendC::Duplicate(nt_vec_b{buffer}, ({_TYPES[value.dtype]})0, nt_vec_aligned);"
        )

    fence("V_MTE2")

    for value, buffer in dag.inputs:
        dtype = _TYPES[value.dtype]
        body.extend(
            (
                f"::AscendC::GlobalTensor<{dtype}> nt_vec_g{buffer};",
                f"nt_vec_g{buffer}.SetGlobalBuffer({value.name} + nt_vec_start);",
                f"::AscendC::DataCopyPad(nt_vec_b{buffer}, nt_vec_g{buffer}, ::AscendC::DataCopyExtParams{{1, (::uint32_t)(nt_vec_count * sizeof({dtype})), 0, 0, 0}}, ::AscendC::DataCopyPadExtParams<{dtype}>(false, 0, 0, ({dtype})0));",
            )
        )

    fence("MTE2_V")

    def operand(value, buffer, index=None):
        if not value.scalar:
            return f"nt_vec_b{buffer}" + (f".GetValue({index})" if index else "")

        literal = (
            target.literal(value.literal) if value.literal is not None else value.name
        )

        return target.cast(value.dtype if value.dtype in _TYPES else "float32", literal)

    for step in dag.steps:
        code = step.operation.opcode
        destination = f"nt_vec_b{step.destination}"
        args = [operand(v, b) for v, b in zip(step.operands, step.sources)]
        scalar = step.mode == "scalar" or (
            step.mode == "cast" and step.operands[0].dtype == step.result.dtype
        )

        if step.mode == "binary":
            body.append(
                f"::AscendC::{_BINARY[code]}({destination}, {args[0]}, {args[1]}, nt_vec_aligned);"
            )
        elif step.mode == "scalar_binary":
            if step.operands[0].scalar:
                args.reverse()

            instruction = "Adds" if code == "arith.add" else "Muls"
            body.append(
                f"::AscendC::{instruction}({destination}, {args[0]}, {args[1]}, nt_vec_aligned);"
            )
        elif step.mode == "cast" and not scalar:
            rounding = "CAST_RINT" if step.result.dtype == "float16" else "CAST_NONE"
            body.append(
                f"::AscendC::Cast({destination}, {args[0]}, ::AscendC::RoundMode::{rounding}, nt_vec_aligned);"
            )

        if scalar:
            fence("V_S")
            args = [
                operand(v, b, "nt_vec_i") for v, b in zip(step.operands, step.sources)
            ]

            if code in COMPARISON_OPERATORS:
                expression = f"({args[0]} {COMPARISON_OPERATORS[code]} {args[1]})"
            elif code == "select.where":
                expression = f"({args[0]} ? {args[1]} : {args[2]})"
            elif code == "tensor.cast":
                expression = args[0]
            else:
                args = [f"(float)({arg})" for arg in args]
                expression = f"({args[0]} {BINARY_OPERATORS[code]} {args[1]})"

            dtype = "float32" if step.result.dtype == "bool" else step.result.dtype
            expression = target.cast(dtype, expression)
            body.append(
                f"for (::uint32_t nt_vec_i = 0; nt_vec_i < nt_vec_aligned; ++nt_vec_i) {{ {destination}.SetValue(nt_vec_i, nt_vec_i < nt_vec_count ? {expression} : ({_TYPES[dtype]})0); }}"
            )
            fence("S_V")
        else:
            body.append("::nt_vec_barrier();")

    fence("V_MTE3")

    for output, buffer, value in dag.stores:
        dtype = _TYPES[value.dtype]
        body.extend(
            (
                f"::AscendC::GlobalTensor<{dtype}> nt_vec_output;",
                f"nt_vec_output.SetGlobalBuffer({output} + nt_vec_start);",
                f"::AscendC::DataCopyPad(nt_vec_output, nt_vec_b{buffer}, ::AscendC::DataCopyExtParams{{1, (::uint32_t)(nt_vec_count * sizeof({dtype})), 0, 0, 0}});",
            )
        )

    fence("MTE3_V")
    lines.extend("    " + line for line in body)
    lines.append("}")

    return "\n".join(lines)
