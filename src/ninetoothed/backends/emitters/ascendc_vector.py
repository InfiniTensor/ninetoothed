"""Vector elementwise lowering for the AscendC backend.

Shared typed DAGs over float32 1-D elementwise kernels map onto
TPipe/LocalTensor vector intrinsics with aligned block copies; a scalar
tail loop covers the non-8-multiple remainder (raw DataCopy counts must
stay aligned).  Unary functions without a direct intrinsic compose from
the primitive vector set (e.g. sigmoid = Reciprocal(Add(Exp(Neg(x)), 1))).
"""

import math
from dataclasses import replace

from ninetoothed.backends.ascendc_resources import ASCENDC_UB_BYTES
from ninetoothed.backends.emitters.ascendc_primitives import (
    numeric_lower_bound,
    render_division,
    render_extrema,
    render_refinement,
    render_unary_zero_repair,
)
from ninetoothed.backends.emitters.vector_dag import VectorLowering, plan_vector_dag
from ninetoothed.compiler.passes import program_facts
from ninetoothed.ir import ssa

_VEC_BINARY = {
    "arith.add": "Add",
    "arith.sub": "Sub",
    "arith.mul": "Mul",
    "arith.div": "Div",
    "arith.maximum": "MaxStrict",
    "arith.minimum": "MinStrict",
}

_VEC_COMMUTATIVE = {"arith.add", "arith.mul", "arith.maximum", "arith.minimum"}

_VEC_SCALAR_BINARY = {
    "arith.add": "Adds",
    "arith.mul": "Muls",
    "arith.maximum": "MaxStrict",
    "arith.minimum": "MinStrict",
}

_VEC_UNARY = {
    "math.exp": "Exp",
    "math.log": "Ln",
    "math.abs": "Abs",
    "math.sqrt": "Sqrt",
    "math.rsqrt": "RsqrtStrict",
    "math.reciprocal": "ReciprocalStrict",
    "math.relu": "Relu",
    "math.floor": "Floor",
    "math.ceil": "Ceil",
    # CANN vector trig approximations lose full-range float32 argument
    # reduction (e.g. Sin(1e8)).  Keep these on the scalar compensated
    # range-reduction path until a bounded-precision policy is explicit.
    "math.tanh": "Tanh",
    "math.erf": "Erf",
    "math.atan": "Atan",
}

_VECTOR_ELEM_CHUNK = 8192
_MAX_STAGES = 8


def match_vector_elementwise(context, normalize_dtype):
    """Return a compiled vector plan for a float32 1-D elementwise store."""
    from ninetoothed.backends.emitters.staging import canonical_flat_view

    if any(
        info.ndim != 0 and not canonical_flat_view(info)
        for name in (*context.variables, *context.outputs)
        if (info := context.tensors.get(name)) is not None
    ):
        return None

    stores = [op for op in context.stores if len(op.operands) == 2]

    if len(stores) != 1 or len(context.outputs) != 1:
        return None

    output = context.outputs[0]
    store = stores[0]

    if store.operands[1] != output:
        return None

    output_info = context.tensors.get(output)

    if output_info is None or output_info.ndim != 1:
        return None

    output_dtype = normalize_dtype(output_info.dtype)

    if output_dtype not in {"float32", "float16"}:
        return None

    value = store.operands[0]
    producer = context.operations.get(value)

    if output_dtype == "float16":
        if (
            producer is None
            or producer.opcode != "tensor.cast"
            or len(producer.operands) != 1
        ):
            return None

        value = producer.operands[0]

    dag_context = _packet_context(context, value, output)
    dag = plan_vector_dag(
        dag_context,
        tuple(name for name in context.variables if context.tensors[name].ndim == 1),
        (output,),
        normalize_dtype=normalize_dtype,
        classify=_packet_primitive,
        dtype_bytes={"float32": 4},
    )

    if (
        dag is None
        or len(dag.steps) > _MAX_STAGES
        or any(store.value.scalar for store in dag.stores)
    ):
        return None

    plan = AscendPacketStorage(output, dag)

    if plan is not None:
        from ninetoothed.dtype import normalize_dtype as explicit_dtype

        def declared_dtype(info):
            return explicit_dtype((info.attrs or {}).get("source_dtype", info.dtype))

        plan.output_dtype = output_dtype
        plan.output_dtype_known = declared_dtype(output_info) in {
            "float32",
            "float16",
        }
        source_shape = (output_info.attrs or {}).get("source_shape")

        if source_shape is not None and len(source_shape) == 1:
            extent = str(source_shape[0])

            if extent.isascii() and extent.isdecimal():
                plan.static_extent = int(extent)

        plan.input_dtypes = tuple(
            declared_dtype(context.tensors[name]) for name in plan.inputs
        )
        program = getattr(context.kernel, "ssa", None)
        plan.numeric_bounds = (
            program_facts(program, "numeric_bounds") if program else {}
        )

        if plan.large_vector_eligible:
            from ninetoothed.backends.toolchain import (
                normalize_ascendc_arch,
                resolve_ascendc_arch,
            )

            options = context.kernel.compiler_options.get("backend_options", {})

            try:
                arch = normalize_ascendc_arch(options.get("arch", "native"))

                if arch in {"native", "dav-c220"}:
                    plan.arch = resolve_ascendc_arch(arch)
            except (TypeError, ValueError, OSError):
                plan.arch = None

        if plan.chunk != plan.fallback_chunk:
            context.scheduled_metadata["ascendc_vector_chunk"] = {
                "policy": "static-large-single-input-zero-or-one-stage",
                "static_extent": plan.static_extent,
                "native_chunk": plan.chunk,
                "fallback_chunk": plan.fallback_chunk,
                "ub_bytes": plan.ub_bytes(plan.chunk),
                "usable_ub_bytes": ASCENDC_UB_BYTES,
                "arch": plan.arch,
                "maximum_float_repeats": 255,
                "chunk_alignment": 128,
            }

    return plan


def _packet_primitive(operation, operands, dtype):
    """Select SDK primitives for a serial queue packet over the shared DAG."""
    if dtype != "float32" or any(value.dtype != "float32" for value in operands):
        return None

    if any(
        value.literal is not None and not math.isfinite(value.literal)
        for value in operands
    ):
        return None

    code = operation.opcode.replace("call.", "math.")

    if code in _VEC_BINARY and len(operands) == 2:
        lhs, rhs = operands

        if lhs.scalar and rhs.scalar:
            return None

        if code == "arith.div" and not rhs.scalar:
            return VectorLowering(
                "ReciprocalStrict" if lhs.literal == 1 else "DivStrict"
            )

        if lhs.scalar or rhs.scalar:
            if code not in _VEC_SCALAR_BINARY:
                return None
            return VectorLowering(_VEC_SCALAR_BINARY[code])
        return VectorLowering(_VEC_BINARY[code])

    if len(operands) == 1 and not operands[0].scalar:
        if code == "arith.neg":
            return VectorLowering("Muls")

        if code in _VEC_UNARY:
            return VectorLowering(_VEC_UNARY[code])
    return None


def _packet_context(context, value, output):
    """Expand only target primitive recipes, preserving their original F32 steps."""
    operations = dict(context.operations)
    used = set(operations) | set(context.variables) | set(context.outputs)

    def add(code, operands, type_, final=None, literal=None):
        name = final or f"%packet{len(used)}"

        while final is None and name in used:
            name += "_"

        used.add(name)
        result = ssa.Value(name=name, type=type_)
        operations[name] = ssa.Operation(
            opcode=code,
            operands=operands,
            results=(result,),
            attrs={"value": literal} if code == "arith.constant" else {},
        )

        return name

    scalar_type = ssa.Type(kind="scalar", dtype="float32")

    for name, operation in tuple(operations.items()):
        code = operation.opcode.replace("call.", "math.")

        if (
            code not in {"math.sigmoid", "math.exp2", "math.log2"}
            or len(operation.operands) != 1
        ):
            continue

        type_ = operation.results[0].type
        operand = operation.operands[0]
        constant = add(
            "arith.constant",
            (),
            scalar_type,
            literal={
                "math.sigmoid": -1.0,
                "math.exp2": 0.69314718,
                "math.log2": 1.44269504,
            }[code],
        )

        if code == "math.log2":
            logarithm = add("math.log", (operand,), type_)
            add("arith.mul", (logarithm, constant), type_, final=name)
        else:
            scaled = add("arith.mul", (operand, constant), type_)

            if code == "math.exp2":
                add("math.exp", (scaled,), type_, final=name)
            else:
                exponential = add("math.exp", (scaled,), type_)
                one = add("arith.constant", (), scalar_type, literal=1.0)
                denominator = add("arith.add", (exponential, one), type_)
                add("math.reciprocal", (denominator,), type_, final=name)

    store = replace(context.stores[0], operands=(value, output))
    tensors = {
        **context.tensors,
        output: replace(context.tensors[output], dtype="float32"),
    }

    return replace(context, operations=operations, stores=(store,), tensors=tensors)


def _packet_stages(dag):
    """Render a storage view of typed nodes without reconstructing expression trees."""
    names = {
        step.node.result.name: f"nt_v{index + 8}"
        for index, step in enumerate(dag.steps)
    }
    stages = []

    def operand(value):
        if value.literal is not None:
            text = f"{float(value.literal):.9g}"

            return ("literal", text + ("f" if "." in text or "e" in text else ".0f"))
        return (
            ("temp", names[value.name])
            if value.name in names
            else ("input", value.name)
        )

    for step in dag.steps:
        args = tuple(operand(value) for value in step.node.operands)
        intrinsic = step.node.lowering.mode

        if step.node.operation.opcode == "arith.neg":
            args += (("literal", "-1.0f"),)

        if intrinsic == "ReciprocalStrict" and len(args) == 2:
            args = args[1:]

        if intrinsic in {"Adds", "Muls"} and args[0][0] == "literal":
            args = args[::-1]

        if intrinsic in {"Add", "Mul", "Max", "Min"} and tuple(
            kind for kind, _ in args
        ) == ("input", "temp"):
            args = args[::-1]

        stages.append((intrinsic, args, names[step.node.result.name]))
    return tuple(stages)


class AscendPacketStorage:
    def __init__(self, output, dag):
        self.output = output
        self.dag = dag
        self.inputs = tuple(item.value.name for item in dag.inputs)
        self.stages = _packet_stages(dag)
        self.output_dtype = "float32"
        self.output_dtype_known = False
        self.input_dtypes = ()
        self.static_extent = None
        self.arch = None
        self.resource_plan = None
        self.temporary_buffers = None
        self.ssa_values = {
            destination: (step.node.result.name,)
            for step, (_, _, destination) in zip(dag.steps, self.stages)
        }
        self.numeric_bounds = {}
        self.input_queue_slots = 2
        self.validity_summary = False
        self.native_division = False

    def native_division_proofs(self):
        """Prove physical storage, not merely distinct C++ variable names."""
        stages = self.stages
        buffered = stages if self.output_dtype == "float16" else stages[:-1]
        mapping = (
            self.temporary_buffers
            if self.temporary_buffers is not None
            else tuple((dst, i) for i, (_, _, dst) in enumerate(buffered))
        )
        slots = {name: ("temporary", slot) for name, slot in mapping}
        slots.update({name: ("input", i) for i, name in enumerate(self.inputs)})
        proofs = []

        for index, (intrinsic, operands, destination) in enumerate(stages):
            if intrinsic not in {"DivStrict", "ReciprocalStrict"}:
                continue

            dst_slot = (
                ("output", 0)
                if self.output_dtype == "float32" and index == len(stages) - 1
                else slots.get(destination)
            )
            operand_slots = tuple(
                slots.get(name) for kind, name in operands if kind != "literal"
            )
            proved = (
                self.output_dtype_known
                and self.input_dtypes
                and all(dtype == "float32" for dtype in self.input_dtypes)
                and self.strict_scratch
                and self.float_scratch
                and dst_slot is not None
                and all(slot is not None and slot != dst_slot for slot in operand_slots)
            )
            proofs.append(
                {
                    "operation": intrinsic,
                    "destination": destination,
                    "destination_slot": dst_slot,
                    "operand_slots": operand_slots,
                    "operands_preserved": bool(proved),
                    "source_positions": "VECIN or VECCALC",
                    "destination_position": "VECOUT or VECCALC",
                    "scratch_position": "separate VECCALC",
                    "lifetime": "through every original sparse repair",
                }
            )
        return tuple(proofs)

    def lower_bound(self, operand):
        """Look up only SSA-proved float32 values consumed by this operand."""
        kind, name = operand
        keys = self.ssa_values.get(name, ()) if kind == "temp" else (name,)
        bounds = [
            bound
            for key in keys
            if (bound := numeric_lower_bound(self.numeric_bounds, key)) is not None
        ]

        return max(bounds) if bounds else None

    def bound_consumers(self):
        result = []

        for intrinsic, operands, destination in self.stages:
            if intrinsic not in {"ReciprocalStrict", "RsqrtStrict", "DivStrict"}:
                continue

            operand = operands[1] if intrinsic == "DivStrict" else operands[0]
            bound = self.lower_bound(operand)

            if bound is None or bound < float.fromhex("0x1p-120"):
                continue

            kind, name = operand
            result.append(
                {
                    "operation": intrinsic,
                    "destination": destination,
                    "ssa_values": self.ssa_values.get(name, ())
                    if kind == "temp"
                    else (name,),
                    "lower": bound,
                }
            )
        return result

    def live_temporary_buffers(self):
        """Project the shared DAG's slots into the packet's VECCALC arena."""
        steps = (
            self.dag.steps if self.output_dtype == "float16" else self.dag.steps[:-1]
        )
        slots = dict.fromkeys(step.destination for step in steps)
        slots = {slot: index for index, slot in enumerate(slots)}

        return tuple(
            (self.stages[index][2], slots[step.destination])
            for index, step in enumerate(steps)
        )

    @property
    def fallback_chunk(self):
        """Keep the original scalar fallback task domain and UB policy."""
        chunk = _VECTOR_ELEM_CHUNK // (
            1 if len(self.stages) <= 1 else (2 if len(self.stages) <= 4 else 4)
        )
        # Float32's final stage writes directly to its output queue. Count it
        # once; half outputs retain every float stage and add a cast queue.
        buffers = 2 * len(self.inputs) + len(self.stages)

        if not self.stages or self.output_dtype == "float16":
            buffers += 1

        if self.float_scratch:
            buffers += 1

        mask_bytes = (0.25 if self.strict_scratch else 0) + (
            0.125 if self.extrema_scratch else 0
        )

        while chunk * (4 * buffers + mask_bytes) > ASCENDC_UB_BYTES:
            chunk //= 2
        return chunk

    @property
    def chunk(self):
        """Amortize one-packet setup only for proved large, simple vectors."""
        if self.resource_plan is not None:
            return self.resource_plan.selected.elements

        original = self.fallback_chunk

        if not self.large_vector_eligible or self.arch != "dav-c220":
            return original

        # One float repeat has 64 lanes. Stay below the explicit 255-repeat
        # API bound and align both float and half writeback to full sectors.
        maximum = min(2 * _VECTOR_ELEM_CHUNK, 255 * 64) // 128 * 128

        for candidate in range(maximum, original, -128):
            if self.ub_bytes(candidate) <= ASCENDC_UB_BYTES:
                return candidate

        return original

    @property
    def large_vector_eligible(self):
        return (
            self.static_extent is not None
            and self.static_extent >= 16 * 1024 * 1024
            and self.input_dtypes == ("float32",)
            and len(self.stages) <= 1
            and not self.strict_scratch
            and self.output_dtype_known
            and self.output_dtype in {"float32", "float16"}
        )

    def ub_bytes(self, chunk):
        """Count declared queue slots and scratch with 32-byte allocation."""

        def align(size):
            return (size + 31) // 32 * 32

        allocation = self.input_queue_slots * len(self.inputs) * align(chunk * 4)
        allocation += align(chunk * (2 if self.output_dtype == "float16" else 4))
        temporaries = len(self.stages)

        if self.stages and self.output_dtype == "float32":
            temporaries -= 1

        if self.temporary_buffers is not None:
            temporaries = (
                max((slot for _, slot in self.temporary_buffers), default=-1) + 1
            )

        allocation += temporaries * align(chunk * 4)

        if self.float_scratch:
            allocation += align(chunk * 4)

        if self.strict_scratch:
            allocation += 2 * align(chunk // 8)

        if self.extrema_scratch:
            allocation += align(chunk // 8)

        return allocation

    @property
    def strict_scratch(self):
        return any(
            op
            in {
                "ReciprocalStrict",
                "RsqrtStrict",
                "DivStrict",
                "MaxStrict",
                "MinStrict",
                "Tanh",
                "Floor",
                "Ceil",
            }
            for op, _, _ in self.stages
        )

    @property
    def float_scratch(self):
        return any(
            op in {"ReciprocalStrict", "RsqrtStrict", "DivStrict"}
            for op, _, _ in self.stages
        )

    @property
    def extrema_scratch(self):
        return any(op in {"MaxStrict", "MinStrict"} for op, _, _ in self.stages)


_TAIL_UNARY = {
    "Exp": "nt_exp",
    "Ln": "nt_log",
    "Abs": "nt_fabs",
    "Sqrt": "sqrt",
    "Rsqrt": "nt_rsqrt",
    "RsqrtStrict": "nt_rsqrt",
    "Reciprocal": None,
    "ReciprocalStrict": None,
    "Relu": None,
    "Floor": "nt_floor",
    "Ceil": "nt_ceil",
    "Sin": "nt_sin",
    "Cos": "nt_cos",
    "Tan": "nt_tan",
    "Tanh": "nt_tanh",
    "Erf": "nt_erf",
    "Atan": "nt_atan",
    "Sigmoid": "nt_sigmoid",
}

_TAIL_BINARY = {
    "Add": "+",
    "Sub": "-",
    "Mul": "*",
    "Div": "/",
    "DivStrict": "/",
    "Max": "?>",
    "Min": "?<",
    "MaxStrict": "?>",
    "MinStrict": "?<",
    "Adds": "+",
    "Muls": "*",
    "Maxs": "?>",
    "Mins": "?<",
}


def _tail_expression(stages):
    """Render a scalar tail expression mirroring the vector stages."""
    stage_by_dst = {dst: (intrinsic, args) for intrinsic, args, dst in stages}
    memo = {}

    def render_arg(arg):
        kind, name = arg

        if kind == "input":
            return f"nt_p_{name}[j]"

        if kind == "literal":
            return name

        return render_stage(name)

    def render_stage(dst):
        if dst in memo:
            return memo[dst]

        intrinsic, args = stage_by_dst[dst]
        rendered = tuple(render_arg(a) for a in args)
        expr = _apply_tail(intrinsic, rendered)
        memo[dst] = expr

        return expr

    last = stages[-1][2]
    intrinsic, args = stage_by_dst[last]

    return _apply_tail(intrinsic, tuple(render_arg(a) for a in args))


def _apply_tail(intrinsic, rendered):
    if intrinsic in _TAIL_UNARY and len(rendered) == 1:
        fn = _TAIL_UNARY[intrinsic]

        if fn is None:
            if intrinsic in {"Reciprocal", "ReciprocalStrict"}:
                return f"(1.0f / {rendered[0]})"

            return f"({rendered[0]} > 0.0f ? {rendered[0]} : 0.0f)"

        return f"{fn}({rendered[0]})"

    if intrinsic in _TAIL_BINARY and len(rendered) == 2:
        op = _TAIL_BINARY[intrinsic]

        if op in {"?>", "?<"}:
            lhs, rhs = rendered
            comparison = ">" if op == "?>" else "<"

            return f"({lhs} != {lhs} ? {lhs} : ({rhs} != {rhs} ? {rhs} : ({lhs} {comparison} {rhs} ? {lhs} : {rhs})))"

        return f"({rendered[0]} {op} {rendered[1]})"

    return None


def render_vector_elementwise(plan, total_expr):
    """Emit the vector kernel body for the compiled plan."""
    if plan.native_division:
        proofs = plan.native_division_proofs()

        if not proofs or not all(proof["operands_preserved"] for proof in proofs):
            raise ValueError(
                "Native division requires proved float32 vector operand lifetimes."
            )

    chunk = plan.chunk
    output_type = "half" if plan.output_dtype == "float16" else "float"
    # Scalar GM tails write back 64-byte lines. DMA and the scalar tail must
    # not touch the same line, even inside one logical task.
    alignment = 32 if output_type == "half" else 16
    names = (*plan.inputs, plan.output)

    setbuf = "\n".join(
        f"    GlobalTensor<{output_type if name == plan.output else 'float'}> nt_gm_{name};"
        f"\n    nt_gm_{name}.SetGlobalBuffer((__gm__ {output_type if name == plan.output else 'float'}*){name}_gm);"
        f"\n    __gm__ {output_type if name == plan.output else 'float'}* nt_p_{name} = (__gm__ {output_type if name == plan.output else 'float'}*){name}_gm;"
        for name in names
    )

    if not plan.stages:
        # Pure copy: aligned block move with a scalar tail.
        src = plan.inputs[0]

        copy = (
            "Cast(nt_lo, nt_a0, RoundMode::CAST_RINT, (int32_t)nt_aligned);"
            if output_type == "half"
            else "DataCopy(nt_lo, nt_a0, (uint32_t)nt_aligned);"
        )

        return f"""    if ((int64_t)GetBlockIdx() >= ({total_expr} + {chunk} - 1) / {chunk}) {{
        return;
    }}
    int64_t nt_base = (int64_t)GetBlockIdx() * {chunk};
    int64_t nt_cnt = {total_expr} - nt_base;
    if (nt_cnt > {chunk}) {{ nt_cnt = {chunk}; }}
    int64_t nt_aligned = nt_cnt & ~{alignment - 1}LL;
{setbuf}
    if (nt_aligned > 0) {{
        TPipe nt_pipe;
        TQue<TPosition::VECIN, 1> nt_q0;
        TQue<TPosition::VECOUT, 1> nt_qo;
        nt_pipe.InitBuffer(nt_q0, {plan.input_queue_slots}, {chunk} * sizeof(float));
        nt_pipe.InitBuffer(nt_qo, 1, {chunk} * sizeof({output_type}));
        LocalTensor<float> nt_in0 = nt_q0.AllocTensor<float>();
        DataCopy(nt_in0, nt_gm_{src}[nt_base], (uint32_t)nt_aligned);
        nt_q0.EnQue(nt_in0);
        LocalTensor<float> nt_a0 = nt_q0.DeQue<float>();
        LocalTensor<{output_type}> nt_lo = nt_qo.AllocTensor<{output_type}>();
        {copy}
        nt_qo.EnQue(nt_lo);
        LocalTensor<{output_type}> nt_vo = nt_qo.DeQue<{output_type}>();
        DataCopy(nt_gm_{plan.output}[nt_base], nt_vo, (uint32_t)nt_aligned);
        nt_q0.FreeTensor<float>(nt_a0);
        nt_qo.FreeTensor<{output_type}>(nt_vo);
    }}
    if (nt_aligned < nt_cnt) {{ PipeBarrier<PIPE_ALL>(); }}
    for (int64_t j = nt_base + nt_aligned; j < nt_base + nt_cnt; j++) {{
        nt_p_{plan.output}[j] = nt_p_{src}[j];
    }}"""

    loads = "\n".join(
        f"        DataCopy(nt_in{i}, nt_gm_{name}[nt_base], (uint32_t)nt_aligned);"
        for i, name in enumerate(plan.inputs)
    )

    in_queues = "\n".join(
        f"        TQue<TPosition::VECIN, 2> nt_q{i};\n"
        f"        nt_pipe.InitBuffer(nt_q{i}, {plan.input_queue_slots}, {chunk} * sizeof(float));\n"
        f"        LocalTensor<float> nt_in{i} = nt_q{i}.AllocTensor<float>();"
        for i in range(len(plan.inputs))
    )

    enq_deq = "\n".join(
        f"        nt_q{i}.EnQue(nt_in{i});" for i in range(len(plan.inputs))
    )
    deq = "\n".join(
        f"        LocalTensor<float> nt_a{i} = nt_q{i}.DeQue<float>();"
        for i in range(len(plan.inputs))
    )
    frees = "\n".join(
        f"        nt_q{i}.FreeTensor<float>(nt_a{i});" for i in range(len(plan.inputs))
    )

    temps = [dst for _, _, dst in plan.stages]
    last = temps[-1]

    buffered_temps = temps if output_type == "half" else temps[:-1]
    temporary_buffers = (
        plan.temporary_buffers
        if plan.temporary_buffers is not None
        else tuple((dst, index) for index, dst in enumerate(buffered_temps))
    )
    temporary_slots = max((slot for _, slot in temporary_buffers), default=-1) + 1
    temp_allocs = "\n".join(
        f"        TBuf<TPosition::VECCALC> nt_b{i};\n"
        f"        nt_pipe.InitBuffer(nt_b{i}, {chunk} * sizeof(float));"
        for i in range(temporary_slots)
    )
    temp_gets = "\n".join(
        f"        LocalTensor<float> {dst} = nt_b{slot}.Get<float>();"
        for dst, slot in temporary_buffers
    )

    if plan.strict_scratch:
        temp_allocs += f"""
        TBuf<TPosition::VECCALC> nt_ref_mask_buf0;
        TBuf<TPosition::VECCALC> nt_ref_mask_buf1;
        nt_pipe.InitBuffer(nt_ref_mask_buf0, {chunk} / 8);
        nt_pipe.InitBuffer(nt_ref_mask_buf1, {chunk} / 8);"""
        temp_gets += """
        LocalTensor<uint8_t> nt_ref_mask0 = nt_ref_mask_buf0.Get<uint8_t>();
        LocalTensor<uint8_t> nt_ref_mask1 = nt_ref_mask_buf1.Get<uint8_t>();"""

    if plan.float_scratch:
        temp_allocs += f"\n        TBuf<TPosition::VECCALC> nt_ref_buf0;\n        nt_pipe.InitBuffer(nt_ref_buf0, {chunk} * sizeof(float));"
        temp_gets += "\n        LocalTensor<float> nt_ref0 = nt_ref_buf0.Get<float>();"

    if plan.extrema_scratch:
        temp_allocs += f"\n        TBuf<TPosition::VECCALC> nt_ref_mask_buf2;\n        nt_pipe.InitBuffer(nt_ref_mask_buf2, {chunk} / 8);"
        temp_gets += "\n        LocalTensor<uint8_t> nt_ref_mask2 = nt_ref_mask_buf2.Get<uint8_t>();"

    def src_expr(arg):
        kind, name = arg

        if kind == "input":
            return f"nt_a{plan.inputs.index(name)}"

        if kind == "literal":
            return name

        return name

    stage_calls = []

    for intrinsic, args, dst in plan.stages:
        srcs = ", ".join(src_expr(a) for a in args)
        destination = "nt_lo" if dst == last and output_type == "float" else dst

        if intrinsic == "DivStrict":
            stage_calls.append(
                render_division(
                    destination,
                    src_expr(args[0]),
                    src_expr(args[1]),
                    "nt_aligned",
                    scalar_lhs=args[0][0] == "literal",
                    known_rhs_lower_bound=plan.lower_bound(args[1]),
                    validity_summary=plan.validity_summary,
                    native_division=plan.native_division,
                    operands_preserved=plan.native_division,
                )
            )
            continue

        if intrinsic in {"ReciprocalStrict", "RsqrtStrict"}:
            stage_calls.append(
                render_refinement(
                    intrinsic,
                    destination,
                    srcs,
                    "nt_aligned",
                    known_lower_bound=plan.lower_bound(args[0]),
                    validity_summary=plan.validity_summary,
                    native_division=plan.native_division,
                    operands_preserved=plan.native_division,
                )
            )
            continue

        if intrinsic in {"MaxStrict", "MinStrict"}:
            stage_calls.append(
                render_extrema(
                    intrinsic,
                    destination,
                    src_expr(args[0]),
                    src_expr(args[1]),
                    "nt_aligned",
                    scalar_lhs=args[0][0] == "literal",
                    scalar_rhs=args[1][0] == "literal",
                    swap_inputs=args[0][0] == "input"
                    and args[1][0] in {"temp", "literal"},
                )
            )
            continue

        if intrinsic in {"Tanh", "Floor", "Ceil"}:
            stage_calls.append(
                render_unary_zero_repair(
                    intrinsic, destination, src_expr(args[0]), "nt_aligned"
                )
            )
            continue

        stage_calls.append(
            f"        {intrinsic}({destination}, {srcs}, (int32_t)nt_aligned);"
        )
        stage_calls.append("        PipeBarrier<PIPE_V>();")

    if output_type == "half":
        stage_calls.append(
            f"        Cast(nt_lo, {last}, RoundMode::CAST_RINT, (int32_t)nt_aligned);"
        )

    stage_calls = "\n".join(stage_calls)

    tail = _tail_expression(plan.stages)

    if tail is None:
        raise ValueError("The AscendC vector plan has no scalar tail expression.")

    if output_type == "half":
        tail = f"(half)(float)({tail})"

    tail_block = (
        f"    for (int64_t j = nt_base + nt_aligned; "
        f"j < nt_base + nt_cnt; j++) {{\n"
        f"        nt_p_{plan.output}[j] = {tail};\n"
        f"    }}"
    )

    return f"""    if ((int64_t)GetBlockIdx() >= ({total_expr} + {chunk} - 1) / {chunk}) {{
        return;
    }}
    int64_t nt_base = (int64_t)GetBlockIdx() * {chunk};
    int64_t nt_cnt = {total_expr} - nt_base;
    if (nt_cnt > {chunk}) {{ nt_cnt = {chunk}; }}
    int64_t nt_aligned = nt_cnt & ~{alignment - 1}LL;
{setbuf}
    if (nt_aligned > 0) {{
        TPipe nt_pipe;
{in_queues}
        TQue<TPosition::VECOUT, 1> nt_qo;
        nt_pipe.InitBuffer(nt_qo, 1, {chunk} * sizeof({output_type}));
{temp_allocs}
{loads}
{enq_deq}
{deq}
{temp_gets}
        LocalTensor<{output_type}> nt_lo = nt_qo.AllocTensor<{output_type}>();
{stage_calls}
        nt_qo.EnQue(nt_lo);
        LocalTensor<{output_type}> nt_vo = nt_qo.DeQue<{output_type}>();
        DataCopy(nt_gm_{plan.output}[nt_base], nt_vo, (uint32_t)nt_aligned);
{frees}
        nt_qo.FreeTensor<{output_type}>(nt_vo);
    }}
    if (nt_aligned < nt_cnt) {{ PipeBarrier<PIPE_ALL>(); }}
{tail_block}"""


__all__ = [
    "match_vector_elementwise",
    "render_vector_elementwise",
    "_VECTOR_ELEM_CHUNK",
]
