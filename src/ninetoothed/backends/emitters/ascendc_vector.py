"""Vector elementwise lowering for the AscendC backend.

Expression trees over float32 1-D elementwise kernels map onto
TPipe/LocalTensor vector intrinsics with aligned block copies; a scalar
tail loop covers the non-8-multiple remainder (raw DataCopy counts must
stay aligned).  Unary functions without a direct intrinsic compose from
the primitive vector set (e.g. sigmoid = Reciprocal(Add(Exp(Neg(x)), 1))).
"""

_VEC_BINARY = {
    "arith.add": "Add",
    "arith.sub": "Sub",
    "arith.mul": "Mul",
    "arith.div": "Div",
    "arith.maximum": "Max",
    "arith.minimum": "Min",
}

_VEC_COMMUTATIVE = {"arith.add", "arith.mul", "arith.maximum", "arith.minimum"}

_VEC_SCALAR_BINARY = {
    "arith.add": "Adds",
    "arith.mul": "Muls",
    "arith.maximum": "Maxs",
    "arith.minimum": "Mins",
}

_VEC_UNARY = {
    "math.exp": "Exp",
    "math.log": "Ln",
    "math.abs": "Abs",
    "math.sqrt": "Sqrt",
    "math.rsqrt": "Rsqrt",
    "math.reciprocal": "Reciprocal",
    "math.relu": "Relu",
    "math.floor": "Floor",
    "math.ceil": "Ceil",
    "math.sin": "Sin",
    "math.cos": "Cos",
    "math.tan": "Tan",
    "math.tanh": "Tanh",
    "math.erf": "Erf",
    "math.atan": "Atan",
    "math.sigmoid": "Sigmoid",
}

_VECTOR_ELEM_CHUNK = 8192
_MAX_STAGES = 8


def match_vector_elementwise(context, normalize_dtype):
    """Return a compiled vector plan for a float32 1-D elementwise store."""
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

    if normalize_dtype(output_info.dtype) != "float32":
        return None

    tree = _build_tree(context, normalize_dtype, store.operands[0])

    if tree is None:
        return None

    return _compile(tree, output)


def _tensor_kind(context, normalize_dtype, name):
    info = context.tensors.get(name)

    if info is None:
        return None

    if normalize_dtype(info.dtype) != "float32":
        return None

    return "scalar" if info.ndim == 0 else "tensor"


def _build_tree(context, normalize_dtype, value, depth=0):
    if depth > 7:
        return None

    kind = _tensor_kind(context, normalize_dtype, value)

    if kind is not None:
        return ("input", value, kind)

    producer = context.operations.get(value)

    if producer is None:
        return None

    if producer.opcode == "arith.constant":
        constant = producer.attrs.get("value") if producer.attrs else None

        if isinstance(constant, (int, float)):
            text = f"{float(constant):.9g}"

            if "e" not in text and "." not in text:
                text += ".0"

            return ("literal", text + "f")

        return None

    opcode = producer.opcode
    children = tuple(
        _build_tree(context, normalize_dtype, operand, depth + 1)
        for operand in producer.operands
    )

    if any(child is None for child in children):
        return None

    if opcode.startswith("call."):
        opcode = "math." + opcode[len("call.") :]

    composed = _composed_tree(opcode)

    if composed is not None:
        if len(children) != 1 or children[0][0] != "input":
            return None

        return _substitute(composed, children[0])

    if opcode == "arith.div" and len(children) == 2:
        # The vector Div intrinsic computes unreliably on this platform
        # (measured O(1) errors); a / b rewrites to a * Reciprocal(b), and
        # 1 / b collapses to Reciprocal(b) outright.  Sigmoid-shaped
        # denominators a / (1 + exp(-c*a)) lower to a * Sigmoid(c*a).
        lhs, rhs = children

        if lhs[0] == "input":
            sigmoid_form = _match_sigmoid_denominator(rhs, lhs)

            if sigmoid_form is not None:
                return ("call", "Mul", (lhs, sigmoid_form), "")

        if rhs[0] in {"input", "call"}:
            if lhs[0] == "literal":
                try:
                    if float(lhs[1].rstrip("f")) == 1.0:
                        return ("call", "Reciprocal", (rhs,), "")
                except ValueError:
                    pass

            recip = ("call", "Reciprocal", (rhs,), "")

            return ("call", "Mul", (lhs, recip), "")

    if opcode in _VEC_BINARY:
        if len(children) != 2 or any(
            c[0] not in {"input", "call", "literal"} for c in children
        ):
            return None

        if any(c[0] == "literal" for c in children):
            # Literal operand: `1 / x` maps to Reciprocal; other literals
            # commute to the scalar intrinsic form.  The non-literal side
            # may itself be a composed expression (e.g. `1 + exp(-x)`).
            literal = next(c for c in children if c[0] == "literal")
            other = next(c for c in children if c[0] != "literal")

            if other[0] not in {"input", "call"}:
                return None

            try:
                is_one = float(literal[1].rstrip("f")) == 1.0
            except ValueError:
                is_one = False

            if opcode == "arith.div" and is_one and children[0][0] == "literal":
                return ("call", "Reciprocal", (other,), "")

            if opcode not in _VEC_SCALAR_BINARY:
                return None

            return ("call", _VEC_SCALAR_BINARY[opcode], (other, literal), "")

        return ("call", _VEC_BINARY[opcode], children, "")

    if opcode in _VEC_SCALAR_BINARY:
        if len(children) != 2:
            return None

        tensor = None
        scalar = None

        for child in children:
            if child[0] != "input":
                return None

            if child[2] == "tensor" and tensor is None:
                tensor = child

            elif child[2] == "scalar" and scalar is None:
                scalar = child

            else:
                return None

        if tensor is None or scalar is None:
            return None

        return ("call", _VEC_SCALAR_BINARY[opcode], (tensor, scalar), "")

    if opcode in _VEC_UNARY:
        if len(children) != 1 or children[0][0] not in {"input", "call"}:
            return None

        return ("call", _VEC_UNARY[opcode], children, "")

    if opcode.startswith("math."):
        base = opcode[len("math.") :]

        if base in _VEC_UNARY:
            if len(children) != 1 or children[0][0] not in {"input", "call"}:
                return None

            return ("call", _VEC_UNARY[base], children, "")

    return None


def _match_sigmoid_denominator(denominator, operand):
    """Match a ``1 + exp(-c*x)`` denominator tree over ``operand``.

    Returns the equivalent ``Sigmoid(c*x)`` tree, or ``None``.
    """
    if denominator[0] != "call" or denominator[1] != "Adds":
        return None

    args = denominator[2]

    if len(args) != 2:
        return None

    exp_node, one = args

    if one[0] != "literal" or float(one[1].rstrip("f")) != 1.0:
        return None

    if exp_node[0] != "call" or exp_node[1] != "Exp":
        return None

    (inner,) = exp_node[2]

    scale = 1.0
    current = inner

    while current[0] == "call" and current[1] == "Muls" and len(current[2]) == 2:
        factor = current[2][1]

        if factor[0] != "literal":
            return None

        scale *= float(factor[1].rstrip("f"))
        current = current[2][0]

    if scale >= 0 or current != operand:
        return None

    positive = -scale

    scaled = (
        "call",
        "Muls",
        (operand, ("literal", f"{positive:.9g}f")),
        "",
    )

    return ("call", "Sigmoid", (scaled,), "")


def _substitute(tree, replacement):
    kind = tree[0]

    if kind == "input":
        return replacement

    if kind == "call":
        return (
            "call",
            tree[1],
            tuple(_substitute(c, replacement) for c in tree[2]),
            tree[3],
        )

    return tree


def _composed_tree(opcode):
    x = ("input", "@", "tensor")

    if opcode == "arith.neg":
        return ("call", "Muls", (x, ("literal", "-1.0f")), "")

    if opcode == "math.exp2":
        scaled = ("call", "Muls", (x, ("literal", "0.69314718f")), "")

        return ("call", "Exp", (scaled,), "")

    if opcode == "math.log2":
        ln = ("call", "Ln", (x,), "")

        return ("call", "Muls", (ln, ("literal", "1.44269504f")), "")

    return None


class _Plan:
    def __init__(self, output, inputs, stages):
        self.output = output
        self.inputs = tuple(inputs)
        self.stages = tuple(stages)


def _compile(tree, output):
    stages = []
    temps = {}

    def emit(node):
        kind = node[0]

        if kind == "input":
            return ("input", node[1])

        if kind == "literal":
            return ("literal", node[1])

        intrinsic = node[1]
        args = tuple(emit(child) for child in node[2])

        if any(arg is None for arg in args):
            return None

        for arg in args:
            if arg[0] == "call_result":
                return None

        key = (intrinsic, tuple(args))

        if key in temps:
            return ("temp", temps[key])

        dst = f"nt_v{len(stages) + 8}"
        stages.append((intrinsic, args, dst))
        temps[key] = dst

        return ("temp", dst)

    result = emit(tree)

    if result is None:
        return None

    if len(stages) > _MAX_STAGES:
        return None

    if result[0] == "input":
        # Pure copy: no compute stages, result is the input itself.
        return _Plan(output, [result[1]], [])

    if result[0] != "temp" or not stages:
        return None

    # Vector ops with a VECIN-queue tensor in src0 alongside a VECCALC
    # tensor in src1 miscompute on this platform (measured 0.0 for
    # `Mul(input, temp)`); commutative stages normalize to temp-first.
    commutative = {"Add", "Mul", "Max", "Min"}
    normalized = []

    for intrinsic, args, dst in stages:
        kinds = [kind for kind, _ in args]

        if intrinsic in commutative and len(args) == 2 and kinds == ["input", "temp"]:
            args = (args[1], args[0])

        normalized.append((intrinsic, args, dst))

    stages = normalized

    # Reciprocal over a computed (temp) operand only computes correctly as
    # the terminal stage on this platform (measured 0.0 when its result
    # feeds a later vector op), so such chains stay on the scalar path;
    # `x / y` (reciprocal of a raw input) remains vectorizable.
    for index, (intrinsic, args, dst) in enumerate(stages):
        if intrinsic != "Reciprocal":
            continue

        src_is_temp = any(kind == "temp" for kind, _ in args)

        if not src_is_temp:
            continue

        for _later_intrinsic, later_args, _later in stages[index + 1 :]:
            if any(kind == "temp" and name == dst for kind, name in later_args):
                return None

    inputs = []

    for _, args, _dst in stages:
        for kind, name in args:
            if kind == "input" and name not in inputs:
                inputs.append(name)

    return _Plan(output, inputs, stages)


_TAIL_UNARY = {
    "Exp": "nt_exp",
    "Ln": "nt_log",
    "Abs": "nt_fabs",
    "Sqrt": "sqrt",
    "Rsqrt": "nt_rsqrt",
    "Reciprocal": None,
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
    "Max": "?>",
    "Min": "?<",
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
            if intrinsic == "Reciprocal":
                return f"(1.0f / {rendered[0]})"

            return f"({rendered[0]} > 0.0f ? {rendered[0]} : 0.0f)"

        return f"{fn}({rendered[0]})"

    if intrinsic in _TAIL_BINARY and len(rendered) == 2:
        op = _TAIL_BINARY[intrinsic]

        if op == "?>":
            return f"({rendered[0]} > {rendered[1]} ? {rendered[0]} : {rendered[1]})"

        if op == "?<":
            return f"({rendered[0]} < {rendered[1]} ? {rendered[0]} : {rendered[1]})"

        return f"({rendered[0]} {op} {rendered[1]})"

    return None


def render_vector_elementwise(plan, total_expr):
    """Emit the vector kernel body for the compiled plan."""
    chunk = (
        _VECTOR_ELEM_CHUNK
        if len(plan.stages) <= 1
        else (
            _VECTOR_ELEM_CHUNK // 2
            if len(plan.stages) <= 4
            else _VECTOR_ELEM_CHUNK // 4
        )
    )
    names = (*plan.inputs, plan.output)

    setbuf = "\n".join(
        f"    GlobalTensor<float> nt_gm_{name};"
        f"\n    nt_gm_{name}.SetGlobalBuffer((__gm__ float*){name}_gm);"
        f"\n    __gm__ float* nt_p_{name} = (__gm__ float*){name}_gm;"
        for name in names
    )

    if not plan.stages:
        # Pure copy: aligned block move with a scalar tail.
        src = plan.inputs[0]

        return f"""    if ((int64_t)GetBlockIdx() >= ({total_expr} + {chunk} - 1) / {chunk}) {{
        return;
    }}
    int64_t nt_base = (int64_t)GetBlockIdx() * {chunk};
    int64_t nt_cnt = {total_expr} - nt_base;
    if (nt_cnt > {chunk}) {{ nt_cnt = {chunk}; }}
    int64_t nt_aligned = nt_cnt & ~7LL;
{setbuf}
    if (nt_aligned > 0) {{
        TPipe nt_pipe;
        TQue<TPosition::VECIN, 1> nt_q0;
        TQue<TPosition::VECOUT, 1> nt_qo;
        nt_pipe.InitBuffer(nt_q0, 1, {chunk} * sizeof(float));
        nt_pipe.InitBuffer(nt_qo, 1, {chunk} * sizeof(float));
        LocalTensor<float> nt_in0 = nt_q0.AllocTensor<float>();
        DataCopy(nt_in0, nt_gm_{src}[nt_base], (uint32_t)nt_aligned);
        nt_q0.EnQue(nt_in0);
        LocalTensor<float> nt_a0 = nt_q0.DeQue<float>();
        LocalTensor<float> nt_lo = nt_qo.AllocTensor<float>();
        DataCopy(nt_lo, nt_a0, (uint32_t)nt_aligned);
        nt_qo.EnQue(nt_lo);
        LocalTensor<float> nt_vo = nt_qo.DeQue<float>();
        DataCopy(nt_gm_{plan.output}[nt_base], nt_vo, (uint32_t)nt_aligned);
        nt_q0.FreeTensor<float>(nt_a0);
        nt_qo.FreeTensor<float>(nt_vo);
    }}
    for (int64_t j = nt_base + nt_aligned; j < nt_base + nt_cnt; j++) {{
        nt_p_{plan.output}[j] = nt_p_{src}[j];
    }}"""

    loads = "\n".join(
        f"        DataCopy(nt_in{i}, nt_gm_{name}[nt_base], (uint32_t)nt_aligned);"
        for i, name in enumerate(plan.inputs)
    )

    in_queues = "\n".join(
        f"        TQue<TPosition::VECIN, 2> nt_q{i};\n"
        f"        nt_pipe.InitBuffer(nt_q{i}, 2, {chunk} * sizeof(float));\n"
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

    temp_allocs = "\n".join(
        f"        TBuf<TPosition::VECCALC> nt_b{i};\n"
        f"        nt_pipe.InitBuffer(nt_b{i}, {chunk} * sizeof(float));"
        for i in range(len(temps) - 1)
    )
    temp_gets = "\n".join(
        f"        LocalTensor<float> {dst} = nt_b{i}.Get<float>();"
        for i, dst in enumerate(temps[:-1])
    )

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

        if dst == last:
            stage_calls.append(
                f"        {intrinsic}(nt_lo, {srcs}, (int32_t)nt_aligned);"
            )
        else:
            stage_calls.append(
                f"        {intrinsic}({dst}, {srcs}, (int32_t)nt_aligned);"
            )

    stage_calls = "\n".join(stage_calls)

    tail = _tail_expression(plan.stages)

    if tail is None:
        tail = "0.0f"

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
    int64_t nt_aligned = nt_cnt & ~7LL;
{setbuf}
    if (nt_aligned > 0) {{
        TPipe nt_pipe;
{in_queues}
        TQue<TPosition::VECOUT, 1> nt_qo;
        nt_pipe.InitBuffer(nt_qo, 1, {chunk} * sizeof(float));
{temp_allocs}
{loads}
{enq_deq}
{deq}
{temp_gets}
        LocalTensor<float> nt_lo = nt_qo.AllocTensor<float>();
{stage_calls}
        nt_qo.EnQue(nt_lo);
        LocalTensor<float> nt_vo = nt_qo.DeQue<float>();
        DataCopy(nt_gm_{plan.output}[nt_base], nt_vo, (uint32_t)nt_aligned);
{frees}
        nt_qo.FreeTensor<float>(nt_vo);
    }}
{tail_block}"""


__all__ = [
    "match_vector_elementwise",
    "render_vector_elementwise",
    "_VECTOR_ELEM_CHUNK",
]
