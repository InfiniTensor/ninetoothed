"""Vector reduce+broadcast lowering for the AscendC backend.

Row reductions over aligned 2-D inputs map onto UB staging with ping-pong
tree halving (Add/Max on TBuf intermediates).  Unaligned column counts
fall back to the scalar cooperative path because GM DataCopy addresses
must stay 32-byte aligned.
"""

_REDUCE_CHUNK = 8192


def match_reduce_broadcast(context):
    """Detect `reduce_op(x) [+ scale] broadcast to the row shape`.

    Returns ``(op, input, output, rows, cols, scale)`` or ``None``.
    """
    stores = [op for op in context.stores if len(op.operands) == 2]

    if len(stores) != 1 or len(context.outputs) != 1:
        return None

    output = context.outputs[0]
    store = stores[0]

    if store.operands[1] != output:
        return None

    producer = context.operations.get(store.operands[0])

    if producer is None or producer.opcode != "arith.add":
        return None

    if len(producer.operands) != 2:
        return None

    chain = None

    for operand in producer.operands:
        chain = _extract_reduce_chain(operand, context)

        if chain is not None:
            break

    if chain is None:
        return None

    operator, x_name, scale = chain
    input_info = context.tensors.get(x_name)
    output_info = context.tensors.get(output)

    if input_info is None or output_info is None:
        return None

    if input_info.ndim != 2 or output_info.ndim != 2:
        return None

    attrs = input_info.attrs or {}
    shape = attrs.get("source_shape") or input_info.shape

    if not shape or len(shape) != 2:
        return None

    cols_str = str(shape[1])

    try:
        cols = int(cols_str)

        if cols % 8 != 0 or cols > _REDUCE_CHUNK:
            return None
    except ValueError:
        return None

    if operator not in {"sum", "max", "min", "sum_sq"}:
        return None

    return (operator, x_name, output, str(shape[0]), cols_str, scale)


def _extract_reduce_chain(value, context):
    """Walk a value back to the reduction + scale expression."""
    if value is None:
        return None

    node = context.operations.get(value)

    if node is None:
        return None

    opcode = node.opcode

    if opcode.startswith("reduce."):
        operator = opcode[len("reduce.") :]

        if operator not in {"sum", "max", "min"} or not node.operands:
            return None

        return (operator, node.operands[0], "")

    if opcode == "arith.div" and len(node.operands) == 2:
        inner = _extract_reduce_chain(node.operands[0], context)

        if inner is None:
            return None

        divisor = context.operations.get(node.operands[1])

        if divisor is not None and divisor.opcode == "arith.constant":
            value2 = divisor.attrs.get("value") if divisor.attrs else None

            if isinstance(value2, (int, float)) and value2 != 0:
                text = f"{float(value2):.9g}"

                if "." not in text:
                    text += ".0"

                return (inner[0], inner[1], " / " + text + "f")

        if divisor is not None and divisor.opcode == "shape.dim":
            dim = divisor.attrs.get("dim") if divisor.attrs else None

            if dim == 1:
                input_info = context.tensors.get(inner[1])

                if input_info is not None:
                    attrs = input_info.attrs or {}
                    shape = attrs.get("source_shape")

                    if shape and len(shape) == 2:
                        try:
                            cols = float(shape[1])
                            text = f"{cols:.9g}"

                            if "." not in text:
                                text += ".0"

                            return (inner[0], inner[1], " / " + text + "f")
                        except (ValueError, TypeError):
                            pass

        return None

    if opcode == "math.sqrt" and len(node.operands) == 1:
        sum_op = context.operations.get(node.operands[0])

        if sum_op is not None and sum_op.opcode == "reduce.sum" and sum_op.operands:
            sq_op = context.operations.get(sum_op.operands[0])

            if (
                sq_op is not None
                and sq_op.opcode == "arith.mul"
                and len(sq_op.operands) == 2
                and sq_op.operands[0] == sq_op.operands[1]
            ):
                return ("sum_sq", sq_op.operands[0], "")

        return None

    return None


def render_reduce_broadcast(match):
    """Emit a UB vector reduce+broadcast with PipeBarrier sync."""
    operator, name, output, rows, cols, scale = match
    chunk = _REDUCE_CHUNK
    n = chr(10)
    lines = [
        f"    if ((int64_t)GetBlockIdx() >= ({rows})) {{ return; }}",
        f"    const int64_t nt_cols = {cols};",
        f"    GlobalTensor<float> nt_gm_{name};",
        f"    nt_gm_{name}.SetGlobalBuffer((__gm__ float*){name}_gm);",
        f"    GlobalTensor<float> nt_gm_{output};",
        f"    nt_gm_{output}.SetGlobalBuffer((__gm__ float*){output}_gm);",
        "    TPipe nt_pipe;",
        "    TQue<TPosition::VECIN, 1> nt_qi;",
        "    TQue<TPosition::VECOUT, 1> nt_qo;",
        "    TBuf<TPosition::VECCALC> nt_bw;",
        f"    nt_pipe.InitBuffer(nt_qi, 1, {chunk} * sizeof(float));",
        f"    nt_pipe.InitBuffer(nt_qo, 1, {chunk} * sizeof(float));",
        f"    nt_pipe.InitBuffer(nt_bw, {chunk} * sizeof(float));",
        "    LocalTensor<float> nt_x = nt_qi.AllocTensor<float>();",
        f"    DataCopy(nt_x, nt_gm_{name}[(int64_t)GetBlockIdx() * nt_cols], (uint32_t)nt_cols);",
        "    nt_qi.EnQue(nt_x);",
        "    nt_x = nt_qi.DeQue<float>();",
        "    LocalTensor<float> nt_w = nt_bw.Get<float>();",
        "    // Vector copy preserves the original for the broadcast store.",
        "    Adds(nt_w, nt_x, 0.0f, (int32_t)nt_cols);",
        "    PipeBarrier<PIPE_V>();",
    ]

    tree_op = "Max" if operator == "max" else "Add"

    if operator == "min":
        tree_op = "Min"

    if operator in {"sum", "sum_sq", "max", "min"}:
        if operator == "sum_sq":
            lines.append("    Mul(nt_w, nt_w, nt_w, (int32_t)nt_cols);")
            lines.append("    PipeBarrier<PIPE_V>();")

        lines += [
            "    int32_t nt_len = (int32_t)nt_cols;",
            "    while (nt_len >= 128) {",
            "        int32_t nt_h = nt_len / 2;",
            "        if (nt_h % 8 != 0) { nt_h &= ~7; }",
            "        if (nt_h < 64) { break; }",
            f"        {tree_op}(nt_w, nt_w, nt_w[nt_h], (int32_t)nt_h);",
            "        PipeBarrier<PIPE_V>();",
            "        nt_len = nt_h;",
            "    }",
        ]

        if operator in {"sum", "sum_sq"}:
            lines += [
                "    float nt_result = 0.0f;",
                "    for (int32_t nt_j = 0; nt_j < nt_len; nt_j++) {",
                "        nt_result += nt_w.GetValue((uint32_t)nt_j);",
                "    }",
            ]
        elif operator == "max":
            lines += [
                "    float nt_result = nt_w.GetValue(0);",
                "    for (int32_t nt_j = 1; nt_j < nt_len; nt_j++) {",
                "        if (nt_w.GetValue((uint32_t)nt_j) > nt_result) {",
                "            nt_result = nt_w.GetValue((uint32_t)nt_j);",
                "        }",
                "    }",
            ]
        else:
            lines += [
                "    float nt_result = nt_w.GetValue(0);",
                "    for (int32_t nt_j = 1; nt_j < nt_len; nt_j++) {",
                "        if (nt_w.GetValue((uint32_t)nt_j) < nt_result) {",
                "            nt_result = nt_w.GetValue((uint32_t)nt_j);",
                "        }",
                "    }",
            ]

        lines.append("    PipeBarrier<PIPE_V>();")

        if scale:
            lines.append("    nt_result = nt_result" + scale + ";")

        if operator == "sum_sq":
            lines.append("    nt_result = sqrt(nt_result);")

        lines += [
            "    auto nt_event = nt_pipe.FetchEventID(HardEvent::S_V);",
            "    SetFlag<HardEvent::S_V>(nt_event);",
            "    WaitFlag<HardEvent::S_V>(nt_event);",
            "    LocalTensor<float> nt_out = nt_qo.AllocTensor<float>();",
            "    Duplicate(nt_out, nt_result, (int32_t)nt_cols);",
            "    PipeBarrier<PIPE_V>();",
            "    nt_qo.EnQue(nt_out);",
            "    LocalTensor<float> nt_ov = nt_qo.DeQue<float>();",
            f"    DataCopy(nt_gm_{output}[(int64_t)GetBlockIdx() * nt_cols], nt_ov, (uint32_t)nt_cols);",
            "    nt_qi.FreeTensor<float>(nt_x);",
            "    nt_qo.FreeTensor<float>(nt_ov);",
        ]

    return n.join(lines)


__all__ = ["match_reduce_broadcast", "render_reduce_broadcast"]
