"""Vector reduce+broadcast lowering for the AscendC backend.

Row reductions over aligned 2-D inputs map onto UB staging with ping-pong
tree halving (Add/Max on TBuf intermediates).  Unaligned column counts
fall back to the scalar cooperative path because GM DataCopy addresses
must stay 32-byte aligned.
"""

from ninetoothed.backends.emitters import patterns
from ninetoothed.backends.emitters.ascendc_primitives import render_nan_propagation

_REDUCE_CHUNK = 8192


def match_reduce_broadcast(context):
    """Select an aligned row pattern proved by the shared SSA matcher."""
    match = patterns.match_reduce_broadcast(context, allow_direct=True)

    if match is None:
        return None

    try:
        cols = int(match.cols)
    except ValueError:
        return None
    return match if cols > 0 and cols % 8 == 0 and cols <= _REDUCE_CHUNK else None


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

    if operator in {"max", "min"}:
        lines[10:10] = [
            "    TBuf<TPosition::VECCALC> nt_nan_buf;",
            f"    nt_pipe.InitBuffer(nt_nan_buf, {chunk} / 8);",
            "    auto nt_nan_mask = nt_nan_buf.Get<uint8_t>();",
        ]

    if operator in {"sum", "sum_sq", "max", "min"}:
        if operator == "sum_sq":
            lines.append("    Mul(nt_w, nt_w, nt_w, (int32_t)nt_cols);")
            lines.append("    PipeBarrier<PIPE_V>();")

        lines += [
            "    int32_t nt_len = (int32_t)nt_cols;",
            "    while (nt_len >= 128) {",
            "        int32_t nt_h = nt_len / 2;",
            "        if (nt_h % 8 != 0 || (nt_len & 1)) { break; }",
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

        if operator in {"max", "min"}:
            lines.append(render_nan_propagation("nt_result", "nt_x", "nt_cols"))

        if operator == "sum_sq":
            lines.append("    nt_result = sqrt(nt_result);")

        if scale:
            lines.append("    nt_result = nt_result" + scale + ";")

        lines += [
            "    auto nt_event = nt_pipe.FetchEventID(HardEvent::S_V);",
            "    SetFlag<HardEvent::S_V>(nt_event);",
            "    WaitFlag<HardEvent::S_V>(nt_event);",
            "    LocalTensor<float> nt_out = nt_qo.AllocTensor<float>();",
            (
                "    Muls(nt_out, nt_x, 0.0f, (int32_t)nt_cols);\n"
                "    Adds(nt_out, nt_out, nt_result, (int32_t)nt_cols);"
                if match.broadcast_zero is not None
                else "    Duplicate(nt_out, nt_result, (int32_t)nt_cols);"
            ),
            "    PipeBarrier<PIPE_V>();",
            "    nt_qo.EnQue(nt_out);",
            "    LocalTensor<float> nt_ov = nt_qo.DeQue<float>();",
            f"    DataCopy(nt_gm_{output}[(int64_t)GetBlockIdx() * nt_cols], nt_ov, (uint32_t)nt_cols);",
            "    nt_qi.FreeTensor<float>(nt_x);",
            "    nt_qo.FreeTensor<float>(nt_ov);",
        ]

    return n.join(lines)


def match_row_reduce(context):
    """Select collapsed row stores with an explicit, safe output layout."""
    match = patterns.match_row_reduce(context, allow_direct=True)

    if match is None or match.broadcast_zero is not None:
        return None
    return match


def render_row_reduce(match):
    """Emit a UB vector row reduce; 16 rows per block own a 64B cache line."""
    operator, name, output, rows, cols, scale = match
    tree_op = "Max" if operator == "max" else "Add"

    if operator == "min":
        tree_op = "Min"

    if operator in {"sum", "sum_sq"}:
        accumulate = """        nt_result = 0.0f;
        for (int32_t nt_j = 0; nt_j < nt_len; nt_j++) {
            nt_result += nt_w.GetValue((uint32_t)nt_j);
        }"""
        gm_step = "nt_result += nt_v;"
    else:
        cmp = ">" if operator == "max" else "<"
        accumulate = f"""        nt_result = nt_w.GetValue(0);
        for (int32_t nt_j = 1; nt_j < nt_len; nt_j++) {{
            float nt_v = nt_w.GetValue((uint32_t)nt_j);
            if (nt_v {cmp} nt_result) {{ nt_result = nt_v; }}
        }}"""
        gm_step = f"if (nt_v != nt_v || nt_v {cmp} nt_result) {{ nt_result = nt_v; }}"

    square = ""

    if operator == "sum_sq":
        square = """        Mul(nt_w, nt_w, nt_w, (int32_t)nt_cols);
        PipeBarrier<PIPE_V>();"""

    post = []

    if operator == "sum_sq":
        post.append("        nt_result = sqrt(nt_result);")

    if scale:
        post.append("        nt_result = nt_result" + scale + ";")

    post_lines = chr(10).join(post)
    extrema = operator in {"max", "min"}
    nan_buffer = (
        f"""    TBuf<TPosition::VECCALC> nt_nan_buf;
    nt_pipe.InitBuffer(nt_nan_buf, {_REDUCE_CHUNK} / 8);
    auto nt_nan_mask = nt_nan_buf.Get<uint8_t>();"""
        if extrema
        else ""
    )
    nan_guard = (
        render_nan_propagation("nt_result", "nt_x", "nt_cols") if extrema else ""
    )
    free_before = "" if extrema else "            nt_qi.FreeTensor<float>(nt_x);"
    free_after = "            nt_qi.FreeTensor<float>(nt_x);" if extrema else ""

    return f"""    const int64_t nt_cols = {cols};
    const int64_t nt_row0 = (int64_t)GetBlockIdx() * 16;
    if (nt_row0 >= ({rows})) {{ return; }}
    const int32_t nt_nr = (int32_t)(((({rows}) - nt_row0) < 16) ? (({rows}) - nt_row0) : 16);
    GlobalTensor<float> nt_gm_{name};
    nt_gm_{name}.SetGlobalBuffer((__gm__ float*){name}_gm);
    GlobalTensor<float> nt_gm_{output};
    nt_gm_{output}.SetGlobalBuffer((__gm__ float*){output}_gm);
    TPipe nt_pipe;
    TQue<TPosition::VECIN, 1> nt_qi;
    TQue<TPosition::VECOUT, 1> nt_qo;
    TBuf<TPosition::VECCALC> nt_bw;
    nt_pipe.InitBuffer(nt_qi, 1, {_REDUCE_CHUNK} * sizeof(float));
    nt_pipe.InitBuffer(nt_qo, 1, 16 * sizeof(float));
    nt_pipe.InitBuffer(nt_bw, {_REDUCE_CHUNK} * sizeof(float));
{nan_buffer}
    LocalTensor<float> nt_w = nt_bw.Get<float>();
    float nt_vals[16];
    for (int32_t nt_r = 0; nt_r < nt_nr; nt_r++) {{
        float nt_result = 0.0f;
        if ((nt_cols % 8 == 0) && (nt_cols <= {_REDUCE_CHUNK})) {{
            LocalTensor<float> nt_x = nt_qi.AllocTensor<float>();
            DataCopy(nt_x, nt_gm_{name}[(nt_row0 + nt_r) * nt_cols], (uint32_t)nt_cols);
            nt_qi.EnQue(nt_x);
            nt_x = nt_qi.DeQue<float>();
            Adds(nt_w, nt_x, 0.0f, (int32_t)nt_cols);
            PipeBarrier<PIPE_V>();
{free_before}
{square}
            int32_t nt_len = (int32_t)nt_cols;
            while (nt_len >= 128) {{
                int32_t nt_h = nt_len / 2;
                if (nt_h % 8 != 0 || (nt_len & 1)) {{ break; }}
                if (nt_h < 64) {{ break; }}
                {tree_op}(nt_w, nt_w, nt_w[nt_h], (int32_t)nt_h);
                PipeBarrier<PIPE_V>();
                nt_len = nt_h;
            }}
{accumulate}
{nan_guard}
{free_after}
            PipeBarrier<PIPE_V>();
        }} else {{
            int64_t nt_base = (nt_row0 + nt_r) * nt_cols;
            for (int64_t nt_j = 0; nt_j < nt_cols; nt_j++) {{
                float nt_v = nt_gm_{name}[nt_base + nt_j].GetValue(0);
                {"nt_v *= nt_v;" if operator == "sum_sq" else ""}
                if (nt_j == 0) {{ nt_result = nt_v; }}
                else {{ {gm_step} }}
            }}
        }}
{post_lines}
        nt_vals[nt_r] = nt_result;
    }}
    auto nt_event = nt_pipe.FetchEventID(HardEvent::S_V);
    SetFlag<HardEvent::S_V>(nt_event);
    WaitFlag<HardEvent::S_V>(nt_event);
    LocalTensor<float> nt_o8 = nt_qo.AllocTensor<float>();
    for (int32_t nt_r = 0; nt_r < 16; nt_r++) {{
        nt_o8.SetValue((uint32_t)nt_r, (nt_r < nt_nr) ? nt_vals[nt_r] : 0.0f);
    }}
    SetFlag<HardEvent::S_V>(nt_event);
    WaitFlag<HardEvent::S_V>(nt_event);
    nt_qo.EnQue(nt_o8);
    LocalTensor<float> nt_ov8 = nt_qo.DeQue<float>();
    if (nt_nr == 16) {{
        DataCopy(nt_gm_{output}[nt_row0], nt_ov8, (uint32_t)16);
    }} else {{
        // The final output allocation contains only nt_nr valid elements.
        for (int32_t nt_r = 0; nt_r < nt_nr; nt_r++) {{
            nt_gm_{output}.SetValue((uint32_t)(nt_row0 + nt_r), nt_vals[nt_r]);
        }}
    }}
    nt_qo.FreeTensor<float>(nt_ov8);"""


__all__ = [
    "match_reduce_broadcast",
    "render_reduce_broadcast",
    "match_row_reduce",
    "render_row_reduce",
]
