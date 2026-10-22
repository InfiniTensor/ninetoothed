"""Strict Ascend vector primitives shared by packet and row storage domains."""

import math
from collections.abc import Mapping


def numeric_lower_bound(facts, name):
    """Accept the shared float32 proof schema, without inventing input facts."""
    if not isinstance(facts, Mapping):
        return None

    fact = facts.get(name)

    if not isinstance(fact, Mapping):
        return None

    lower = fact.get("lower")

    if type(lower) is not float or not math.isfinite(lower):
        return None
    return lower


def _render_mask_summary_vector(positive):
    rhs = "nt_ref_mask0" if positive else "nt_ref_mask1"

    return f"""        bool nt_ref_all_valid = false;
        if (nt_ref_prefix >= 2048) {{
            // Exact integer packed-validity fold; original repair masks remain live.
            int32_t nt_ref_words = nt_ref_prefix / 16;
            int32_t nt_ref_capacity = 16;
            while (nt_ref_capacity < nt_ref_words) {{ nt_ref_capacity <<= 1; }}
            LocalTensor<int16_t> nt_ref_gate = nt_ref0.ReinterpretCast<int16_t>();
            PipeBarrier<PIPE_V>();
            Duplicate(nt_ref_gate, (int16_t)-1, nt_ref_capacity);
            PipeBarrier<PIPE_V>();
            And(nt_ref_gate, nt_ref_mask0.ReinterpretCast<int16_t>(), {rhs}.ReinterpretCast<int16_t>(), nt_ref_words);
            PipeBarrier<PIPE_V>();
            for (int32_t nt_ref_half = nt_ref_capacity / 2; nt_ref_half >= 16; nt_ref_half /= 2) {{
                And(nt_ref_gate, nt_ref_gate, nt_ref_gate[nt_ref_half], nt_ref_half);
                PipeBarrier<PIPE_V>();
            }}
        }}"""


def _render_mask_summary_scalar():
    return """        if (nt_ref_prefix >= 2048) {
            LocalTensor<uint64_t> nt_ref_gate_bits = nt_ref0.ReinterpretCast<uint64_t>();
            nt_ref_all_valid = (nt_ref_gate_bits.GetValue(0) & nt_ref_gate_bits.GetValue(1) &
                               nt_ref_gate_bits.GetValue(2) & nt_ref_gate_bits.GetValue(3)) == ~0ULL;
        }"""


def _render_fold_exception_repair(dst, src, count, scalar, *, positive, magnitude=""):
    if positive:
        compare = f"CompareScalar(nt_ref_mask0, {src}, 0x1p120f, CMPMODE::LE, (uint32_t)nt_ref_prefix);"
        bits = "LocalTensor<uint64_t> nt_ref_bits0 = nt_ref_mask0.ReinterpretCast<uint64_t>();"
        valid = "nt_ref_bits0.GetValue(nt_word)"
        tail = f"float nt_value = {src}.GetValue(nt_j);\n            if (nt_value <= 0x1p120f) {{ continue; }}"
    else:
        compare = """CompareScalar(nt_ref_mask0, nt_ref0, 0x1p-120f, CMPMODE::GE, (uint32_t)nt_ref_prefix);
            CompareScalar(nt_ref_mask1, nt_ref0, 0x1p120f, CMPMODE::LE, (uint32_t)nt_ref_prefix);"""
        bits = """LocalTensor<uint64_t> nt_ref_bits0 = nt_ref_mask0.ReinterpretCast<uint64_t>();
        LocalTensor<uint64_t> nt_ref_bits1 = nt_ref_mask1.ReinterpretCast<uint64_t>();"""
        valid = "nt_ref_bits0.GetValue(nt_word) & nt_ref_bits1.GetValue(nt_word)"
        tail = f"""float nt_magnitude = nt_ref0.GetValue(nt_j);
            if (nt_magnitude >= 0x1p-120f && nt_magnitude <= 0x1p120f) {{ continue; }}
            float nt_value = {src}.GetValue(nt_j);"""

    intro = (
        "// SSA-proved positive float32 lower bound; retain NaN/Inf repair."
        if positive
        else magnitude + "\n        PipeBarrier<PIPE_V>();"
    )

    return f"""        {{
        {intro}
        const int32_t nt_ref_prefix = (int32_t){count} & ~127;
        if (nt_ref_prefix > 0) {{
            {compare}
        }}
{_render_mask_summary_vector(positive)}
        auto nt_ref_event = nt_pipe.FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(nt_ref_event);
        WaitFlag<HardEvent::V_S>(nt_ref_event);
{_render_mask_summary_scalar()}
        {bits}
        if (!nt_ref_all_valid) {{
            for (int32_t nt_word = 0; nt_word * 64 < nt_ref_prefix; nt_word++) {{
                uint64_t nt_valid = {valid};
                if (nt_valid == ~0ULL) {{ continue; }}
                for (int32_t nt_lane = 0; nt_lane < 64; nt_lane++) {{
                    if ((nt_valid >> nt_lane) & 1ULL) {{ continue; }}
                    int32_t nt_j = nt_word * 64 + nt_lane;
                    float nt_value = {src}.GetValue(nt_j);
                    {dst}.SetValue(nt_j, {scalar});
                }}
            }}
        }}
        // The float scratch tail is outside the integer prefix workspace.
        for (int32_t nt_j = nt_ref_prefix; nt_j < (int32_t){count}; nt_j++) {{
            {tail}
            {dst}.SetValue(nt_j, {scalar});
        }}
        auto nt_ref_store = nt_pipe.FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(nt_ref_store);
        WaitFlag<HardEvent::S_V>(nt_ref_store);
        }}"""


def render_refinement(
    intrinsic,
    dst,
    src,
    count,
    *,
    known_lower_bound=None,
    validity_summary=False,
    native_division=False,
    operands_preserved=False,
):
    """Refine normal lanes twice and repair exceptional lanes with scalar math.

    Packed validity masks make the ordinary path inspect one scalar word for
    64 lanes.  Bounds exclude hardware denormal/overflow cases; NaN, infinity,
    zero and extreme exponents retain the scalar backend's semantics.
    The caller owns one float buffer and two packed-mask buffers and must
    not alias them with any live source or destination.
    """
    if native_division and operands_preserved and intrinsic == "ReciprocalStrict":
        repair = _render_exception_repair(
            dst,
            src,
            count,
            f"Abs(nt_ref0, {src}, (int32_t){count});",
            "1.0f / nt_value",
            known_lower_bound=known_lower_bound,
            validity_summary=validity_summary,
        )

        return f"""        {{
        // Native Div with original IEEE sparse repair; typed UB operands survive repair.
        Duplicate(nt_ref0, 1.0f, (int32_t){count});
        PipeBarrier<PIPE_V>();
        Div({dst}, nt_ref0, {src}, (int32_t){count});
        PipeBarrier<PIPE_V>();
        {repair}
        }}"""

    base = "Reciprocal" if intrinsic == "ReciprocalStrict" else "Rsqrt"
    scalar = "1.0f / nt_value" if base == "Reciprocal" else "nt_rsqrt(nt_value)"
    magnitude = (
        f"Abs(nt_ref0, {src}, (int32_t){count});"
        if base == "Reciprocal"
        else f"Adds(nt_ref0, {src}, 0.0f, (int32_t){count});"
    )
    multiply = (
        f"Mul(nt_ref0, nt_ref0, {dst}, (int32_t){count});\n"
        "            PipeBarrier<PIPE_V>();\n"
        f"            Muls(nt_ref0, nt_ref0, -0.5f, (int32_t){count});\n"
        "            PipeBarrier<PIPE_V>();\n"
        f"            Adds(nt_ref0, nt_ref0, 1.5f, (int32_t){count});"
        if base == "Rsqrt"
        else f"Muls(nt_ref0, nt_ref0, -1.0f, (int32_t){count});\n"
        "            PipeBarrier<PIPE_V>();\n"
        f"            Adds(nt_ref0, nt_ref0, 2.0f, (int32_t){count});"
    )

    return f"""        {{
        {base}({dst}, {src}, (int32_t){count});
        PipeBarrier<PIPE_V>();
        for (int nt_iteration = 0; nt_iteration < 2; nt_iteration++) {{
            Mul(nt_ref0, {dst}, {src}, (int32_t){count});
            PipeBarrier<PIPE_V>();
            {multiply}
            PipeBarrier<PIPE_V>();
            Mul({dst}, {dst}, nt_ref0, (int32_t){count});
            PipeBarrier<PIPE_V>();
        }}
        {_render_exception_repair(dst, src, count, magnitude, scalar, known_lower_bound=known_lower_bound, validity_summary=validity_summary)}
        }}"""


def _render_exception_repair(
    dst,
    src,
    count,
    magnitude,
    scalar,
    *,
    known_lower_bound=None,
    validity_summary=False,
):
    if (
        type(known_lower_bound) is float
        and math.isfinite(known_lower_bound)
        and known_lower_bound >= float.fromhex("0x1p-120")
    ):
        return _render_positive_exception_repair(
            dst, src, count, scalar, validity_summary=validity_summary
        )

    if validity_summary:
        return _render_fold_exception_repair(
            dst, src, count, scalar, positive=False, magnitude=magnitude
        )

    return f"""        {{
        {magnitude}
        PipeBarrier<PIPE_V>();
        const int32_t nt_ref_prefix = (int32_t){count} & ~127;
        if (nt_ref_prefix > 0) {{
            CompareScalar(nt_ref_mask0, nt_ref0, 0x1p-120f, CMPMODE::GE, (uint32_t)nt_ref_prefix);
            CompareScalar(nt_ref_mask1, nt_ref0, 0x1p120f, CMPMODE::LE, (uint32_t)nt_ref_prefix);
        }}
        auto nt_ref_event = nt_pipe.FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(nt_ref_event);
        WaitFlag<HardEvent::V_S>(nt_ref_event);
        LocalTensor<uint64_t> nt_ref_bits0 = nt_ref_mask0.ReinterpretCast<uint64_t>();
        LocalTensor<uint64_t> nt_ref_bits1 = nt_ref_mask1.ReinterpretCast<uint64_t>();
        for (int32_t nt_word = 0; nt_word * 64 < nt_ref_prefix; nt_word++) {{
            uint64_t nt_valid = nt_ref_bits0.GetValue(nt_word) & nt_ref_bits1.GetValue(nt_word);
            int32_t nt_lanes = (int32_t){count} - nt_word * 64;
            if (nt_lanes > 64) {{ nt_lanes = 64; }}
            uint64_t nt_all = nt_lanes == 64 ? ~0ULL : ((1ULL << nt_lanes) - 1ULL);
            if ((nt_valid & nt_all) == nt_all) {{ continue; }}
            for (int32_t nt_lane = 0; nt_lane < nt_lanes; nt_lane++) {{
                if ((nt_valid >> nt_lane) & 1ULL) {{ continue; }}
                int32_t nt_j = nt_word * 64 + nt_lane;
                float nt_value = {src}.GetValue(nt_j);
                {dst}.SetValue(nt_j, {scalar});
            }}
        }}
        // C220 counter-mode comparisons truncate to complete repeats.
        // Never read unwritten packed bits for the remaining valid lanes.
        for (int32_t nt_j = nt_ref_prefix; nt_j < (int32_t){count}; nt_j++) {{
            float nt_magnitude = nt_ref0.GetValue(nt_j);
            if (nt_magnitude >= 0x1p-120f && nt_magnitude <= 0x1p120f) {{ continue; }}
            float nt_value = {src}.GetValue(nt_j);
            {dst}.SetValue(nt_j, {scalar});
        }}
        auto nt_ref_store = nt_pipe.FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(nt_ref_store);
        WaitFlag<HardEvent::S_V>(nt_ref_store);
        }}"""


def _render_positive_exception_repair(
    dst, src, count, scalar, *, validity_summary=False
):
    """Keep the upper-domain/NaN/Inf check after a proved positive bound."""
    if validity_summary:
        return _render_fold_exception_repair(dst, src, count, scalar, positive=True)

    return f"""        {{
        // SSA-proved positive float32 lower bound; retain NaN/Inf repair.
        const int32_t nt_ref_prefix = (int32_t){count} & ~127;
        if (nt_ref_prefix > 0) {{
            CompareScalar(nt_ref_mask0, {src}, 0x1p120f, CMPMODE::LE, (uint32_t)nt_ref_prefix);
        }}
        auto nt_ref_event = nt_pipe.FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(nt_ref_event);
        WaitFlag<HardEvent::V_S>(nt_ref_event);
        LocalTensor<uint64_t> nt_ref_bits0 = nt_ref_mask0.ReinterpretCast<uint64_t>();
        for (int32_t nt_word = 0; nt_word * 64 < nt_ref_prefix; nt_word++) {{
            uint64_t nt_valid = nt_ref_bits0.GetValue(nt_word);
            if (nt_valid == ~0ULL) {{ continue; }}
            for (int32_t nt_lane = 0; nt_lane < 64; nt_lane++) {{
                if ((nt_valid >> nt_lane) & 1ULL) {{ continue; }}
                int32_t nt_j = nt_word * 64 + nt_lane;
                float nt_value = {src}.GetValue(nt_j);
                {dst}.SetValue(nt_j, {scalar});
            }}
        }}
        for (int32_t nt_j = nt_ref_prefix; nt_j < (int32_t){count}; nt_j++) {{
            float nt_value = {src}.GetValue(nt_j);
            if (nt_value <= 0x1p120f) {{ continue; }}
            {dst}.SetValue(nt_j, {scalar});
        }}
        auto nt_ref_store = nt_pipe.FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(nt_ref_store);
        WaitFlag<HardEvent::S_V>(nt_ref_store);
        }}"""


def render_division(
    dst,
    lhs,
    rhs,
    count,
    *,
    scalar_lhs=False,
    known_rhs_lower_bound=None,
    validity_summary=False,
    native_division=False,
    operands_preserved=False,
):
    """Repair original divisions when reciprocal intermediates are unsafe."""
    scalar_left = lhs if scalar_lhs else f"{lhs}.GetValue(nt_j)"
    scalar = f"({scalar_left}) / {rhs}.GetValue(nt_j)"

    if native_division and operands_preserved:
        first = (
            f"Duplicate(nt_ref0, (float)({lhs}), (int32_t){count});\n        PipeBarrier<PIPE_V>();\n        Div({dst}, nt_ref0, {rhs}, (int32_t){count});"
            if scalar_lhs
            else f"Div({dst}, {lhs}, {rhs}, (int32_t){count});"
        )
        result = f"""        {{
        // Native Div with original IEEE sparse repair; typed UB operands survive repair.
        {first}
        PipeBarrier<PIPE_V>();
        """
        result += _render_exception_repair(
            dst,
            rhs,
            count,
            f"Abs(nt_ref0, {rhs}, (int32_t){count});",
            scalar,
            known_lower_bound=known_rhs_lower_bound,
            validity_summary=validity_summary,
        )

        if not scalar_lhs:
            result += _render_exception_repair(
                dst,
                lhs,
                count,
                f"Abs(nt_ref0, {lhs}, (int32_t){count});",
                scalar,
                validity_summary=validity_summary,
            )
        return result + "\n        }"

    intrinsic = "Muls" if scalar_lhs else "Mul"
    result = render_refinement(
        "ReciprocalStrict",
        dst,
        rhs,
        count,
        known_lower_bound=known_rhs_lower_bound,
        validity_summary=validity_summary,
    )
    result += f"\n        {intrinsic}({dst}, {dst}, {lhs}, (int32_t){count});\n        PipeBarrier<PIPE_V>();\n"
    result += _render_exception_repair(
        dst,
        rhs,
        count,
        f"Abs(nt_ref0, {rhs}, (int32_t){count});",
        scalar,
        known_lower_bound=known_rhs_lower_bound,
        validity_summary=validity_summary,
    )

    if not scalar_lhs:
        result += _render_exception_repair(
            dst,
            lhs,
            count,
            f"Abs(nt_ref0, {lhs}, (int32_t){count});",
            scalar,
            validity_summary=validity_summary,
        )
    return result


def render_unary_zero_repair(operator, dst, source, count):
    """Keep SDK nonzero results and restore the sign of exact zero outputs."""
    tested = source if operator == "Tanh" else dst
    repair = (
        f"{dst}.SetValue(nt_lane, {source}.GetValue(nt_lane));"
        if operator == "Tanh"
        else f"{dst}.ReinterpretCast<uint32_t>().SetValue(nt_lane, "
        f"{source}.ReinterpretCast<uint32_t>().GetValue(nt_lane) & 0x80000000U);"
    )

    tail_repair = (
        f"{dst}.SetValue(nt_lane, nt_zero_value);" if operator == "Tanh" else repair
    )

    return f"""        {{ // Preserve signed zero after the SDK {operator} primitive.
        {operator}({dst}, {source}, (int32_t){count});
        PipeBarrier<PIPE_ALL>();
        const int32_t nt_zero_prefix = (({count}) / 128) * 128;
        if (nt_zero_prefix != 0) {{
            CompareScalar(nt_ref_mask0, {tested}, 0.0f, CMPMODE::EQ, nt_zero_prefix);
            PipeBarrier<PIPE_ALL>();
            auto nt_zero_words = nt_ref_mask0.ReinterpretCast<uint32_t>();
            for (int32_t nt_word = 0; nt_word < nt_zero_prefix / 32; nt_word++) {{
                uint32_t nt_pending = nt_zero_words.GetValue(nt_word);
                if (nt_pending == 0) {{ continue; }}
                for (int32_t nt_bit = 0; nt_bit < 32; nt_bit++) {{
                    if ((nt_pending & (1U << nt_bit)) != 0) {{
                        int32_t nt_lane = nt_word * 32 + nt_bit;
                        {repair}
                    }}
                }}
            }}
        }}
        for (int32_t nt_lane = nt_zero_prefix; nt_lane < ({count}); nt_lane++) {{
            float nt_zero_value = {tested}.GetValue(nt_lane);
            if (nt_zero_value == 0.0f) {{ {tail_repair} }}
        }}
        PipeBarrier<PIPE_ALL>();
        }}"""


def render_extrema(
    operator,
    dst,
    lhs,
    rhs,
    count,
    *,
    scalar_lhs=False,
    scalar_rhs=False,
    swap_inputs=False,
):
    """Vector extrema preserve ordered NaNs and right-operand signed-zero ties.

    Ordinary words avoid per-lane scalar work. Unordered or zero ties and the
    partial compare-mask tail consume the original scalar expression. The
    caller supplies three packed-mask buffers, without a uniform float buffer.
    """
    intrinsic = "Max" if operator == "MaxStrict" else "Min"
    comparison = ">" if operator == "MaxStrict" else "<"
    lhs_value = f"(float)({lhs})" if scalar_lhs else f"{lhs}.GetValue(nt_lane)"
    rhs_value = f"(float)({rhs})" if scalar_rhs else f"{rhs}.GetValue(nt_lane)"

    if scalar_lhs or scalar_rhs:
        scalar, vector = (lhs, rhs) if scalar_lhs else (rhs, lhs)
        operation = (
            f"{intrinsic}s({dst}, {vector}, (float)({scalar}), (int32_t)({count}));"
        )
        masks = f"""CompareScalar(nt_ref_mask0, {vector}, nt_inf(), CMPMODE::LE, (uint32_t)nt_ext_prefix);
        CompareScalar(nt_ref_mask1, {vector}, 0.0f, CMPMODE::NE, (uint32_t)nt_ext_prefix);
        PipeBarrier<PIPE_ALL>();
        And(nt_ref_mask0.ReinterpretCast<uint16_t>(), nt_ref_mask0.ReinterpretCast<uint16_t>(), nt_ref_mask1.ReinterpretCast<uint16_t>(), nt_ext_prefix / 16);"""
        pending = f"((float)({scalar}) != (float)({scalar}) ? ~0ULL : ~nt_ext_lmask.GetValue(nt_word))"
    else:
        sources = (rhs, lhs) if swap_inputs else (lhs, rhs)
        operation = (
            f"{intrinsic}({dst}, {sources[0]}, {sources[1]}, (int32_t)({count}));"
        )
        masks = f"""CompareScalar(nt_ref_mask0, {lhs}, nt_inf(), CMPMODE::LE, (uint32_t)nt_ext_prefix);
        CompareScalar(nt_ref_mask1, {rhs}, nt_inf(), CMPMODE::LE, (uint32_t)nt_ext_prefix);
        PipeBarrier<PIPE_ALL>();
        And(nt_ref_mask0.ReinterpretCast<uint16_t>(), nt_ref_mask0.ReinterpretCast<uint16_t>(), nt_ref_mask1.ReinterpretCast<uint16_t>(), nt_ext_prefix / 16);
        PipeBarrier<PIPE_ALL>();
        CompareScalar(nt_ref_mask1, {lhs}, 0.0f, CMPMODE::EQ, (uint32_t)nt_ext_prefix);
        CompareScalar(nt_ref_mask2, {rhs}, 0.0f, CMPMODE::EQ, (uint32_t)nt_ext_prefix);
        PipeBarrier<PIPE_ALL>();
        And(nt_ref_mask1.ReinterpretCast<uint16_t>(), nt_ref_mask1.ReinterpretCast<uint16_t>(), nt_ref_mask2.ReinterpretCast<uint16_t>(), nt_ext_prefix / 16);"""
        pending = "(~nt_ext_lmask.GetValue(nt_word) | nt_ext_rmask.GetValue(nt_word))"

    return f"""{{
    // Strict vector extrema: sparse NaN and signed-zero repair.
    {operation}
    PipeBarrier<PIPE_ALL>();
    int32_t nt_ext_prefix = (int32_t)({count}) / 128 * 128;
    auto nt_ext_lmask = nt_ref_mask0.ReinterpretCast<uint64_t>();
    auto nt_ext_rmask = nt_ref_mask1.ReinterpretCast<uint64_t>();
    if (nt_ext_prefix != 0) {{
        {masks}
        PipeBarrier<PIPE_ALL>();
        for (int32_t nt_word = 0; nt_word * 64 < nt_ext_prefix; nt_word++) {{
            uint64_t nt_fix = {pending};
            if (nt_fix != 0ULL) {{
                for (int32_t nt_bit = 0; nt_bit < 64; nt_bit++) {{
                    if ((nt_fix >> nt_bit) & 1ULL) {{
                        int32_t nt_lane = nt_word * 64 + nt_bit;
                        float nt_lhs = {lhs_value};
                        float nt_rhs = {rhs_value};
                        {dst}.SetValue(nt_lane, nt_lhs != nt_lhs ? nt_lhs : (nt_rhs != nt_rhs ? nt_rhs : (nt_lhs {comparison} nt_rhs ? nt_lhs : nt_rhs)));
                    }}
                }}
            }}
        }}
    }}
    for (int32_t nt_lane = nt_ext_prefix; nt_lane < ({count}); nt_lane++) {{
        float nt_lhs = {lhs_value};
        float nt_rhs = {rhs_value};
        {dst}.SetValue(nt_lane, nt_lhs != nt_lhs ? nt_lhs : (nt_rhs != nt_rhs ? nt_rhs : (nt_lhs {comparison} nt_rhs ? nt_lhs : nt_rhs)));
    }}
    PipeBarrier<PIPE_ALL>();
}}"""


def render_nan_propagation(result, source, count, mask="nt_nan_mask"):
    """Preserve extrema NaNs without reading unwritten compare-mask tails."""
    return f"""{{
    int32_t nt_nan_prefix = (int32_t)({count}) / 128 * 128;
    if (nt_nan_prefix != 0) {{
        CompareScalar({mask}, {source}, nt_inf(), CMPMODE::LE, (uint32_t)nt_nan_prefix);
    }}
    PipeBarrier<PIPE_ALL>();
    auto nt_nan_bits = {mask}.ReinterpretCast<uint64_t>();
    for (int32_t nt_word = 0; nt_word * 64 < nt_nan_prefix; nt_word++) {{
        if (nt_nan_bits.GetValue(nt_word) != ~0ULL) {{ {result} = nt_nan(); break; }}
    }}
    for (int32_t nt_j = nt_nan_prefix; nt_j < ({count}); nt_j++) {{
        float nt_tail_value = {source}.GetValue(nt_j);
        if (nt_tail_value != nt_tail_value) {{ {result} = nt_nan(); break; }}
    }}
}}"""
