"""AscendC scalar mathematical primitives and their actual dependencies."""

import re

from ninetoothed.backends.emitters.math_helpers import render_float32_trig_support

_ASCENDC_MATH_SUPPORT = (
    """// Scalar transcendental functions are not hardware built-ins on AscendC,
// so the generated kernels carry compact software implementations.  Device
// functions cannot recurse, so every routine is range-reduced inline.
union NtFloatBits {
    float f;
    int32_t i;
};

__aicore__ inline float nt_inf() {
    NtFloatBits v;
    v.i = 0x7f800000;
    return v.f;
}

__aicore__ inline float nt_nan() {
    NtFloatBits v;
    v.i = 0x7fc00000;
    return v.f;
}

__aicore__ inline float nt_exp(float x) {
    if (x != x) { return x; }
    if (x >= 88.72283935546875f) { return nt_inf(); }
    if (x < -104.0f) { return 0.0f; }
    float t = x * 1.442695041f;
    int k = (int)(t + (t > 0.0f ? 0.5f : -0.5f));
    float r = (float)k;
    x = (x - r * 0.693145751953125f) - r * 1.428606765330187e-6f;
    // 10th-order minimax polynomial for exp on [-ln2/2, ln2/2]
    float p = 1.0f + x * (1.0f + x * (0.5000000000f + x * (0.1666666667f
        + x * (0.04166666667f + x * (0.008333333333f + x * (0.001388888889f
        + x * (0.0001984126984f + x * (0.00002480158730f
        + x * (0.000002755731922f + x * 0.0000002755731922f)))))))));
    // Scale in two steps: a single 2^k exponent field overflows into the
    // sign bit for subnormal results (k + 127 <= 0) and produces a huge
    // negative value instead.  Halving the exponent keeps both scale
    // factors in range, and IEEE multiplication rounds gradually.
    int k1 = k / 2;
    int k2 = k - k1;
    NtFloatBits s1;
    NtFloatBits s2;
    s1.i = (k1 + 127) << 23;
    s2.i = (k2 + 127) << 23;
    return (p * s1.f) * s2.f;
}

__aicore__ inline float nt_log(float x) {
    if (x != x || x < 0.0f) { return nt_nan(); }
    if (x == 0.0f) { return -nt_inf(); }
    if (x == nt_inf()) { return x; }
    NtFloatBits v;
    v.f = x;
    int biased = (v.i >> 23) & 0xff;
    float m;
    int e;
    if (biased == 0) {
        // Subnormal inputs carry no implicit bit: x = M * 2^-149.
        e = -126;
        m = (float)(v.i & 0x007fffff) * (1.0f / 8388608.0f);
    }
    else {
        e = biased - 127;
        v.i = (v.i & 0x007fffff) | 0x3f800000;
        m = v.f;
    }
    // Normalize m into [0.75, 1.5] so |u| <= 0.2 below.
    while (m > 1.5f) { m = m * 0.5f; e = e + 1; }
    while (m < 0.75f) { m = m * 2.0f; e = e - 1; }
    // log(m) = 2*artanh(u) with u = (m-1)/(m+1): truncation ~ u^11/11.
    float u = (m - 1.0f) / (m + 1.0f);
    float u2 = u * u;
    float p = u * (1.0f + u2 * (0.33333334f + u2 * (0.2f + u2 * (0.14285715f
        + u2 * 0.11111111f))));
    return e * 0.69314718f + 2.0f * p;
}

__aicore__ inline float nt_tanh(float x) {
    if (x == 0.0f) { return x; }
    float ax = x < 0.0f ? -x : x;
    if (ax > 10.0f) { return x < 0.0f ? -1.0f : 1.0f; }
    float e = nt_expm1(2.0f * ax);
    float y = e / (e + 2.0f);
    return x < 0.0f ? -y : y;
}

__aicore__ inline float nt_sigmoid(float x) {
    return 1.0f / (1.0f + nt_exp(-x));
}

"""
    + render_float32_trig_support("__aicore__")
    + """

__aicore__ inline float nt_tan(float x) {
    return nt_sin(x) / nt_cos(x);
}

__aicore__ inline float nt_sinh(float x) {
    if (x == 0.0f) { return x; }
    float ax = x < 0.0f ? -x : x;
    float y;
    if (ax < 1.0f) {
        float e = nt_expm1(ax);
        y = 0.5f * (e + e / (1.0f + e));
    } else {
        float e = nt_exp(0.5f * ax);
        y = (0.5f * e) * e - (0.5f / e) / e;
    }
    return x < 0.0f ? -y : y;
}

__aicore__ inline float nt_cosh(float x) {
    float ax = x < 0.0f ? -x : x;
    float e = nt_exp(0.5f * ax);
    return (0.5f * e) * e + (0.5f / e) / e;
}

// atanh-free, recursion-free atan: fold x > 1 through the reciprocal
// identity atan(x) = pi/2 - atan(1/x), then arguments above 0.35 through
// the additive sqrt(3) identity, so the odd Taylor polynomial only sees
// |v| <= 2 - sqrt(3) ~ 0.268.
__aicore__ inline float nt_atan_poly(float x) {
    float v2 = x * x;
    return x * (1.0f - v2 * (0.33333334f - v2 * (0.2f - v2 * (0.14285715f
        - v2 * (0.11111111f - v2 * 0.09090909f)))));
}

__aicore__ inline float nt_atan_pos(float x) {
    const float NT_HALF_PI = 1.5707964f;
    const float NT_PI_6 = 0.5235988f;
    const float NT_SQRT3 = 1.7320508f;

    if (x > 1.0f) {
        float v = 1.0f / x;
        float inner = nt_atan_poly(v);

        if (v > 0.35f) {
            inner = nt_atan_poly((NT_SQRT3 * v - 1.0f) / (NT_SQRT3 + v))
                + NT_PI_6;
        }
        return NT_HALF_PI - inner;
    }

    if (x > 0.35f) {
        return nt_atan_poly((NT_SQRT3 * x - 1.0f) / (NT_SQRT3 + x)) + NT_PI_6;
    }

    return nt_atan_poly(x);
}

__aicore__ inline float nt_atan(float x) {
    if (x < 0.0f) { return -nt_atan_pos(-x); }
    return nt_atan_pos(x);
}

__aicore__ inline float nt_asin(float x) {
    if (x != x || x > 1.0f || x < -1.0f) { return nt_nan(); }
    if (x == 0.0f) { return x; }
    float r = 1.0f - x * x;
    float root = r >= 0.0f ? sqrt(r) : 0.0f;
    float v = x >= 0.0f ? (root > 0.0f ? x / root : nt_inf()) : -1.0f;
    if (x < 0.0f) {
        v = root > 0.0f ? x / root : -nt_inf();
    }
    return nt_atan(v);
}

__aicore__ inline float nt_acos(float x) {
    return 1.5707964f - nt_asin(x);
}

__aicore__ inline float nt_atan2(float y, float x) {
    const float NT_PI_F = 3.14159265f;
    const float NT_HALF_PI = 1.5707964f;
    if (x != x || y != y) { return nt_nan(); }
    NtFloatBits bx;
    NtFloatBits by;
    bx.f = x;
    by.f = y;
    bool negative_x = bx.i < 0;
    bool negative_y = by.i < 0;
    if (y == 0.0f) {
        return negative_x ? (negative_y ? -NT_PI_F : NT_PI_F) : y;
    }
    if (x == 0.0f) { return negative_y ? -NT_HALF_PI : NT_HALF_PI; }
    bx.i &= 0x7fffffff;
    by.i &= 0x7fffffff;
    float angle;
    if (by.f == nt_inf()) {
        angle = bx.f == nt_inf()
            ? (negative_x ? 2.35619449f : 0.785398163f) : NT_HALF_PI;
    }
    else if (bx.f == nt_inf()) {
        angle = negative_x ? NT_PI_F : 0.0f;
    }
    else {
        angle = nt_atan_pos(by.f / bx.f);
        if (negative_x) { angle = NT_PI_F - angle; }
    }
    return negative_y ? -angle : angle;
}

__aicore__ inline float nt_floor(float x) {
    NtFloatBits v;
    v.f = x;
    int exponent = ((v.i >> 23) & 0xff) - 127;
    // Large finite float32 values already have no fractional bits.
    if (exponent >= 23 || (v.i & 0x7fffffff) == 0) { return x; }
    if (exponent < 0) { return x < 0.0f ? -1.0f : 0.0f; }
    int32_t fraction = (1 << (23 - exponent)) - 1;
    if ((v.i & fraction) == 0) { return x; }
    v.i &= ~fraction;
    return x < 0.0f ? v.f - 1.0f : v.f;
}

__aicore__ inline float nt_ceil(float x) {
    return -nt_floor(-x);
}

__aicore__ inline float nt_pow(float x, float y) {
    if (y == 0.0f) { return 1.0f; }
    if (x == 1.0f) { return 1.0f; }
    if (x != x || y != y) { return nt_nan(); }
    NtFloatBits base_bits;
    NtFloatBits exponent_bits;
    base_bits.f = x;
    exponent_bits.f = y;
    int exponent = ((exponent_bits.i >> 23) & 0xff) - 127;
    // Above 2^24 float32 exponents are all even; inspect the significand
    // without a large float-to-int conversion.
    bool odd = exponent >= 0 && exponent <= 23 && y == nt_floor(y)
        && ((((exponent_bits.i & 0x007fffff) | 0x00800000)
            >> (23 - exponent)) & 1);
    float ax = x < 0.0f ? -x : x;
    if (y == nt_inf() || y == -nt_inf()) {
        if (ax == 1.0f) { return 1.0f; }
        return ((ax > 1.0f) == (y > 0.0f)) ? nt_inf() : 0.0f;
    }
    if (ax == 0.0f || ax == nt_inf()) {
        float special = ((ax == 0.0f) == (y > 0.0f)) ? 0.0f : nt_inf();
        return base_bits.i < 0 && odd ? -special : special;
    }
    if (x < 0.0f && y != nt_floor(y)) {
        // Negative base with a non-integer exponent is undefined.
        return nt_nan();
    }
    NtFloatBits v;
    v.f = (x < 0.0f) ? -x : x;
    int biased = (v.i >> 23) & 0xff;
    float m;
    int e;
    if (biased == 0) {
        // Subnormal bases carry no implicit bit: x = M * 2^-149.
        e = -126;
        m = (float)(v.i & 0x007fffff) * (1.0f / 8388608.0f);
    }
    else {
        e = biased - 127;
        v.i = (v.i & 0x007fffff) | 0x3f800000;
        m = v.f;
    }
    float lg = e * 0.69314718f + nt_log(m);
    float r = nt_exp(((x < 0.0f) ? nt_floor(y) : y) * lg);
    if (x < 0.0f) {
        return odd ? -r : r;
    }
    return r;
}

__aicore__ inline float nt_fabs(float x) {
    NtFloatBits v;
    v.f = x;
    v.i &= 0x7fffffff;
    return v.f;
}

template <class T>
__aicore__ inline typename std::enable_if<std::is_integral<T>::value, T>::type nt_fabs(T x) {
    // Match two's-complement tensor abs at the signed minimum without -MIN UB.
    return x == std::numeric_limits<T>::lowest() ? x : (x < 0 ? -x : x);
}

__aicore__ inline float nt_rsqrt(float x) {
    return 1.0f / sqrt(x);
}

__aicore__ inline float nt_exp2(float x) {
    return nt_exp(x * 0.69314718f);
}

__aicore__ inline float nt_log2(float x) {
    return nt_log(x) * 1.44269504f;
}

__aicore__ inline float nt_log10(float x) {
    return nt_log(x) * 0.43429448f;
}

__aicore__ inline float nt_log1p(float x) {
    if (x == 0.0f || x == nt_inf()) { return x; }
    if (x < -1.0f || x != x) { return nt_nan(); }
    if (x == -1.0f) { return -nt_inf(); }
    if (x >= -0.5f && x <= 0.5f) {
        float u = x / (2.0f + x);
        float u2 = u * u;
        return 2.0f * u * (1.0f + u2 * (1.0f/3.0f + u2 * (1.0f/5.0f
            + u2 * (1.0f/7.0f + u2 * (1.0f/9.0f + u2 * (1.0f/11.0f
            + u2 * (1.0f/13.0f + u2 * (1.0f/15.0f + u2 / 17.0f))))))));
    }
    float y = 1.0f + x;
    return nt_log(y) - ((y - 1.0f) - x) / y;
}

__aicore__ inline float nt_expm1(float x) {
    if (x == 0.0f) { return x; }
    if (x >= -0.5f && x <= 0.5f) {
        return x * (1.0f + x * (0.5f + x * (1.0f/6.0f + x * (1.0f/24.0f
            + x * (1.0f/120.0f + x * (1.0f/720.0f + x * (1.0f/5040.0f
            + x * (1.0f/40320.0f + x * (1.0f/362880.0f
            + x * (1.0f/3628800.0f + x / 39916800.0f))))))))));
    }
    return nt_exp(x) - 1.0f;
}

__aicore__ inline float nt_erf(float x) {
    if (x == 0.0f) { return x; }
    float s = x < 0.0f ? -1.0f : 1.0f;
    x = x * s;
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f)
        * t - 0.284496736f) * t + 0.254829592f) * t * nt_exp(-x * x);
    return s * y;
}"""
)


# Split definitions once; emit only the transitive helpers referenced by the body.
_STARTS = tuple(
    re.finditer(
        r"(?m)^(?:template[^\n]*\n)?__aicore__ inline [^\n]*?\b(nt_\w+)\(",
        _ASCENDC_MATH_SUPPORT,
    )
)
_DEFINITIONS = {}

for _i, _match in enumerate(_STARTS):
    _end = (
        _STARTS[_i + 1].start() if _i + 1 < len(_STARTS) else len(_ASCENDC_MATH_SUPPORT)
    )
    _DEFINITIONS.setdefault(_match[1], []).append(
        _ASCENDC_MATH_SUPPORT[_match.start() : _end]
    )


def render_math_support(body):
    needed = set()
    pieces = []

    def include(name):
        if name in needed or name not in _DEFINITIONS:
            return

        needed.add(name)
        source = "\n".join(_DEFINITIONS[name])

        for dependency in re.findall(r"\b(nt_\w+)\s*\(", source):
            include(dependency)

        pieces.append(source)

    for name in re.findall(r"\b(nt_\w+)\s*\(", body):
        include(name)

    prefix = _ASCENDC_MATH_SUPPORT[: _STARTS[0].start()] if pieces else ""

    return prefix + "\n".join(pieces)
