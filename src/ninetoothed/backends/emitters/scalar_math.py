"""IEEE scalar operations missing from vendor device standard libraries."""


def render_scalar_math_support(qualifier):
    return _SUPPORT.replace("NT_DEVICE", qualifier)


_SUPPORT = """union NtScalarBits { float f; int32_t i; };
static NT_DEVICE float nt_copysign(float x, float y) {
    NtScalarBits a, b; a.f = x; b.f = y;
    a.i = (a.i & 0x7fffffff) | (b.i & (-2147483647 - 1));
    return a.f;
}
static NT_DEVICE bool nt_isnan(float x) { return x != x; }
static NT_DEVICE bool nt_isfinite(float x) {
    NtScalarBits b; b.f = x; return (b.i & 0x7f800000) != 0x7f800000;
}
static NT_DEVICE float nt_fmod(float x, float y) {
    NtScalarBits a, b; a.f = x; b.f = y;
    int sign = a.i & (-2147483647 - 1);
    a.i &= 0x7fffffff; b.i &= 0x7fffffff;
    if (b.i == 0 || a.i >= 0x7f800000 || b.i > 0x7f800000) {
        a.i = 0x7fc00000; return a.f;
    }
    if (a.i < b.i) { return x; }
    float remainder = a.f, divisor = b.f;
    // Binary long division: all subtractions are exact by Sterbenz's lemma.
    while (divisor <= remainder * 0.5f) { divisor *= 2.0f; }
    while (divisor >= b.f) {
        if (remainder >= divisor) { remainder -= divisor; }
        divisor *= 0.5f;
    }
    a.f = remainder; a.i |= sign; return a.f;
}
static NT_DEVICE float nt_nearbyint(float x) {
    NtScalarBits b; b.f = x;
    if ((b.i & 0x7fffffff) >= 0x4b000000) { return x; }
    int32_t n = (int32_t)x;
    float delta = x - (float)n;
    float magnitude = delta < 0.0f ? -delta : delta;
    if (magnitude > 0.5f || (magnitude == 0.5f && (n & 1))) {
        n += x < 0.0f ? -1 : 1;
    }
    return n == 0 ? nt_copysign(0.0f, x) : (float)n;
}
template <class A, class B>
static NT_DEVICE auto _nt_modulo(A x, B y)
    -> typename std::enable_if<std::is_integral<A>::value && std::is_integral<B>::value, decltype(x % y)>::type {
    auto r = x % y;
    return r != 0 && ((r < 0) != (y < 0)) ? r + y : r;
}
template <class A, class B>
static NT_DEVICE typename std::enable_if<!std::is_integral<A>::value || !std::is_integral<B>::value, float>::type
_nt_modulo(A x, B y) {
    float r = nt_fmod((float)x, (float)y);
    return r != 0.0f && ((r < 0.0f) != (y < 0)) ? r + (float)y : r;
}
"""
