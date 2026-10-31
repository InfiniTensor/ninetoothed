"""Exact software integer data; MLU scalar address arithmetic is only 48 bit."""


def render_int64_support():
    mixed = []

    for operator in (
        "+",
        "-",
        "*",
        "/",
        "%",
        "&",
        "|",
        "^",
        "<<",
        ">>",
        "==",
        "!=",
        "<",
        ">",
        "<=",
        ">=",
    ):
        result = (
            "bool" if operator in ("==", "!=", "<", ">", "<=", ">=") else "NtWide<S>"
        )
        mixed.append(f"""template <class T, bool S>
static __mlu_func__ typename std::enable_if<std::is_integral<T>::value, {result}>::type
operator{operator}(T lhs, NtWide<S> rhs) {{ return NtWide<S>((int64_t)lhs) {operator} rhs; }}""")
    return _SUPPORT + "\n" + "\n".join(mixed)


_SUPPORT = r"""template <bool Signed> struct NtWide;
template <class F, bool S> static __mlu_func__ F nt_wide_float(NtWide<S> value);

template <bool Signed> struct NtWide {
    uint32_t lo, hi;
    NtWide() = default;
    template <bool Other> __mlu_func__ NtWide(NtWide<Other> x) : lo(x.lo), hi(x.hi) {}
    __mlu_func__ NtWide(uint32_t low, uint32_t high) : lo(low), hi(high) {}
    __mlu_func__ NtWide(int32_t x) : lo((uint32_t)x), hi(x < 0 ? 0xffffffffu : 0) {}
    __mlu_func__ NtWide(uint32_t x) : lo(x), hi(0) {}
    __mlu_func__ NtWide(int64_t x) : lo((uint32_t)x), hi((uint32_t)(x >> 32)) {
        if (x < 0) { hi |= 0xffff0000u; }
    }
    __mlu_func__ NtWide(float x) { assign_float(x); }
    __mlu_func__ NtWide(double x) { assign_float(x); }
    template <class F> __mlu_func__ void assign_float(F x) {
        bool negative = x < F(0);
        if (negative) { x = -x; }
        hi = (uint32_t)(x / F(4294967296.0));
        lo = (uint32_t)(x - F(hi) * F(4294967296.0));
        if (negative) { lo = ~lo + 1u; hi = ~hi + (lo == 0); }
    }
    explicit __mlu_func__ operator bool() const { return (lo | hi) != 0; }
    explicit __mlu_func__ operator int8_t() const { return (int8_t)lo; }
    explicit __mlu_func__ operator uint8_t() const { return (uint8_t)lo; }
    explicit __mlu_func__ operator int16_t() const { return (int16_t)lo; }
    explicit __mlu_func__ operator uint16_t() const { return (uint16_t)lo; }
    explicit __mlu_func__ operator int32_t() const { return (int32_t)lo; }
    explicit __mlu_func__ operator uint32_t() const { return lo; }
    explicit __mlu_func__ operator int64_t() const {
        return ((int64_t)(int32_t)hi << 32) | lo;
    }
    explicit __mlu_func__ operator uint64_t() const {
        return ((uint64_t)hi << 32) | lo;
    }
    explicit __mlu_func__ operator float() const {
        return nt_wide_float<float>(*this);
    }
    explicit __mlu_func__ operator double() const {
        return nt_wide_float<double>(*this);
    }
    __mlu_func__ NtWide operator~() const { return NtWide(~lo, ~hi); }
    __mlu_func__ NtWide operator-() const { return NtWide(~lo + 1u, ~hi + (lo == 0)); }
    __mlu_func__ NtWide operator+(NtWide b) const {
        uint32_t low = lo + b.lo; return NtWide(low, hi + b.hi + (low < lo));
    }
    __mlu_func__ NtWide operator-(NtWide b) const { return *this + (-b); }
    __mlu_func__ NtWide operator*(NtWide b) const {
        uint32_t t = (lo & 65535u) * (b.lo & 65535u);
        t = (lo >> 16) * (b.lo & 65535u) + (t >> 16);
        uint32_t middle = (t & 65535u) + (lo & 65535u) * (b.lo >> 16);
        uint32_t high = (lo >> 16) * (b.lo >> 16) + (t >> 16) + (middle >> 16);
        return NtWide(lo * b.lo, high + hi * b.lo + lo * b.hi);
    }
    __mlu_func__ NtWide operator&(NtWide b) const { return NtWide(lo & b.lo, hi & b.hi); }
    __mlu_func__ NtWide operator|(NtWide b) const { return NtWide(lo | b.lo, hi | b.hi); }
    __mlu_func__ NtWide operator^(NtWide b) const { return NtWide(lo ^ b.lo, hi ^ b.hi); }
    __mlu_func__ bool operator==(NtWide b) const { return lo == b.lo && hi == b.hi; }
    __mlu_func__ bool operator!=(NtWide b) const { return !(*this == b); }
    __mlu_func__ bool operator<(NtWide b) const {
        if (hi != b.hi) { return Signed ? (int32_t)hi < (int32_t)b.hi : hi < b.hi; }
        return lo < b.lo;
    }
    __mlu_func__ bool operator>(NtWide b) const { return b < *this; }
    __mlu_func__ bool operator<=(NtWide b) const { return !(b < *this); }
    __mlu_func__ bool operator>=(NtWide b) const { return !(*this < b); }
    __mlu_func__ NtWide operator<<(NtWide count) const {
        uint32_t n = count.lo;
        if (count.hi || n >= 64) { return NtWide(0u, 0u); }
        if (!n) { return *this; }
        if (n >= 32) { return NtWide(0u, lo << (n - 32)); }
        return NtWide(lo << n, (hi << n) | (lo >> (32 - n)));
    }
    __mlu_func__ NtWide operator>>(NtWide count) const {
        uint32_t n = count.lo;
        uint32_t sign = Signed && (int32_t)hi < 0 ? 0xffffffffu : 0u;
        if (count.hi || n >= 64) { return NtWide(sign, sign); }
        if (!n) { return *this; }
        if (n >= 32) {
            uint32_t low = Signed ? (uint32_t)((int32_t)hi >> (n - 32)) : hi >> (n - 32);
            return NtWide(low, sign);
        }
        uint32_t high = Signed ? (uint32_t)((int32_t)hi >> n) : hi >> n;
        return NtWide((lo >> n) | (hi << (32 - n)), high);
    }
    __mlu_func__ NtWide divide(NtWide divisor, bool remainder) const {
        bool neg_a = Signed && (int32_t)hi < 0;
        bool neg_b = Signed && (int32_t)divisor.hi < 0;
        NtWide numerator = neg_a ? -*this : *this;
        if (neg_b) { divisor = -divisor; }
        NtWide quotient(0u, 0u), rest(0u, 0u);
        for (int bit = 63; bit >= 0; --bit) {
            bool overflow = (rest.hi & 0x80000000u) != 0;
            rest.hi = (rest.hi << 1) | (rest.lo >> 31);
            rest.lo = (rest.lo << 1) | (bit >= 32 ? (numerator.hi >> (bit - 32)) & 1u : (numerator.lo >> bit) & 1u);
            if (overflow || rest.hi > divisor.hi || (rest.hi == divisor.hi && rest.lo >= divisor.lo)) {
                rest = rest - divisor;
                if (bit >= 32) { quotient.hi |= 1u << (bit - 32); }
                else { quotient.lo |= 1u << bit; }
            }
        }
        return remainder ? (neg_a ? -rest : rest) : (neg_a != neg_b ? -quotient : quotient);
    }
    __mlu_func__ NtWide operator/(NtWide b) const { return divide(b, false); }
    __mlu_func__ NtWide operator%(NtWide b) const { return divide(b, true); }
};
using NtInt64 = NtWide<true>;
using NtUInt64 = NtWide<false>;

template <class F, bool S> static __mlu_func__ F nt_wide_float(NtWide<S> value) {
    bool negative = S && (int32_t)value.hi < 0;
    NtUInt64 magnitude(negative ? -value : value);
    uint32_t word = magnitude.hi ? magnitude.hi : magnitude.lo;
    int bits = magnitude.hi ? 32 : 0;
    for (; word; word >>= 1) { ++bits; }
    int shift = bits - std::numeric_limits<F>::digits;
    if (shift < 0) { shift = 0; }
    NtUInt64 kept = magnitude >> NtUInt64((uint32_t)shift);
    if (shift) {
        // Round once using the complete discarded suffix, ties to even.
        NtUInt64 rest = magnitude - (kept << NtUInt64((uint32_t)shift));
        NtUInt64 half = NtUInt64(1u) << NtUInt64((uint32_t)(shift - 1));
        if (rest > half || (rest == half && (kept.lo & 1u))) {
            kept = kept + NtUInt64(1u);
        }
    }
    // After rounding, both words and their sum are exact in F. Scaling by
    // an exact power of two cannot introduce another rounding step.
    union { uint32_t bits; float value; } scale;
    scale.bits = (uint32_t)(shift + 127) << 23;
    F result = (F(kept.hi) * F(4294967296.0) + F(kept.lo)) * F(scale.value);
    return negative ? -result : result;
}

template <bool S> static __mlu_func__ NtWide<S> nt_abs(NtWide<S> value) {
    return S && (int32_t)value.hi < 0 ? -value : value;
}
template <class T> static __mlu_func__
typename std::enable_if<std::is_integral<T>::value, T>::type nt_abs(T value) {
    using U = typename std::make_unsigned<T>::type;
    return value < T(0) ? T(U(0) - U(value)) : value;
}
template <class T> static __mlu_func__
typename std::enable_if<!std::is_integral<T>::value, T>::type nt_abs(T value) {
    return T(fabsf((float)value));
}
static __mlu_func__ double nt_abs(double value) { return fabs(value); }
template <bool S> static __mlu_func__ int32_t nt_cast_int32(NtWide<S> value) {
    return (int32_t)value.lo;
}
template <bool S> static __mlu_func__ NtWide<S> _nt_modulo(NtWide<S> x, NtWide<S> y) {
    NtWide<S> zero(0u, 0u), r = x % y;
    return r != zero && ((r < zero) != (y < zero)) ? r + y : r;
}
template <bool Signed> static NtWide<Signed> nt_host_wide(int64_t value) {
    NtWide<Signed> result;
    const uint32_t *words = (const uint32_t *)&value;
    result.lo = words[0]; result.hi = words[1]; return result;
}
"""
