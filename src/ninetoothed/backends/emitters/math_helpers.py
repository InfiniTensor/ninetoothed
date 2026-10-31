"""Portable float32 trigonometric helpers for scalar device emitters.

Range reduction consumes the exact float32 significand, with compensated
two-part constants for every power of two up to the maximum finite exponent.
The emitter supplies the device function qualifier, bit union and infinity.
"""

_FLOAT32_TRIG_SUPPORT = """__aicore__ inline void nt_trig_add(float &hi, float &lo, float term_hi, float term_lo) {
    float sum = hi + term_hi;
    float virtual_term = sum - hi;
    lo += (hi - (sum - virtual_term)) + (term_hi - virtual_term) + term_lo;
    hi = sum;
    sum = hi + lo;
    lo = (hi - sum) + lo;
    hi = sum;
}

__aicore__ inline float nt_trig_remainder(float x) {
    // Reduce the exact float32 significand by adding 2^k mod 2*pi.
    // Each table entry has two float32 parts generated with 170-digit pi.
    // Compensation keeps full-range inputs independent of an overflowing
    // float-to-int conversion or a large, rounded multiple of 2*pi.
    float ax = x < 0.0f ? -x : x;
    if (ax != ax) { return ax; }
    if (ax <= 3.14159274f) { return x; }
    if (ax >= nt_inf()) { return ax - ax; }
    const float high[128] = {1.000000000e+0f, 2.000000000e+0f, -2.283185244e+0f, 1.716814637e+0f, -2.849555969e+0f, 5.840734839e-1f, 1.168146968e+0f, 2.336293936e+0f, -1.610597610e+0f, 3.061990023e+0f, -1.592050642e-1f, -3.184101284e-1f, -6.368202567e-1f, -1.273640513e+0f, -2.547281027e+0f, 1.188623071e+0f, 2.377246141e+0f, -1.528693080e+0f, -3.057386160e+0f, 1.684130132e-1f, 3.368260264e-1f, 6.736520529e-1f, 1.347304106e+0f, 2.694608212e+0f, -8.939688802e-1f, -1.787937760e+0f, 2.707309723e+0f, -8.685656190e-1f, -1.737131238e+0f, 2.808922768e+0f, -6.653397083e-1f, -1.330679417e+0f, -2.661358833e+0f, 9.604676962e-1f, 1.920935392e+0f, -2.441314459e+0f, 1.400556207e+0f, 2.801112413e+0f, -6.809606552e-1f, -1.361921310e+0f, -2.723842621e+0f, 8.355000615e-1f, 1.671000123e+0f, -2.941184998e+0f, 4.008153677e-1f, 8.016307354e-1f, 1.603261471e+0f, -3.076662302e+0f, 1.298606992e-1f, 2.597213984e-1f, 5.194427967e-1f, 1.038885593e+0f, 2.077771187e+0f, -2.127642870e+0f, 2.027899504e+0f, -2.227386236e+0f, 1.828412890e+0f, -2.626359463e+0f, 1.030466318e+0f, 2.060932636e+0f, -2.161319971e+0f, 1.960545301e+0f, -2.362094641e+0f, 1.558995962e+0f, 3.117991924e+0f, -4.720140249e-2f, -9.440280497e-2f, -1.888056099e-1f, -3.776112199e-1f, -7.552224398e-1f, -1.510444880e+0f, -3.020889759e+0f, 2.414059192e-1f, 4.828118384e-1f, 9.656236768e-1f, 1.931247354e+0f, -2.420690775e+0f, 1.441804051e+0f, 2.883608103e+0f, -5.159693360e-1f, -1.031938672e+0f, -2.063877344e+0f, 2.155430794e+0f, -1.972323895e+0f, 2.338537455e+0f, -1.606110215e+0f, 3.070964813e+0f, -1.412555873e-1f, -2.825111747e-1f, -5.650223494e-1f, -1.130044699e+0f, -2.260089397e+0f, 1.763006568e+0f, -2.757172108e+0f, 7.688411474e-1f, 1.537682295e+0f, 3.075364590e+0f, -1.324560642e-1f, -2.649121284e-1f, -5.298242569e-1f, -1.059648514e+0f, -2.119297028e+0f, 2.044591188e+0f, -2.194002867e+0f, 1.895179391e+0f, -2.492826462e+0f, 1.297532320e+0f, 2.595064640e+0f, -1.093055844e+0f, -2.186111689e+0f, 1.910961747e+0f, -2.461261988e+0f, 1.360661507e+0f, 2.721323013e+0f, -8.405393362e-1f, -1.681078672e+0f, 2.921028137e+0f, -4.411291778e-1f, -8.822583556e-1f, -1.764516711e+0f, 2.754151821e+0f, -7.748815417e-1f, -1.549763083e+0f, -3.099526167e+0f, 8.413313329e-2f, 1.682662666e-1f, 3.365325332e-1f, 6.730650663e-1f};
    const float low[128] = {0.000000000e+0f, 0.000000000e+0f, -6.357301885e-8f, 5.563627070e-8f, 4.769952255e-8f, -1.984187215e-8f, -3.968374429e-8f, -7.936748858e-8f, 1.611058664e-8f, 9.579419213e-8f, -6.095082661e-9f, -1.219016532e-8f, -2.438033064e-8f, -4.876066129e-8f, -9.752132257e-8f, -1.226033497e-8f, -2.452066994e-8f, 6.594929935e-9f, 1.318985987e-8f, 5.457725383e-10f, 1.091545077e-9f, 2.183090153e-9f, 4.366180306e-9f, 8.732360612e-9f, 1.349634715e-8f, 2.699269430e-8f, 1.175584075e-7f, -7.270137381e-9f, -1.454027476e-8f, 3.449246933e-8f, 5.411921578e-9f, 1.082384316e-8f, 2.164768631e-8f, -1.234089630e-8f, -2.468179261e-8f, -1.129366041e-7f, -4.309089974e-8f, -8.618179947e-8f, 2.481955308e-9f, 4.963910616e-9f, 9.927821232e-9f, 2.382401654e-8f, 4.764803307e-8f, 3.172304730e-8f, 7.809825675e-9f, 1.561965135e-8f, 3.123930270e-8f, -1.094416668e-9f, 1.779541070e-9f, 3.559082140e-9f, 7.118164280e-9f, 1.423632856e-8f, 2.847265712e-8f, -6.627705496e-9f, 5.031760608e-8f, 3.706219687e-8f, 1.848812303e-8f, -2.659677456e-8f, 1.037947150e-8f, 2.075894301e-8f, -2.205513283e-8f, 1.946275496e-8f, -2.464751070e-8f, 1.427799834e-8f, 2.855599668e-8f, 9.895549935e-10f, 1.979109987e-9f, 3.958219974e-9f, 7.916439948e-9f, 1.583287990e-8f, 3.166575979e-8f, 6.333151958e-8f, -3.479042388e-9f, -6.958084775e-9f, -1.391616955e-8f, -2.783233910e-8f, 1.191808821e-7f, -5.569308925e-8f, -1.113861785e-7f, 1.167785069e-8f, 2.335570137e-8f, 4.671140275e-8f, -8.142275476e-8f, 1.200005695e-8f, 8.757313452e-8f, -7.636043797e-9f, 4.830092948e-8f, 3.226519940e-9f, 6.453039880e-9f, 1.290607976e-8f, 2.581215952e-8f, 5.162431904e-8f, 4.761236738e-8f, 3.165171236e-8f, 7.667154911e-9f, 1.533430982e-8f, 3.066861964e-8f, -2.235776453e-9f, -4.471552906e-9f, -8.943105811e-9f, -1.788621162e-8f, -3.577242325e-8f, -7.971829419e-9f, -7.951667413e-8f, 2.374895303e-8f, -1.607511280e-8f, 3.142279681e-8f, 6.284559362e-8f, -5.709112827e-8f, -1.141822565e-7f, -4.558220468e-8f, 8.368115090e-8f, -7.483254905e-9f, -1.496650981e-8f, 2.570324931e-8f, 5.140649861e-8f, -7.203256303e-8f, 9.781162547e-10f, 1.956232509e-9f, 3.912465019e-9f, 7.139794889e-8f, 1.961823415e-8f, 3.923646830e-8f, 7.847293659e-8f, -2.998524096e-9f, -5.997048191e-9f, -1.199409638e-8f, -2.398819277e-8f};
    NtFloatBits bits;
    bits.f = ax;
    int exponent = ((bits.i >> 23) & 0xff) - 127;
    int mantissa = (bits.i & 0x007fffff) | 0x00800000;
    int first_power = exponent - 23;
    float hi = 0.0f;
    float lo = 0.0f;
    if (first_power < 0) {
        int fractional_bits = -first_power;
        NtFloatBits scale;
        scale.i = (127 - fractional_bits) << 23;
        hi = (float)(mantissa & ((1 << fractional_bits) - 1)) * scale.f;
    }
    for (int bit = 0; bit < 24; bit++) {
        int power = first_power + bit;
        if (power >= 0 && (mantissa & (1 << bit))) {
            nt_trig_add(hi, lo, high[power], low[power]);
            if (hi > 3.14159274f) {
                nt_trig_add(hi, lo, -6.28318548f, 1.74845553e-7f);
            } else if (hi < -3.14159274f) {
                nt_trig_add(hi, lo, 6.28318548f, -1.74845553e-7f);
            }
        }
    }
    float remainder = hi + lo;
    return x < 0.0f ? -remainder : remainder;
}

__aicore__ inline float nt_trig_poly(float s, bool cosine) {
    // Nearest-quadrant fold leaves |t| <= pi/4 for both series.
    int q = (int)(s * 0.63661975f + (s < 0.0f ? -0.5f : 0.5f));
    float t = (s - (float)q * 1.57079625f) - (float)q * 7.54978942e-8f;
    if (cosine) { q += 1; }
    float t2 = t * t;
    float sin_t = t * (1.0f - t2 * (0.16666667f - t2 * (0.0083333310f
        - t2 * (1.9841270e-4f - t2 * 2.7557314e-6f))));
    float cos_t = 1.0f - t2 * (0.50000000f - t2 * (0.041666638f
        - t2 * (0.0013888378f - t2 * 2.4433157e-5f)));
    float result = (q & 1) ? cos_t : sin_t;
    if (q & 2) { result = -result; }
    return result;
}

__aicore__ inline float nt_sin(float x) {
    return nt_trig_poly(nt_trig_remainder(x), false);
}

__aicore__ inline float nt_cos(float x) {
    return nt_trig_poly(nt_trig_remainder(x), true);
}
"""


def render_float32_trig_support(
    qualifier, *, bit_type="NtFloatBits", infinity="nt_inf"
):
    """Render scalar sin/cos with accurate full-range float32 range reduction.

    The bit union must expose a float ``f`` and a signed 32-bit integer ``i``.
    ``infinity`` names a zero-argument function returning positive infinity.
    """
    return (
        _FLOAT32_TRIG_SUPPORT.replace("__aicore__", qualifier)
        .replace("NtFloatBits", bit_type)
        .replace("nt_inf()", f"{infinity}()")
    )


__all__ = ["render_float32_trig_support"]
