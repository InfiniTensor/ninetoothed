#ifndef NINETOOTHED_RVNE_NATIVE_RUNTIME_H
#define NINETOOTHED_RVNE_NATIVE_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*NtConvKernel)(const uint32_t *, const uint64_t *, const uint64_t *, int32_t *);
typedef int (*NtPoolKernel)(const int32_t *, int32_t *);
typedef int (*NtLifKernel)(const int32_t *, const int32_t *, const int32_t *,
                         int32_t *, int32_t *, int32_t *);

typedef struct NtConvSpec {
    int64_t batch, cin, height, width, cout, kh, kw, sh, sw, ph, pw;
    NtConvKernel kernel;
    const uint64_t *weight_low;
    const uint64_t *weight_high;
} NtConvSpec;

typedef struct NtPoolSpec {
    int64_t batch, channels, height, width;
    NtPoolKernel kernel;
} NtPoolSpec;

typedef struct NtLifSpec {
    int64_t count;
    NtLifKernel kernel;
} NtLifSpec;

enum NtStatus {
    NT_SUCCESS = 0,
    NT_INVALID_ARGUMENT = -1,
    NT_INVALID_SHAPE = -2,
    NT_ALLOCATION_FAILED = -10,
    NT_UNKNOWN_SHAPE = -20,
    NT_INPUT_COUNT_MISMATCH = -21,
    NT_WEIGHT_COUNT_MISMATCH = -22,
    NT_OUTPUT_COUNT_MISMATCH = -23,
    NT_INVALID_VALUE = -24,
    NT_KERNEL_FAILED = -25
};

extern const NtConvSpec nt_conv_specs[];
extern const size_t nt_conv_count;
extern const NtPoolSpec nt_pool_specs[];
extern const size_t nt_pool_count;
extern const NtLifSpec nt_lif_specs[];
extern const size_t nt_lif_count;

/* These adapters call only kernels compiled for the dimensions in their specs.
 * Inputs and outputs use contiguous NCHW layout. The convolution adapter requires
 * binary input and resets RVNE neuron registers and parameters before execution;
 * callers must keep persistent LIF state in memory and serialize RVNE access.
 * LIF outputs must not overlap each other or any of the three input buffers.
 */
int nt_conv_i32(const NtConvSpec *spec, const int32_t *input, int32_t *output);
int nt_pool_i32(const NtPoolSpec *spec, const int32_t *input, int32_t *output);
int nt_lif_i32(const NtLifSpec *spec, const int32_t *current, const int32_t *voltage,
               const int32_t *syn_current, int32_t *spike,
               int32_t *next_voltage, int32_t *next_current);

/* Compatibility entrypoints for the supplied SConv C ABI. Input floats are
 * converted by x != 0; finite weights are rounded halfway away from zero and
 * clipped to [-8, 7]. The caller's weights are used on every call. Shapes without
 * a compiled registry entry fail explicitly. Element counts are checked before
 * writing output, and non-finite weights are rejected.
 */
int conv_compute_currents_from_f32_ptr_defaultNP_f32out(
    int32_t T, int32_t CHI, int32_t HI, int32_t WI,
    int32_t CHO, int32_t K, int32_t STRIDE, int32_t PADDING,
    const float *input_linear, int64_t input_elems,
    const float *weight_linear, int64_t weight_elems,
    float *out_linear, int64_t out_elems);

int conv_compute_currents_from_f32_ptr_defaultNP_f32out_nonsquare(
    int32_t T, int32_t CHI, int32_t HI, int32_t WI, int32_t CHO,
    int32_t KH, int32_t KW, int32_t STRIDE_H, int32_t STRIDE_W,
    int32_t PAD_H, int32_t PAD_W,
    const float *input_linear, int64_t input_elems,
    const float *weight_linear, int64_t weight_elems,
    float *out_linear, int64_t out_elems);

#ifdef __cplusplus
}
#endif

#endif
