#include "native_runtime.h"

#include <initializer_list>
#include <limits.h>
#include <math.h>
#include <stdlib.h>

namespace {

bool product(std::initializer_list<size_t> factors, size_t *result) {
    size_t value = 1;
    for (size_t factor : factors) {
        if (factor == 0 || value > static_cast<size_t>(INT64_MAX) / factor) {
            return false;
        }
        value *= factor;
    }
    *result = value;
    return true;
}

template <typename T> class Buffer {
public:
    explicit Buffer(size_t count, bool clear = false) : data(nullptr) {
        if (count != 0 && count <= SIZE_MAX / sizeof(T)) {
            data = static_cast<T *>(clear ? calloc(count, sizeof(T))
                                         : malloc(count * sizeof(T)));
        }
    }
    ~Buffer() { free(data); }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    T *data;
};

struct ConvGeometry {
    size_t oh, ow, spatial, positions, synapses, blocks;
    size_t inputs, weights, outputs, patches, packed_weights;
};

bool output_extent(int64_t input, int64_t kernel, int64_t stride,
                   int64_t padding, size_t *output) {
    if (input <= 0 || kernel <= 0 || stride <= 0 || padding < 0 ||
        padding > (INT64_MAX - input) / 2) {
        return false;
    }
    const int64_t padded = input + 2 * padding;
    if (kernel > padded) {
        return false;
    }
    *output = static_cast<size_t>((padded - kernel) / stride + 1);
    return true;
}

bool convolution_geometry(const NtConvSpec &s, ConvGeometry *g) {
    if (s.batch <= 0 || s.cin <= 0 || s.cout <= 0 ||
        !output_extent(s.height, s.kh, s.sh, s.ph, &g->oh) ||
        !output_extent(s.width, s.kw, s.sw, s.pw, &g->ow)) {
        return false;
    }
    const size_t b = s.batch, c = s.cin, h = s.height, w = s.width;
    const size_t oc = s.cout, kh = s.kh, kw = s.kw;
    if (!product({g->oh, g->ow}, &g->spatial) ||
        !product({b, g->spatial}, &g->positions) ||
        !product({c, kh, kw}, &g->synapses)) {
        return false;
    }
    g->blocks = g->synapses / 32 + (g->synapses % 32 != 0);
    return product({b, c, h, w}, &g->inputs) &&
           product({oc, g->synapses}, &g->weights) &&
           product({oc, g->positions}, &g->outputs) &&
           product({g->blocks, g->positions}, &g->patches) &&
           product({g->blocks, oc}, &g->packed_weights);
}

const NtConvSpec *find_convolution(const NtConvSpec &shape) {
    for (size_t i = 0; i < nt_conv_count; ++i) {
        const NtConvSpec &s = nt_conv_specs[i];
        if (s.batch == shape.batch && s.cin == shape.cin &&
            s.height == shape.height && s.width == shape.width &&
            s.cout == shape.cout && s.kh == shape.kh && s.kw == shape.kw &&
            s.sh == shape.sh && s.sw == shape.sw &&
            s.ph == shape.ph && s.pw == shape.pw) {
            return &s;
        }
    }
    return nullptr;
}

void initialize_convolution_registers() {
    // Match the old SConv runtime's defaults, including zero weight shift.
    // Prior hardware LIF calls may have changed both neuron parameter banks.
    __builtin_riscv_calc_sync();
    __builtin_riscv_clear_all_1024();
    __builtin_riscv_set_vtr_64(7, 7);
    __builtin_riscv_set_rpr_64(0, 0);
    __builtin_riscv_set_rvr_64(0, 0);
    __builtin_riscv_set_rcr_64(0, 0);
    __builtin_riscv_set_lpr_64(0, 0, 0, 12, 0, 0, 0, 0, 0, 12, 0, 0);
    for (unsigned int i = 0; i < 32; ++i) {
        __builtin_riscv_set_ntr_32(i, UINT32_MAX);
        __builtin_riscv_set_svr_32(0, i);
        __builtin_riscv_set_sor_32(0, i);
    }
    for (unsigned int i = 0; i < 128; ++i) {
        __builtin_riscv_set_wvr_32(0, i);
    }
}

template <typename T>
void pack_patches(const NtConvSpec &s, const ConvGeometry &g,
                  const T *input, uint32_t *patches) {
    const size_t channels = s.cin, height = s.height, width = s.width;
    const size_t kh = s.kh, kw = s.kw, ph = s.ph, pw = s.pw;
    for (size_t batch = 0; batch < static_cast<size_t>(s.batch); ++batch) {
        for (size_t row = 0; row < g.oh; ++row) {
            for (size_t column = 0; column < g.ow; ++column) {
                const size_t position = (batch * g.oh + row) * g.ow + column;
                for (size_t channel = 0; channel < channels; ++channel) {
                    for (size_t kr = 0; kr < kh; ++kr) {
                        const size_t padded_row = row * s.sh + kr;
                        if (padded_row < ph || padded_row - ph >= height) {
                            continue;
                        }
                        for (size_t kc = 0; kc < kw; ++kc) {
                            const size_t padded_column = column * s.sw + kc;
                            if (padded_column < pw || padded_column - pw >= width) {
                                continue;
                            }
                            const size_t offset =
                                ((batch * channels + channel) * height +
                                 padded_row - ph) * width + padded_column - pw;
                            if (input[offset] != static_cast<T>(0)) {
                                const size_t synapse = (channel * kh + kr) * kw + kc;
                                patches[(synapse / 32) * g.positions + position] |=
                                    UINT32_C(1) << (synapse % 32);
                            }
                        }
                    }
                }
            }
        }
    }
}

int kernel_status(int status) {
    return status <= 0 ? status : NT_KERNEL_FAILED;
}

template <typename Input, typename Output>
int execute_convolution(const NtConvSpec &s, const ConvGeometry &g,
                        const Input *input, const uint64_t *low,
                        const uint64_t *high, Output *output) {
    if (!s.kernel || !low || !high) {
        return NT_INVALID_ARGUMENT;
    }
    Buffer<uint32_t> patches(g.patches, true);
    Buffer<int32_t> currents(g.outputs);
    if (!patches.data || !currents.data) {
        return NT_ALLOCATION_FAILED;
    }
    pack_patches(s, g, input, patches.data);
    initialize_convolution_registers();
    const int status = kernel_status(s.kernel(patches.data, low, high, currents.data));
    if (status != NT_SUCCESS) {
        return status;
    }

    // The generated kernel shares [blocks, channels] weights across positions
    // and produces [channels, batch * spatial]; expose contiguous NCHW here.
    for (size_t batch = 0; batch < static_cast<size_t>(s.batch); ++batch) {
        for (size_t channel = 0; channel < static_cast<size_t>(s.cout); ++channel) {
            for (size_t position = 0; position < g.spatial; ++position) {
                output[(batch * s.cout + channel) * g.spatial + position] =
                    static_cast<Output>(currents.data[
                        channel * g.positions + batch * g.spatial + position]);
            }
        }
    }
    return NT_SUCCESS;
}

int pack_float_weights(const NtConvSpec &s, const ConvGeometry &g,
                       const float *weights, uint64_t *low, uint64_t *high) {
    for (size_t channel = 0; channel < static_cast<size_t>(s.cout); ++channel) {
        for (size_t synapse = 0; synapse < g.synapses; ++synapse) {
            const float value = weights[channel * g.synapses + synapse];
            if (!isfinite(value)) {
                return NT_INVALID_VALUE;
            }
            // Clamp before converting to int so large finite inputs are safe.
            const int quantized = value <= -8.0f ? -8 : value >= 7.0f ? 7 :
                static_cast<int>(value >= 0.0f ? value + 0.5f : value - 0.5f);
            const uint64_t bits = static_cast<uint32_t>(quantized) & UINT32_C(15);
            const size_t lane = synapse % 32;
            uint64_t *destination = lane < 16 ? low : high;
            destination[(synapse / 32) * s.cout + channel] |= bits << ((lane % 16) * 4);
        }
    }
    return NT_SUCCESS;
}

bool overlaps(const void *first, const void *second, size_t bytes) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(first);
    const uintptr_t b = reinterpret_cast<uintptr_t>(second);
    if (a > UINTPTR_MAX - bytes || b > UINTPTR_MAX - bytes) {
        return true;
    }
    return a < b + bytes && b < a + bytes;
}

}  // namespace

extern "C" int nt_conv_i32(const NtConvSpec *spec, const int32_t *input, int32_t *output) {
    if (!spec || !input || !output) {
        return NT_INVALID_ARGUMENT;
    }
    ConvGeometry geometry;
    if (!convolution_geometry(*spec, &geometry)) {
        return NT_INVALID_SHAPE;
    }
    for (size_t i = 0; i < geometry.inputs; ++i) {
        if (input[i] != 0 && input[i] != 1) {
            return NT_INVALID_VALUE;
        }
    }
    return execute_convolution(*spec, geometry, input, spec->weight_low,
                               spec->weight_high, output);
}

extern "C" int nt_pool_i32(const NtPoolSpec *spec, const int32_t *input, int32_t *output) {
    if (!spec || !spec->kernel || !input || !output) {
        return NT_INVALID_ARGUMENT;
    }
    if (spec->batch <= 0 || spec->channels <= 0 || spec->height < 2 || spec->width < 2) {
        return NT_INVALID_SHAPE;
    }
    const size_t batch = spec->batch, channels = spec->channels;
    const size_t height = spec->height, width = spec->width;
    const size_t oh = height / 2, ow = width / 2;
    size_t inputs, outputs, sample_count;
    if (!product({batch, channels, height, width}, &inputs) ||
        !product({batch, channels, oh, ow}, &outputs) ||
        !product({4, outputs}, &sample_count)) {
        return NT_INVALID_SHAPE;
    }
    Buffer<int32_t> samples(sample_count);
    if (!samples.data) {
        return NT_ALLOCATION_FAILED;
    }
    for (size_t group = 0; group < batch * channels; ++group) {
        for (size_t row = 0; row < oh; ++row) {
            for (size_t column = 0; column < ow; ++column) {
                const size_t destination = (group * oh + row) * ow + column;
                const size_t source = (group * height + row * 2) * width + column * 2;
                samples.data[destination] = input[source];
                samples.data[outputs + destination] = input[source + 1];
                samples.data[2 * outputs + destination] = input[source + width];
                samples.data[3 * outputs + destination] = input[source + width + 1];
            }
        }
    }
    return kernel_status(spec->kernel(samples.data, output));
}

extern "C" int nt_lif_i32(
    const NtLifSpec *spec, const int32_t *current, const int32_t *voltage,
    const int32_t *syn_current, int32_t *spike,
    int32_t *next_voltage, int32_t *next_current) {
    if (!spec || !spec->kernel || !current || !voltage || !syn_current ||
        !spike || !next_voltage || !next_current) {
        return NT_INVALID_ARGUMENT;
    }
    if (spec->count <= 0 || static_cast<uint64_t>(spec->count) > SIZE_MAX / sizeof(int32_t)) {
        return NT_INVALID_SHAPE;
    }
    const size_t bytes = static_cast<size_t>(spec->count) * sizeof(int32_t);
    const int32_t *inputs[] = {current, voltage, syn_current};
    int32_t *outputs[] = {spike, next_voltage, next_current};
    for (size_t i = 0; i < 3; ++i) {
        for (const int32_t *input : inputs) {
            if (overlaps(outputs[i], input, bytes)) {
                return NT_INVALID_ARGUMENT;
            }
        }
        for (size_t j = 0; j < i; ++j) {
            if (overlaps(outputs[i], outputs[j], bytes)) {
                return NT_INVALID_ARGUMENT;
            }
        }
    }
    return kernel_status(spec->kernel(current, voltage, syn_current,
                                      spike, next_voltage, next_current));
}

extern "C" int conv_compute_currents_from_f32_ptr_defaultNP_f32out(
    int32_t T, int32_t CHI, int32_t HI, int32_t WI,
    int32_t CHO, int32_t K, int32_t STRIDE, int32_t PADDING,
    const float *input_linear, int64_t input_elems,
    const float *weight_linear, int64_t weight_elems,
    float *out_linear, int64_t out_elems) {
    return conv_compute_currents_from_f32_ptr_defaultNP_f32out_nonsquare(
        T, CHI, HI, WI, CHO, K, K, STRIDE, STRIDE, PADDING, PADDING,
        input_linear, input_elems, weight_linear, weight_elems, out_linear, out_elems);
}

extern "C" int conv_compute_currents_from_f32_ptr_defaultNP_f32out_nonsquare(
    int32_t T, int32_t CHI, int32_t HI, int32_t WI, int32_t CHO,
    int32_t KH, int32_t KW, int32_t STRIDE_H, int32_t STRIDE_W,
    int32_t PAD_H, int32_t PAD_W,
    const float *input_linear, int64_t input_elems,
    const float *weight_linear, int64_t weight_elems,
    float *out_linear, int64_t out_elems) {
    if (!input_linear || !weight_linear || !out_linear) {
        return NT_INVALID_ARGUMENT;
    }
    const NtConvSpec shape = {T, CHI, HI, WI, CHO, KH, KW, STRIDE_H, STRIDE_W,
                              PAD_H, PAD_W, nullptr, nullptr, nullptr};
    ConvGeometry geometry;
    if (!convolution_geometry(shape, &geometry)) {
        return NT_INVALID_SHAPE;
    }
    if (input_elems != static_cast<int64_t>(geometry.inputs)) {
        return NT_INPUT_COUNT_MISMATCH;
    }
    if (weight_elems != static_cast<int64_t>(geometry.weights)) {
        return NT_WEIGHT_COUNT_MISMATCH;
    }
    if (out_elems != static_cast<int64_t>(geometry.outputs)) {
        return NT_OUTPUT_COUNT_MISMATCH;
    }
    const NtConvSpec *spec = find_convolution(shape);
    if (!spec) {
        return NT_UNKNOWN_SHAPE;
    }
    Buffer<uint64_t> low(geometry.packed_weights, true);
    Buffer<uint64_t> high(geometry.packed_weights, true);
    if (!low.data || !high.data) {
        return NT_ALLOCATION_FAILED;
    }
    const int status = pack_float_weights(*spec, geometry, weight_linear, low.data, high.data);
    if (status != NT_SUCCESS) {
        return status;
    }
    return execute_convolution(*spec, geometry, input_linear, low.data, high.data, out_linear);
}
