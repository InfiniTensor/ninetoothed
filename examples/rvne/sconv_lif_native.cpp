#include "native_runtime.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <new>
#include <vector>

namespace {
constexpr size_t input_count = 4 * 3 * 64 * 64;
constexpr size_t output_count = 1 * 3 * 2 * 2 * 9;
int model_status = NT_SUCCESS;
size_t conv_calls = 0, pool_calls = 0, lif_calls = 0;
FILE *trace_file = nullptr;
uint32_t trace_index = 0;

size_t conv_elements(const NtConvSpec &spec) {
    const int64_t h = (spec.height + 2 * spec.ph - spec.kh) / spec.sh + 1;
    const int64_t w = (spec.width + 2 * spec.pw - spec.kw) / spec.sw + 1;
    return static_cast<size_t>(spec.batch * spec.cout * h * w);
}

bool trace(const int32_t *values, size_t count) {
    if (!trace_file) return true;
    const uint32_t header[2] = {trace_index++, static_cast<uint32_t>(count)};
    return fwrite(header, sizeof(uint32_t), 2, trace_file) == 2 &&
           fwrite(values, sizeof(int32_t), count, trace_file) == count;
}

int execute_model(const float *input, float *output) {
    if (nt_conv_count != 10 || nt_pool_count != 3 || nt_lif_count != 9)
        return NT_INVALID_SHAPE;

    size_t capacity = input_count / 4;
    size_t state_capacity = 0;
    for (size_t layer = 0; layer < nt_conv_count; ++layer)
        capacity = std::max(capacity, conv_elements(nt_conv_specs[layer]));
    for (size_t layer = 0; layer < nt_lif_count; ++layer)
        state_capacity = std::max(state_capacity, static_cast<size_t>(nt_lif_specs[layer].count));

    std::vector<int32_t> first(capacity), second(capacity);
    std::vector<int32_t> next_voltage(state_capacity), next_current(state_capacity);
    std::vector<int32_t> voltage[9], current[9];
    for (size_t layer = 0; layer < 9; ++layer) {
        voltage[layer].assign(nt_lif_specs[layer].count, 0);
        current[layer].assign(nt_lif_specs[layer].count, 0);
    }

    for (size_t step = 0; step < 4; ++step) {
        int32_t *value = first.data();
        int32_t *scratch = second.data();
        for (size_t index = 0; index < input_count / 4; ++index)
            value[index] = input[step * (input_count / 4) + index] != 0.0f;

        for (size_t layer = 0; layer < 9; ++layer) {
            int status = nt_conv_i32(&nt_conv_specs[layer], value, scratch);
            if (status) return status;
            ++conv_calls;
            std::swap(value, scratch);
            if (!trace(value, conv_elements(nt_conv_specs[layer]))) return -101;

            if (layer < 3) {
                status = nt_pool_i32(&nt_pool_specs[layer], value, scratch);
                if (status) return status;
                ++pool_calls;
                std::swap(value, scratch);
                const auto &pool = nt_pool_specs[layer];
                size_t count = pool.batch * pool.channels * (pool.height / 2) * (pool.width / 2);
                if (!trace(value, count)) return -101;
            }

            const size_t count = nt_lif_specs[layer].count;
            status = nt_lif_i32(&nt_lif_specs[layer], value, voltage[layer].data(),
                               current[layer].data(), scratch, next_voltage.data(),
                               next_current.data());
            if (status) return status;
            ++lif_calls;
            std::swap(value, scratch);
            std::copy_n(next_voltage.data(), count, voltage[layer].data());
            std::copy_n(next_current.data(), count, current[layer].data());
            if (!trace(value, count) || !trace(voltage[layer].data(), count) ||
                !trace(current[layer].data(), count)) return -101;
        }

        int status = nt_conv_i32(&nt_conv_specs[9], value, scratch);
        if (status) return status;
        ++conv_calls;
        std::swap(value, scratch);
        if (conv_elements(nt_conv_specs[9]) != output_count) return NT_INVALID_SHAPE;
        if (!trace(value, output_count)) return -101;
        int32_t reordered[output_count];
        for (size_t anchor = 0; anchor < 3; ++anchor)
            for (size_t row = 0; row < 2; ++row)
                for (size_t column = 0; column < 2; ++column)
                    for (size_t component = 0; component < 9; ++component) {
                        size_t destination = ((anchor * 2 + row) * 2 + column) * 9 + component;
                        size_t source = (anchor * 9 + component) * 4 + row * 2 + column;
                        reordered[destination] = value[source];
                        output[destination] = static_cast<float>(value[source]);
                    }
        if (!trace(reordered, output_count)) return -101;
    }
    return NT_SUCCESS;
}
} // namespace

extern "C" void main_graph(float input[4][3][64][64], float output[1][3][2][2][9]) {
    conv_calls = pool_calls = lif_calls = trace_index = 0;
    model_status = NT_SUCCESS;
    std::memset(output, 0, output_count * sizeof(float));
    try {
        model_status = execute_model(&input[0][0][0][0], &output[0][0][0][0][0]);
    } catch (const std::bad_alloc &) {
        model_status = NT_ALLOCATION_FAILED;
    }
    if (model_status)
        std::fprintf(stderr, "SConvLif execution failed with status %d\n", model_status);
}

extern "C" int ninetoothed_sconv_lif_status() { return model_status; }

#ifndef NINETOOTHED_NO_MAIN
namespace {
bool read_input(const char *path, float *values, size_t count) {
    FILE *file = std::fopen(path, "rb");
    if (!file) return false;
    bool ok = std::fread(values, sizeof(float), count, file) == count && std::fgetc(file) == EOF;
    std::fclose(file);
    return ok;
}
} // namespace

int main(int argc, char **argv) {
    if (argc != 3 && argc != 4) {
        std::fprintf(stderr, "Usage: %s input.bin output.bin [trace.bin]\n", argv[0]);
        return 64;
    }
    static float input[4][3][64][64];
    static float output[1][3][2][2][9];
    if (!read_input(argv[1], &input[0][0][0][0], input_count)) return 65;
    if (argc == 4) {
        trace_file = std::fopen(argv[3], "wb");
        if (!trace_file) return 66;
    }
    main_graph(input, output);
    if (trace_file) {
        if (std::fclose(trace_file)) model_status = -101;
        trace_file = nullptr;
    }
    if (model_status) return 67;
    FILE *file = std::fopen(argv[2], "wb");
    if (!file) return 68;
    bool ok = std::fwrite(output, sizeof(float), output_count, file) == output_count;
    if (std::fclose(file)) ok = false;
    if (!ok) return 69;
    std::printf("operator_calls=%zu conv2d=%zu max_pool2d=%zu lif=%zu trace_records=%u\n",
                conv_calls + pool_calls + lif_calls, conv_calls, pool_calls, lif_calls, trace_index);
    return 0;
}
#endif
