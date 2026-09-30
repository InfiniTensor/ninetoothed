import functools
import shutil
import subprocess

import pytest
import torch

import ninetoothed
from ninetoothed import Tensor
from ninetoothed.backends.toolchain import find_nvcc


def _arrangement(input, other, alpha, output, block_size):
    return (
        input.tile((block_size,)),
        other.tile((block_size,)),
        alpha,
        output.tile((block_size,)),
    )


def _application(input, other, alpha, output):
    output = input + alpha * other  # noqa: F841


def _premake(dtype, block_size):
    tensors = (
        Tensor(1, dtype=dtype, name="input"),
        Tensor(1, dtype=dtype, name="other"),
        Tensor(0, dtype=ninetoothed.float64, name="alpha"),
        Tensor(1, dtype=dtype, name="output"),
    )

    return functools.partial(_arrangement, block_size=block_size), _application, tensors


@pytest.mark.parametrize("single_config", (False, True))
def test_cpp_build_runs_without_python_after_relocation(
    tmp_path, single_config, monkeypatch
):
    if not torch.cuda.is_available():
        pytest.skip("C++ AOT execution requires CUDA")

    configs = tuple(
        ((), {"dtype": dtype, "block_size": block}, {})
        for dtype in (ninetoothed.float32, ninetoothed.float64)
        for block in (64, 128)
    )

    if single_config:
        configs = configs[:1]

    output_dir = tmp_path / "generated"
    ninetoothed.build(
        _premake,
        configs,
        meta_parameters=("block_size",),
        kernel_name="cpp_add",
        output_dir=tmp_path / "python-only",
        export_cpp=False,
    )

    from ninetoothed.backends.materializers import triton as materializer

    def fail_recompile(*args, **kwargs):
        pytest.fail(
            "A cached build should republish its C++ sources without recompiling."
        )

    monkeypatch.setattr(materializer, "_compile_aot_library", fail_recompile)
    ninetoothed.build(
        _premake,
        configs,
        meta_parameters=("block_size",),
        kernel_name="cpp_add",
        output_dir=output_dir,
        export_cpp=True,
    )
    relocated = tmp_path / "relocated"
    relocated.mkdir()

    for path in output_dir.iterdir():
        if path.suffix in {".h", ".cpp"}:
            shutil.copy2(path, relocated / path.name)

    shutil.rmtree(output_dir)
    main = relocated / "main.cpp"
    main.write_text(_CPP_MAIN, encoding="utf-8")
    executable = relocated / "run"
    subprocess.run(
        [
            find_nvcc(),
            "-std=c++17",
            "-Xcompiler",
            "-pthread",
            *(str(path) for path in sorted(relocated.glob("*.cpp"))),
            "-lcuda",
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [str(executable), "single" if single_config else "multiple"],
        check=True,
        capture_output=True,
        text=True,
        cwd=relocated,
    )


def _named_premake(label):
    return _premake(ninetoothed.float32, 64)


@pytest.mark.parametrize("export_cpp", (None, False, True))
def test_python_build_keeps_string_configuration_keys(tmp_path, export_cpp):
    if not torch.cuda.is_available():
        pytest.skip("Triton AOT execution requires CUDA")

    options = dict(kernel_name="named_add", output_dir=tmp_path, export_cpp=export_cpp)

    if export_cpp:
        with pytest.raises(ValueError, match=r"C\+\+ export requires"):
            ninetoothed.build(_named_premake, [(("fast",), {}, {})], **options)
        return

    kernel = ninetoothed.build(_named_premake, [(("fast",), {}, {})], **options)
    x = torch.randn(257, device="cuda")
    other = torch.randn_like(x)
    output = torch.empty_like(x)
    kernel(x, other, 0.5, output, "fast")
    torch.testing.assert_close(output, x + 0.5 * other)
    assert not list(tmp_path.glob("*.cpp"))


_CPP_MAIN = r"""
#include "cpp_add.h"
#include <cuda.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

void check(CUresult result) {
    if (result != CUDA_SUCCESS) throw std::runtime_error(std::to_string(result));
}

template <typename T>
void run(int dtype, CUstream stream) {
    const uint64_t size = 257;
    std::vector<T> input(size * 2), other(size), output(size * 3);
    for (uint64_t i = 0; i < size; ++i) {
        input[i * 2] = static_cast<T>(i);
        other[i] = static_cast<T>(i + 2);
    }
    CUdeviceptr x, y, z;
    check(cuMemAlloc(&x, input.size() * sizeof(T)));
    check(cuMemAlloc(&y, other.size() * sizeof(T)));
    check(cuMemAlloc(&z, output.size() * sizeof(T)));
    check(cuMemcpyHtoD(x, input.data(), input.size() * sizeof(T)));
    check(cuMemcpyHtoD(y, other.data(), other.size() * sizeof(T)));
    uint64_t shape[] = {size};
    int64_t x_stride[] = {2}, y_stride[] = {1}, z_stride[] = {3};
    double alpha = 0.5;
    NineToothedTensor a{reinterpret_cast<void *>(x), shape, x_stride};
    NineToothedTensor b{reinterpret_cast<void *>(y), shape, y_stride};
    NineToothedTensor c{&alpha, nullptr, nullptr};
    NineToothedTensor out{reinterpret_cast<void *>(z), shape, z_stride};
    check(static_cast<CUresult>(launch_cpp_add(stream, a, b, c, out, dtype)));
    check(cuStreamSynchronize(stream));
    check(cuMemcpyDtoH(output.data(), z, output.size() * sizeof(T)));
    for (uint64_t i = 0; i < size; ++i) {
        if (output[i * 3] != input[i * 2] + alpha * other[i])
            throw std::runtime_error("Strided result mismatch");
    }
    if (launch_cpp_add(stream, a, b, c, out, -1) == 0)
        throw std::runtime_error("Unsupported configuration accepted");
    shape[0] = UINT64_MAX;
    if (launch_cpp_add(stream, a, b, c, out, dtype) == 0)
        throw std::runtime_error("Overflowing shape accepted");
    shape[0] = 0;
    a.data = b.data = out.data = nullptr;
    check(static_cast<CUresult>(launch_cpp_add(stream, a, b, c, out, dtype)));
    check(cuMemFree(x));
    check(cuMemFree(y));
    check(cuMemFree(z));
}

int main(int argc, char **argv) {
    check(cuInit(0));
    CUdevice device;
    CUcontext context;
    check(cuDeviceGet(&device, 0));
    check(cuDevicePrimaryCtxRetain(&context, device));
    check(cuCtxSetCurrent(context));
    CUstream stream;
    check(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    run<float>(NINETOOTHED_FLOAT32, stream);
    if (argc > 1 && std::string(argv[1]) == "multiple")
        run<double>(NINETOOTHED_FLOAT64, stream);
    run<float>(NINETOOTHED_FLOAT32, stream);
    check(cuStreamDestroy(stream));
    check(cuDevicePrimaryCtxRelease(device));
}
"""
