import ctypes
import functools
import itertools
import pathlib
import subprocess

import pytest
import torch
import torch.nn.functional as F

import ninetoothed
import ninetoothed.generation
import tests.test_addmm as addmm
import tests.test_attention as attention
import tests.test_conv2d as conv2d
import tests.test_matmul as matmul
from ninetoothed import Tensor
from ninetoothed.aot import _DTYPE_MAPPING
from tests.utils import get_available_devices


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize(
    "dtype, ninetoothed_dtype", ((torch.bfloat16, ninetoothed.bfloat16),)
)
@pytest.mark.parametrize("size", (45327,))
def test_add(size, dtype, device, ninetoothed_dtype):
    def _arrangement(input, other, output):
        def _arrange(tensor):
            return tensor.tile((256,))

        return _arrange(input), _arrange(other), _arrange(output)

    def _application(input, other, output):
        output = input + other  # noqa: F841

    tensors = tuple(Tensor(1, dtype=ninetoothed_dtype) for _ in range(3))
    caller = device
    kernel_name = f"add{_generate_kernel_name_suffix()}"
    output_dir = ninetoothed.generation.CACHE_DIR

    ninetoothed.make(
        _arrangement,
        _application,
        tensors,
        caller=caller,
        kernel_name=kernel_name,
        output_dir=output_dir,
    )

    launch_func = _generate_launch_func(kernel_name=kernel_name, output_dir=output_dir)

    shape = (size,)

    input = torch.randn(shape, dtype=dtype, device=device)
    other = torch.randn(shape, dtype=dtype, device=device)
    output = torch.empty_like(input)

    _run_launch_func(launch_func, input, other, output)

    expected = torch.add(input, other)

    assert torch.allclose(output, expected)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize(
    "dtype, ninetoothed_dtype, atol", ((torch.float16, ninetoothed.float16, 0.075),)
)
@pytest.mark.parametrize("k", (512,))
@pytest.mark.parametrize("n", (512,))
@pytest.mark.parametrize("m", (512,))
def test_addmm(m, n, k, dtype, device, ninetoothed_dtype, atol):
    arrangement = functools.partial(
        addmm.arrangement, BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64
    )
    application = addmm.application
    tensors = tuple(
        Tensor(ndim, dtype=ninetoothed_dtype) for ndim in (2, 2, 2, 0, 0, 2)
    )
    caller = device
    kernel_name = f"addmm{_generate_kernel_name_suffix()}"
    output_dir = ninetoothed.generation.CACHE_DIR

    ninetoothed.make(
        arrangement,
        application,
        tensors,
        caller=caller,
        kernel_name=kernel_name,
        output_dir=output_dir,
    )

    launch_func = _generate_launch_func(kernel_name=kernel_name, output_dir=output_dir)

    input = torch.randn((m, n), dtype=dtype, device=device)
    mat1 = torch.randn((m, k), dtype=dtype, device=device)
    mat2 = torch.randn((k, n), dtype=dtype, device=device)
    beta = torch.randn((), dtype=dtype)
    alpha = torch.randn((), dtype=dtype)
    output = torch.empty(
        (mat1.shape[0], mat2.shape[1]), dtype=mat1.dtype, device=mat1.device
    )

    _run_launch_func(launch_func, input, mat1, mat2, beta, alpha, output)

    expected = torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha)

    assert torch.allclose(output, expected, atol=atol)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize(
    "dtype, ninetoothed_dtype, rtol, atol",
    ((torch.float16, ninetoothed.float16, 0.01, 0.01),),
)
@pytest.mark.parametrize("emb_dim", (64,))
@pytest.mark.parametrize("seq_len", (1024,))
@pytest.mark.parametrize("num_heads", (4,))
@pytest.mark.parametrize("batch_size", (2,))
def test_attention(
    batch_size,
    num_heads,
    seq_len,
    emb_dim,
    dtype,
    device,
    ninetoothed_dtype,
    rtol,
    atol,
):
    arrangement = functools.partial(
        attention.arrangement, BLOCK_SIZE_M=64, BLOCK_SIZE_N=64
    )
    application = attention.application
    query_, key_, value_, output_ = tuple(
        Tensor(4, dtype=ninetoothed_dtype) for _ in range(4)
    )
    for tensor in (query_, key_, value_, output_):
        tensor.shape = tensor.shape[:-1] + (emb_dim,)
    is_causal_ = Tensor(0, constexpr=True, value=1)
    tensors = (query_, key_, value_, is_causal_, output_)
    caller = device
    kernel_name = f"attention{_generate_kernel_name_suffix()}"
    output_dir = ninetoothed.generation.CACHE_DIR

    ninetoothed.make(
        arrangement,
        application,
        tensors,
        caller=caller,
        kernel_name=kernel_name,
        output_dir=output_dir,
    )

    launch_func = _generate_launch_func(kernel_name=kernel_name, output_dir=output_dir)

    shape = (batch_size, num_heads, seq_len, emb_dim)

    query = torch.randn(shape, dtype=dtype, device=device)
    key = torch.randn(shape, dtype=dtype, device=device)
    value = torch.randn(shape, dtype=dtype, device=device)
    is_causal = torch.tensor(True)
    output = torch.empty(shape, dtype=dtype, device=device)

    _run_launch_func(launch_func, query, key, value, is_causal, output)

    expected = F.scaled_dot_product_attention(
        query, key, value, is_causal=True, scale=1
    )

    assert torch.allclose(output, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize(
    "dtype, ninetoothed_dtype", ((torch.float16, ninetoothed.float16),)
)
@pytest.mark.parametrize("k", (512,))
@pytest.mark.parametrize("n", (512,))
@pytest.mark.parametrize("m", (512,))
def test_matmul(m, n, k, dtype, device, ninetoothed_dtype):
    arrangement = functools.partial(
        matmul.arrangement, BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64
    )
    application = matmul.application
    tensors = tuple(Tensor(2, dtype=ninetoothed_dtype) for _ in range(3))
    caller = device
    kernel_name = f"matmul{_generate_kernel_name_suffix()}"
    output_dir = ninetoothed.generation.CACHE_DIR

    ninetoothed.make(
        arrangement,
        application,
        tensors,
        caller=caller,
        kernel_name=kernel_name,
        output_dir=output_dir,
    )

    launch_func = _generate_launch_func(kernel_name=kernel_name, output_dir=output_dir)

    lhs = torch.randn((m, k), dtype=dtype, device=device)
    rhs = torch.randn((k, n), dtype=dtype, device=device)
    output = torch.empty((lhs.shape[0], rhs.shape[1]), dtype=dtype, device=device)

    _run_launch_func(launch_func, lhs, rhs, output)

    expected = torch.matmul(lhs, rhs)

    assert torch.allclose(output, expected)


@pytest.mark.parametrize("constexpr_shapes", (False, True))
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize(
    "dtype, ninetoothed_dtype, rtol, atol",
    ((torch.float16, ninetoothed.float16, 0.001, 0.001),),
)
@pytest.mark.parametrize("s", (3,))
@pytest.mark.parametrize("r", (3,))
@pytest.mark.parametrize("k", (512,))
@pytest.mark.parametrize("w", (16,))
@pytest.mark.parametrize("h", (16,))
@pytest.mark.parametrize("c", (64,))
@pytest.mark.parametrize("n", (4,))
@pytest.mark.parametrize("test_build", (False, True))
def test_conv2d(
    test_build,
    n,
    c,
    h,
    w,
    k,
    r,
    s,
    dtype,
    device,
    constexpr_shapes,
    ninetoothed_dtype,
    rtol,
    atol,
):
    premake = functools.partial(
        conv2d.premake, dtype=ninetoothed_dtype, constexpr_shapes=constexpr_shapes
    )

    caller = device
    kernel_name = f"conv2d{_generate_kernel_name_suffix()}"
    output_dir = ninetoothed.generation.CACHE_DIR

    if test_build:
        configs = (
            ((), {"block_size_m": 64, "block_size_n": 64, "block_size_k": 64}, {}),
            ((), {"block_size_m": 128, "block_size_n": 32, "block_size_k": 64}, {}),
        )

        ninetoothed.build(
            premake,
            configs,
            caller=caller,
            kernel_name=kernel_name,
            output_dir=output_dir,
        )
    else:
        arrangement, application, tensors = premake()

        ninetoothed.make(
            arrangement,
            application,
            tensors,
            caller=caller,
            kernel_name=kernel_name,
            output_dir=output_dir,
        )

    launch_func = _generate_launch_func(kernel_name=kernel_name, output_dir=output_dir)

    p = h - r + 1
    q = w - s + 1

    input = torch.randn(n, c, h, w, dtype=dtype, device=device)
    filter = torch.randn(k, c, r, s, dtype=dtype, device=device)
    output = torch.empty(n, k, p, q, dtype=dtype, device=device)

    if test_build:
        config = (
            tuple(_DTYPE_MAPPING.keys()).index(ninetoothed_dtype),
            constexpr_shapes,
        ) + tuple(configs[0][1].values())
    else:
        config = ()

    _run_launch_func(launch_func, input, filter, output, *config)

    expected = F.conv2d(input, filter)

    assert torch.allclose(output, expected, rtol=rtol, atol=atol)


class _ArgumentTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("shape", ctypes.POINTER(ctypes.c_uint64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
    ]

    @staticmethod
    def from_torch_tensor(tensor):
        data = ctypes.c_void_p(tensor.data_ptr())
        shape = (ctypes.c_uint64 * len(tensor.shape))(*tensor.shape)
        strides = (ctypes.c_int64 * len(tensor.stride()))(*tensor.stride())

        return _ArgumentTensor(data, shape, strides)


def _run_launch_func(launch_func, *args, **kwargs):
    stream = torch.cuda.Stream()

    arguments = tuple(
        _ArgumentTensor.from_torch_tensor(arg) if isinstance(arg, torch.Tensor) else arg
        for arg in itertools.chain(args, kwargs.values())
    )

    with torch.cuda.stream(stream):
        launch_func(ctypes.c_void_p(stream.cuda_stream), *arguments)


def _generate_launch_func(kernel_name, output_dir):
    output_dir = pathlib.Path(output_dir)

    _compile_library(kernel_name, output_dir)
    library = _load_library(kernel_name, output_dir)
    launch_func_name = f"launch_{kernel_name}"
    launch_func = getattr(library, launch_func_name)

    return launch_func


def _compile_library(kernel_name, output_dir):
    command = [
        "nvcc",
        "-shared",
        "-Xcompiler",
        "-fPIC",
        "-lcuda",
        "-o",
        output_dir / f"{kernel_name}.so",
    ] + list(output_dir.glob(f"{kernel_name}*.cpp"))

    subprocess.run(command, check=True)


def _load_library(kernel_name, kernel_dir):
    return ctypes.CDLL(kernel_dir / f"{kernel_name}.so")


def _generate_kernel_name_suffix():
    count = _generate_kernel_name_suffix._kernel_count
    _generate_kernel_name_suffix._kernel_count += 1

    return f"_{count}"


_generate_kernel_name_suffix._kernel_count = 0
