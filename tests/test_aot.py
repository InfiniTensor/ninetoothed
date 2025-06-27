import ctypes
import functools
import subprocess

import torch
import torch.nn.functional as F

import ninetoothed
import ninetoothed.generation
import tests.test_addmm as addmm
import tests.test_attention as attention
import tests.test_conv2d as conv2d
import tests.test_matmul as matmul
from ninetoothed import Tensor
from tests.skippers import skip_if_cuda_not_available


@skip_if_cuda_not_available
class TestCUDA:
    @classmethod
    def setup_class(cls):
        torch.manual_seed(0)

    def test_addmm(self):
        arrangement = functools.partial(
            addmm.arrangement, BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64
        )
        application = addmm.application
        tensors = tuple(
            Tensor(ndim, dtype=ninetoothed.float16) for ndim in (2, 2, 2, 0, 0, 2)
        )
        caller = "cuda"
        kernel_name = "addmm"
        output_dir = ninetoothed.generation.CACHE_DIR

        launch_func = _generate_launch_func(
            arrangement,
            application,
            tensors,
            caller=caller,
            kernel_name=kernel_name,
            output_dir=output_dir,
        )

        shape = (512, 512)
        dtype = torch.float16
        device = caller

        input = torch.randn(shape, dtype=dtype, device=device)
        mat1 = torch.randn(shape, dtype=dtype, device=device)
        mat2 = torch.randn(shape, dtype=dtype, device=device)
        beta = torch.randn((), dtype=dtype)
        alpha = torch.randn((), dtype=dtype)
        output = torch.empty(
            (mat1.shape[0], mat2.shape[1]), dtype=mat1.dtype, device=mat1.device
        )

        _run_launch_func(launch_func, input, mat1, mat2, beta, alpha, output)

        assert torch.allclose(
            output, torch.addmm(input, mat1, mat2, beta=beta, alpha=alpha), atol=0.075
        )

    def test_attention(self):
        emb_dim = 64

        arrangement = functools.partial(
            attention.arrangement, BLOCK_SIZE_M=64, BLOCK_SIZE_N=64
        )
        application = attention.application
        query_, key_, value_, output_ = tuple(
            Tensor(4, dtype=ninetoothed.float16) for _ in range(4)
        )
        for tensor in (query_, key_, value_, output_):
            tensor.shape = tensor.shape[:-1] + (emb_dim,)
        is_causal_ = Tensor(0, constexpr=True, value=1)
        tensors = (query_, key_, value_, is_causal_, output_)
        caller = "cuda"
        kernel_name = "attention"
        output_dir = ninetoothed.generation.CACHE_DIR

        launch_func = _generate_launch_func(
            arrangement,
            application,
            tensors,
            caller=caller,
            kernel_name=kernel_name,
            output_dir=output_dir,
        )

        shape = (2, 4, 1024, emb_dim)
        dtype = torch.float16
        device = caller

        query = torch.randn(shape, dtype=dtype, device=device)
        key = torch.randn(shape, dtype=dtype, device=device)
        value = torch.randn(shape, dtype=dtype, device=device)
        is_causal = torch.tensor(True)
        output = torch.empty(shape, dtype=dtype, device=device)

        _run_launch_func(launch_func, query, key, value, is_causal, output)

        assert torch.allclose(
            output,
            F.scaled_dot_product_attention(query, key, value, is_causal=True, scale=1),
            atol=0.01,
        )

    def test_matmul(self):
        arrangement = functools.partial(
            matmul.arrangement, BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64
        )
        application = matmul.application
        tensors = tuple(Tensor(2, dtype=ninetoothed.float16) for _ in range(3))
        caller = "cuda"
        kernel_name = "matmul"
        output_dir = ninetoothed.generation.CACHE_DIR

        launch_func = _generate_launch_func(
            arrangement,
            application,
            tensors,
            caller=caller,
            kernel_name=kernel_name,
            output_dir=output_dir,
        )

        shape = (512, 512)
        dtype = torch.float16
        device = caller

        lhs = torch.randn(shape, dtype=dtype, device=device)
        rhs = torch.randn(shape, dtype=dtype, device=device)
        output = torch.empty((lhs.shape[0], rhs.shape[1]), dtype=dtype, device=device)

        _run_launch_func(launch_func, lhs, rhs, output)

        assert torch.allclose(output, torch.matmul(lhs, rhs))

    def test_conv2d(self):
        arrangement = functools.partial(
            conv2d.arrangement, BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64
        )
        application = matmul.application
        tensors = tuple(Tensor(4, dtype=ninetoothed.float16) for _ in range(3))
        caller = "cuda"
        kernel_name = "conv2d"
        output_dir = ninetoothed.generation.CACHE_DIR

        launch_func = _generate_launch_func(
            arrangement,
            application,
            tensors,
            caller=caller,
            kernel_name=kernel_name,
            output_dir=output_dir,
        )

        n, c, h, w = 4, 64, 16, 16
        k, _, r, s = 512, c, 3, 3
        p = h - r + 1
        q = w - s + 1
        dtype = torch.float16
        device = caller

        input = torch.randn(n, c, h, w, dtype=dtype, device=device)
        filter = torch.randn(k, c, r, s, dtype=dtype, device=device)
        output = torch.empty(n, k, p, q, dtype=dtype, device=device)

        _run_launch_func(launch_func, input, filter, output)

        assert torch.allclose(output, F.conv2d(input, filter), atol=0.001, rtol=0.001)


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


def _run_launch_func(launch_func, *tensors):
    stream = torch.cuda.Stream()

    arg_tensors = tuple(_ArgumentTensor.from_torch_tensor(tensor) for tensor in tensors)

    with torch.cuda.stream(stream):
        launch_func(ctypes.c_void_p(stream.cuda_stream), *arg_tensors)

    stream.synchronize()


def _generate_launch_func(
    arrangement, application, tensors, caller, kernel_name, output_dir
):
    ninetoothed.make(
        arrangement,
        application,
        tensors,
        caller=caller,
        kernel_name=kernel_name,
        output_dir=output_dir,
    )

    _compile_library(kernel_name, output_dir)
    library = _load_library(kernel_name, output_dir)
    launch_func_name = f"launch_{kernel_name}"
    launch_func = getattr(library, launch_func_name)
    launch_func.argtypes = (ctypes.c_void_p,) + tuple(_ArgumentTensor for _ in tensors)
    launch_func.restype = ctypes.c_int

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
        output_dir / f"{kernel_name}.c",
    ]

    subprocess.run(command, check=True)


def _load_library(kernel_name, kernel_dir):
    return ctypes.CDLL(kernel_dir / f"{kernel_name}.so")
