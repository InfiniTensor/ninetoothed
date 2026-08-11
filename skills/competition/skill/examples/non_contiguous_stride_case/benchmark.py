"""
non_contiguous_stride_case/benchmark.py
Non-contiguous tensor 的 benchmark: 对比 contiguous vs strided 加法的性能差异。
"""

import torch
from time import perf_counter

try:
    from run import make_add_kernel, make_2d_add_kernel
except ImportError:
    import sys
    sys.path.insert(0, ".")
    from run import make_add_kernel, make_2d_add_kernel


def benchmark(name, x, y=None, warmup=10, repeats=100):
    if y is None:
        y = torch.randn(x.shape[-1], device="cuda")
    kernel = make_add_kernel() if x.ndim == 1 else make_2d_add_kernel()
    out = torch.empty_like(x)
    BLOCK_SIZE = 1024

    for _ in range(warmup):
        kernel(x, y, out, BLOCK_SIZE=BLOCK_SIZE)
        x + y
    torch.cuda.synchronize()

    start = perf_counter()
    for _ in range(repeats):
        kernel(x, y, out, BLOCK_SIZE=BLOCK_SIZE)
    torch.cuda.synchronize()
    kernel_ms = (perf_counter() - start) / repeats * 1000

    start = perf_counter()
    for _ in range(repeats):
        x + y
    torch.cuda.synchronize()
    torch_ms = (perf_counter() - start) / repeats * 1000

    speedup = torch_ms / kernel_ms if kernel_ms > 0 else 0
    print(f"| {name:35s} | {kernel_ms:8.3f} | {torch_ms:8.3f} | {speedup:5.2f}x |")

    return {"name": name, "kernel_ms": kernel_ms, "torch_ms": torch_ms, "speedup": speedup}


def main():
    print("=" * 75)
    print("Non-Contiguous Add Benchmark")
    print("=" * 75)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"GPU: {gpu} | CUDA: {torch.version.cuda}")
    print()
    print(f"| {'Config':35s} | {'本实现(ms)':>8s} | {'PyTorch(ms)':>8s} | {'Speedup':>6s} |")
    print(f"|{'-'*35}|{'-'*10}|{'-'*10}|{'-'*8}|")

    M, N = 256, 1024

    baseline = torch.randn(M, N, device="cuda")
    y = torch.randn(N, device="cuda")

    benchmark(f"contiguous ({M}, {N})", baseline, y)
    benchmark(f"transposed ({N}, {M}).t()", torch.randn(N, M, device="cuda").t(), y)
    benchmark(f"sliced rows [::2]", torch.randn(M*2, N, device="cuda")[::2, :], y)
    benchmark(f"sliced cols [::3]", torch.randn(M, N*3, device="cuda")[:, ::3], y)
    benchmark(f"both sliced", torch.randn(M*2, N*3, device="cuda")[::2, ::3], y)

    # 广播场景
    y_scalar = torch.randn(1, device="cuda")
    benchmark(f"contiguous + scalar (1,)", baseline, y_scalar)
    benchmark(f"transposed + scalar (1,)", torch.randn(N, M, device="cuda").t(), y_scalar)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
