"""
reduction_softmax/benchmark.py
Softmax kernel 的 benchmark。
"""

import torch
from time import perf_counter

try:
    from run import make_softmax, make_softmax_no_autotune
except ImportError:
    import sys
    sys.path.insert(0, ".")
    from run import make_softmax, make_softmax_no_autotune


def benchmark(shape, dtype, label, warmup=10, repeats=100):
    x = torch.randn(shape, device="cuda", dtype=dtype)
    out = torch.empty_like(x)
    dim = -1

    # 选择 kernel: 如果列数 <= 2048 用 autotune，否则用足够大的 BLOCK_SIZE
    cols = shape[-1]
    if cols <= 2048:
        kernel = make_softmax()
        kernel_kwargs = {"BLOCK_SIZE": cols}
    else:
        bs = 1
        while bs < cols:
            bs *= 2
        kernel = make_softmax_no_autotune()
        kernel_kwargs = {"BS": bs}

    # 预热
    for _ in range(warmup):
        kernel(x, out, **kernel_kwargs)
        torch.softmax(x, dim=dim)
    torch.cuda.synchronize()

    # 测量 kernel
    start = perf_counter()
    for _ in range(repeats):
        kernel(x, out, **kernel_kwargs)
    torch.cuda.synchronize()
    kernel_ms = (perf_counter() - start) / repeats * 1000

    # 测量 torch
    start = perf_counter()
    for _ in range(repeats):
        torch.softmax(x, dim=dim)
    torch.cuda.synchronize()
    torch_ms = (perf_counter() - start) / repeats * 1000

    speedup = torch_ms / kernel_ms if kernel_ms > 0 else 0
    print(f"| {label:35s} | {kernel_ms:8.3f} | {torch_ms:8.3f} | {speedup:5.2f}x |")

    return {"label": label, "kernel_ms": kernel_ms, "torch_ms": torch_ms, "speedup": speedup}


def main():
    print("=" * 75)
    print("Softmax Benchmark")
    print("=" * 75)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"GPU: {gpu_name}")
    print(f"CUDA: {torch.version.cuda}")
    print()
    print(f"| {'Config':35s} | {'本实现(ms)':>8s} | {'PyTorch(ms)':>8s} | {'Speedup':>6s} |")
    print(f"|{'-'*35}|{'-'*10}|{'-'*10}|{'-'*8}|")

    # Small: few rows, small cols
    benchmark((4, 1024), torch.float32, "(4, 1024) fp32")
    benchmark((4, 4096), torch.float32, "(4, 4096) fp32")

    # Medium
    benchmark((128, 1024), torch.float32, "(128, 1024) fp32")
    benchmark((128, 4096), torch.float32, "(128, 4096) fp32")

    # Large
    benchmark((4, 65536), torch.float32, "(4, 65536) fp32")
    benchmark((4, 131072), torch.float32, "(4, 131072) fp32")

    # FP16
    benchmark((128, 1024), torch.float16, "(128, 1024) fp16")
    benchmark((128, 4096), torch.float16, "(128, 4096) fp16")

    # Non-contiguous
    x_big = torch.randn(8, 2048, device="cuda", dtype=torch.float32)
    x_nc = x_big[:, ::2]  # (8, 1024), non-contiguous
    out_nc = torch.empty(8, 1024, device="cuda", dtype=torch.float32)
    kernel = make_softmax()
    warmup, repeats = 10, 100
    for _ in range(warmup):
        kernel(x_nc, out_nc, BLOCK_SIZE=1024)
        torch.softmax(x_nc, dim=-1)
    torch.cuda.synchronize()

    start = perf_counter()
    for _ in range(repeats):
        kernel(x_nc, out_nc, BLOCK_SIZE=1024)
    torch.cuda.synchronize()
    k_ms = (perf_counter() - start) / repeats * 1000

    start = perf_counter()
    for _ in range(repeats):
        torch.softmax(x_nc, dim=-1)
    torch.cuda.synchronize()
    t_ms = (perf_counter() - start) / repeats * 1000

    print(f"| {'(8,2048)[:,::2] non-contiguous':35s} | {k_ms:8.3f} | {t_ms:8.3f} | {t_ms/k_ms:5.2f}x |")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
