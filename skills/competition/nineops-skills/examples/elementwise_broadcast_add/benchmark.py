"""
elementwise_broadcast_add/benchmark.py
Broadcast add kernel 的 benchmark。

对照要素:
  1. 算子: element-wise add with broadcast
  2. 输入规模: 多种 shape 和 dtype 组合
  3. 硬件: 自动检测
  4. 布局: BLOCK_SIZE=1024
  5. Baseline: torch.add (CUDA)
  6. Ninetoothed: 本实现
  7. 实现差异: tile 布局选择
  8. Fallback: N/A
"""

import torch
from time import perf_counter

# 尝试导入 kernel，失败则 fallback 到 demo
try:
    from run import make_broadcast_add_elementwise_1d
except ImportError:
    import sys
    sys.path.insert(0, ".")
    from run import make_broadcast_add_elementwise_1d


def benchmark(shape_a, shape_b, dtype, label, warmup=10, repeats=100):
    """Benchmark kernel vs torch add."""
    x = torch.randn(shape_a, device="cuda", dtype=dtype)
    y = torch.randn(shape_b, device="cuda", dtype=dtype)
    out = torch.empty(shape_a if len(shape_a) >= len(shape_b) else shape_b,
                      device="cuda", dtype=dtype)
    kernel = make_broadcast_add_elementwise_1d()
    blk = 1024 if max(out.shape) <= 1024 else 2048

    # 预热
    for _ in range(warmup):
        kernel(x, y, out, BLOCK_SIZE=blk)
        x + y
    torch.cuda.synchronize()

    # 测量 kernel
    start = perf_counter()
    for _ in range(repeats):
        kernel(x, y, out, BLOCK_SIZE=blk)
    torch.cuda.synchronize()
    kernel_ms = (perf_counter() - start) / repeats * 1000

    # 测量 torch
    start = perf_counter()
    for _ in range(repeats):
        x + y
    torch.cuda.synchronize()
    torch_ms = (perf_counter() - start) / repeats * 1000

    speedup = torch_ms / kernel_ms if kernel_ms > 0 else 0

    print(f"| {label:35s} | {kernel_ms:8.3f} | {torch_ms:8.3f} | {speedup:5.2f}x |")

    return {"label": label, "kernel_ms": kernel_ms, "torch_ms": torch_ms, "speedup": speedup}


def main():
    print("=" * 75)
    print("Broadcast Add Benchmark")
    print("=" * 75)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"GPU: {gpu_name}")
    print(f"CUDA: {torch.version.cuda}")
    print()
    print(f"| {'Config':35s} | {'本实现(ms)':>8s} | {'PyTorch(ms)':>8s} | {'Speedup':>6s} |")
    print(f"|{'-'*35}|{'-'*10}|{'-'*10}|{'-'*8}|")

    results = []

    # FP32, various sizes
    sizes = [(1024,), (4096,), (65536,), (131072,)]
    for s in sizes:
        results.append(benchmark(s, s, torch.float32, f"({s[0]},) fp32"))

    # FP16
    results.append(benchmark((4096,), (4096,), torch.float16, "(4096,) fp16"))
    results.append(benchmark((65536,), (65536,), torch.float16, "(65536,) fp16"))

    # BF16 (if available)
    if hasattr(torch, 'bfloat16'):
        try:
            results.append(benchmark((4096,), (4096,), torch.bfloat16, "(4096,) bf16"))
        except Exception:
            print(f"| {'(4096,) bf16':35s} | {'N/A':>8s} | {'N/A':>8s} | {'N/A':>5s} |")

    # 广播场景
    results.append(benchmark((4096,), (1,), torch.float32, "(4096,) + (1,) scalar"))
    results.append(benchmark((256, 768), (768,), torch.float32, "(256,768) + (768,) vec"))
    results.append(benchmark((4, 128, 256), (256,), torch.float32, "(4,128,256) + (256,)"))
    results.append(benchmark((4, 128, 256), (1,), torch.float32, "(4,128,256) + (1,)"))

    # 非连续场景
    x = torch.randn(768, 256, device="cuda")
    y = torch.randn(256, device="cuda")
    # Transposed
    x_t = x.t().contiguous().t()  # force non-contiguous
    kernel = make_broadcast_add_elementwise_1d()
    out = torch.empty(256, 768, device="cuda")

    warmup, repeats = 10, 100
    for _ in range(warmup):
        kernel(x_t, y, out, BLOCK_SIZE=768)
        x_t + y
    torch.cuda.synchronize()

    start = perf_counter()
    for _ in range(repeats):
        kernel(x_t, y, out, BLOCK_SIZE=768)
    torch.cuda.synchronize()
    k_ms = (perf_counter() - start) / repeats * 1000

    start = perf_counter()
    for _ in range(repeats):
        x_t + y
    torch.cuda.synchronize()
    t_ms = (perf_counter() - start) / repeats * 1000

    print(f"| {'(256,768).T + (768,)':35s} | {k_ms:8.3f} | {t_ms:8.3f} | {t_ms/k_ms:5.2f}x |")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
