"""
performance_regression_case/benchmark.py
Matmul kernel 的性能对比 benchmark: 展示 BLOCK_SIZE 对性能的影响。
"""

import torch
from time import perf_counter

try:
    from run import make_matmul, make_matmul_no_autotune
except ImportError:
    import sys
    sys.path.insert(0, ".")
    from run import make_matmul, make_matmul_no_autotune


def benchmark(label, kernel, a, b, warmup=10, repeats=100):
    c = torch.empty(a.shape[0], b.shape[1], device="cuda")
    for _ in range(warmup):
        kernel(a, b, c, bm=16, bn=16, bk=16)
        a @ b
    torch.cuda.synchronize()

    start = perf_counter()
    for _ in range(repeats):
        kernel(a, b, c, bm=16, bn=16, bk=16)
    torch.cuda.synchronize()
    kernel_ms = (perf_counter() - start) / repeats * 1000

    start = perf_counter()
    for _ in range(repeats):
        a @ b
    torch.cuda.synchronize()
    torch_ms = (perf_counter() - start) / repeats * 1000

    speedup = torch_ms / kernel_ms if kernel_ms > 0 else 0
    print(f"| {label:35s} | {kernel_ms:8.3f} | {torch_ms:8.3f} | {speedup:5.2f}x |")

    return {"label": label, "kernel_ms": kernel_ms, "torch_ms": torch_ms, "speedup": speedup}


def main():
    print("=" * 75)
    print("Matmul Performance Regression Benchmark")
    print("=" * 75)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"GPU: {gpu} | CUDA: {torch.version.cuda}")
    print()

    shapes = [
        ("(512, 512, 512)", 512, 512, 512),
        ("(1024, 1024, 1024)", 1024, 1024, 1024),
        ("(2048, 2048, 2048)", 2048, 2048, 2048),
    ]

    configs = [
        ("BLOCK=16", 16, 16, 16),
        ("BLOCK=32", 32, 32, 32),
        ("BLOCK=64x64x32", 64, 64, 32),
        ("BLOCK=128x128x32", 128, 128, 32),
    ]

    for shape_label, M, N, K in shapes:
        print(f"\nMatrix: {shape_label} | dtype: float32")
        print(f"| {'Config':35s} | {'本实现(ms)':>8s} | {'PyTorch(ms)':>8s} | {'Speedup':>6s} |")
        print(f"|{'-'*35}|{'-'*10}|{'-'*10}|{'-'*8}|")

        a = torch.randn(M, K, device="cuda")
        b = torch.randn(K, N, device="cuda")

        for cfg_label, bm, bn, bk in configs:
            kernel = make_matmul_no_autotune(block_size_m=bm, block_size_n=bn, block_size_k=bk)
            c = torch.empty(M, N, device="cuda")
            kernel(a, b, c, bm=bm, bn=bn, bk=bk)
            result = benchmark(cfg_label, kernel, a, b)

        # 也跑 torch 本身的 benchmark 做参考
        warmup, repeats = 10, 100
        for _ in range(warmup):
            a @ b
        torch.cuda.synchronize()
        start = perf_counter()
        for _ in range(repeats):
            a @ b
        torch.cuda.synchronize()
        torch_ms = (perf_counter() - start) / repeats * 1000
        print(f"| {'PyTorch cuBLAS (baseline)':35s} | {'---':>8s} | {torch_ms:8.3f} | {'1.00x':>6s} |")

    # 关键对比: BLOCK=16 vs BLOCK=128，突出退化
    print(f"\n{'=' * 75}")
    print("性能退化摘要 (BLOCK=16 相比 BLOCK=128 的退化倍数):")
    print(f"{'=' * 75}")
    for shape_label, M, N, K in shapes:
        a = torch.randn(M, K, device="cuda")
        b = torch.randn(K, N, device="cuda")

        k16 = make_matmul_no_autotune(16, 16, 16)
        k128 = make_matmul_no_autotune(128, 128, 32)

        c16 = torch.empty(M, N, device="cuda")
        # 测 k16
        for _ in range(10): k16(a, b, c16, bm=16, bn=16, bk=16)
        torch.cuda.synchronize()
        start = perf_counter()
        for _ in range(100): k16(a, b, c16, bm=16, bn=16, bk=16)
        torch.cuda.synchronize()
        ms16 = (perf_counter() - start) / 100 * 1000

        c128 = torch.empty(M, N, device="cuda")
        # 测 k128
        for _ in range(10): k128(a, b, c128, bm=128, bn=128, bk=32)
        torch.cuda.synchronize()
        start = perf_counter()
        for _ in range(100): k128(a, b, c128, bm=128, bn=128, bk=32)
        torch.cuda.synchronize()
        ms128 = (perf_counter() - start) / 100 * 1000

        ratio = ms16 / ms128 if ms128 > 0 else 0
        print(f"  {shape_label}: BLOCK=16 {ms16:.2f}ms vs BLOCK=128 {ms128:.2f}ms → {ratio:.1f}x slower")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
