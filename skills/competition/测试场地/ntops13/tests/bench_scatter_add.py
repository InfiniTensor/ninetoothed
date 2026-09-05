"""
Benchmark: scatter_add vs torch.scatter_add on MetaX C500.

For each (shape, dim, dtype) we measure:
  - GPU time (CUDA events) -- excludes Python dispatch overhead
  - E2E time (wall clock)  -- includes all Python overhead

Reports per-case:
  - nt_gpu_ms, torch_gpu_ms, speedup = torch_gpu / nt_gpu
  - MERE (mean elementwise relative error, fp32 reference)
  - MARE (max absolute relative error)

Also includes a dedicated "high-conflict" case where the index value
range is far smaller than the number of src elements, so atomic_add
gets hit very heavily.
"""

import sys
import time

import torch

sys.path.insert(0, "/data/ntops13/src")
import ntops  # noqa: E402


def _bench(fn, *args, warmup=30, iters=120):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    gpu_ms = start.elapsed_time(end) / iters

    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    e2e_ms = (time.perf_counter() - t0) * 1000 / iters

    return gpu_ms, e2e_ms


def _errors(got, ref, dtype):
    diff = (got.double() - ref.double()).abs()
    denom = ref.double().abs().maximum(torch.tensor(1.0, dtype=torch.float64))
    rel = diff / denom
    return float(diff.mean().item()), float(diff.max().item()), float(rel.mean().item()), float(rel.max().item())


def _make_case(shape, dim, dtype, device, dup_ratio=0.4):
    self_t = torch.randn(shape, dtype=dtype, device=device)
    src_t = torch.randn(shape, dtype=dtype, device=device)
    dim_len = self_t.shape[dim]
    numel = 1
    for s in shape:
        numel *= s
    flat = torch.arange(numel, device=device) % max(1, dim_len // 2)
    if dup_ratio > 0:
        n_dup = int(numel * dup_ratio)
        flat[:n_dup] = torch.randint(0, dim_len, (n_dup,), device=device)
    index = flat.view(shape)
    return self_t, src_t, index


def run_bench():
    device = "cuda"
    cases = [
        # (shape, dim, dtype, label, dup_ratio)
        ((1024,), 0, torch.float32, "1D 1K fp32 dup30", 0.4),
        ((1024,), 0, torch.float16, "1D 1K fp16 dup30", 0.4),
        ((256, 256), 0, torch.float32, "2D 256x256 dim=0 fp32 dup30", 0.4),
        ((256, 256), 1, torch.float16, "2D 256x256 dim=1 fp16 dup30", 0.4),
        ((64, 64, 32), 0, torch.float32, "3D 64x64x32 dim=0 fp32 dup30", 0.4),
        ((64, 64, 32), 1, torch.float16, "3D 64x64x32 dim=1 fp16 dup30", 0.4),
        ((32, 16, 8, 4), 2, torch.float32, "4D 32x16x8x4 dim=2 fp32 dup30", 0.4),
        # High conflict: index range << src elements
        ((8192,), 0, torch.float32, "1D 8K HIGH-CONFLICT (range=4) fp32", 0.99),
        ((2048, 512), 1, torch.float32, "2D 2048x512 HIGH-CONFLICT fp32", 0.99),
        ((128, 128, 16), 0, torch.float16, "3D 128x128x16 HIGH-CONFLICT fp16", 0.99),
        # All-zero index: maximum possible race
        ((4096,), 0, torch.float32, "1D 4K all-zero-index (max race) fp32", 1.0),
    ]

    print("=" * 110)
    print(f"{'case':<42s} | {'nt_gpu':>9s} | {'torch_gpu':>9s} | {'speedup':>8s} | "
          f"{'MAE':>10s} | {'MaxAE':>10s}")
    print("-" * 110)

    for shape, dim, dtype, label, dup in cases:
        self_t, src_t, index = _make_case(shape, dim, dtype, device, dup_ratio=dup)

        if dup >= 0.99:
            # Override index to very small range / all-zero
            if "all-zero" in label:
                index = torch.zeros_like(index)
            else:
                index = torch.randint(0, 4, index.shape, dtype=torch.long, device=device)

        def nt_fn(s=self_t, d=dim, i=index, v=src_t):
            return ntops.torch.scatter_add(s, d, i, v)

        def torch_fn(s=self_t, d=dim, i=index, v=src_t):
            return torch.scatter_add(s, d, i, v)

        nt_gpu, nt_e2e = _bench(nt_fn)
        torch_gpu, torch_e2e = _bench(torch_fn)
        speedup = torch_gpu / nt_gpu if nt_gpu > 0 else float("inf")

        got = nt_fn()
        ref = torch_fn()
        mae, maxae, _mere, _mare = _errors(got, ref, dtype)

        print(f"{label:<42s} | {nt_gpu:9.4f} | {torch_gpu:9.4f} | {speedup:7.2f}x | "
              f"{mae:10.3e} | {maxae:10.3e}")


if __name__ == "__main__":
    run_bench()
