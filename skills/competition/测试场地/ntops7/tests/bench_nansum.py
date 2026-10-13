"""Benchmark nansum: ntops vs torch.nansum."""
import torch
import time
from ntops.torch.nansum import nansum as nt_nansum


def bench(fn, *args, warmup=20, trials=100):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(trials):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / trials


def main():
    device = "cuda"
    configs = [
        ("small  dim", (256, 256), 1, 0.1),
        ("medium dim", (1024, 1024), 1, 0.1),
        ("large  dim", (64, 8192), 1, 0.1),
        ("huge   dim", (32, 16384), 1, 0.1),
        ("small  global", (256, 256), None, 0.1),
        ("medium global", (1024, 1024), None, 0.1),
        ("large  global", (64, 8192), None, 0.1),
    ]

    print("=" * 75)
    print(f"{'config':<20s} {'nt(ms)':>10s} {'torch(ms)':>10s} {'speedup':>10s}")
    print("=" * 75)

    for name, shape, dim, nan_ratio in configs:
        for dtype in [torch.float32, torch.float16]:
            x = torch.randn(shape, dtype=dtype, device=device)
            if nan_ratio > 0:
                mask = torch.rand(shape, device=device) < nan_ratio
                x[mask] = float("nan")

            nt_ms = bench(nt_nansum, x, dim)
            torch_ms = bench(torch.nansum, x, dim)
            speedup = torch_ms / nt_ms

            dtype_str = "fp32" if dtype == torch.float32 else "fp16"
            print(f"{name+' '+dtype_str:<20s} {nt_ms:>10.4f} {torch_ms:>10.4f} {speedup:>9.2f}x")

    # All-NaN benchmark
    print("-" * 75)
    for name, shape, dim in [("all_nan dim", (1024, 1024), 1)]:
        for dtype in [torch.float32, torch.float16]:
            x = torch.full(shape, float("nan"), dtype=dtype, device=device)
            nt_ms = bench(nt_nansum, x, dim)
            torch_ms = bench(torch.nansum, x, dim)
            speedup = torch_ms / nt_ms
            dtype_str = "fp32" if dtype == torch.float32 else "fp16"
            print(f"{name+' '+dtype_str:<20s} {nt_ms:>10.4f} {torch_ms:>10.4f} {speedup:>9.2f}x")

    print("=" * 75)


if __name__ == "__main__":
    main()
