"""Standalone addcmul verification/benchmark script."""
import time
import torch
import sys

sys.path.insert(0, '/data/ntops5/src')

import ntops


THRESHOLDS = {
    torch.float32: 1.22e-4,
    torch.float16: 9.77e-4,
}


def run_one(name, shape, dtype, value=1.0, repeats=5):
    device = "cuda"
    torch.manual_seed(0)
    input = torch.randn(shape, dtype=dtype, device=device)
    tensor1 = torch.randn(shape, dtype=dtype, device=device)
    tensor2 = torch.randn(shape, dtype=dtype, device=device)

    # warm-up (compile)
    small = input[:4, :4]
    _ = ntops.torch.addcmul(small, tensor1[:4, :4], tensor2[:4, :4], value=value)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = ntops.torch.addcmul(input, tensor1, tensor2, value=value)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    elapsed_ms = sorted(times)[len(times) // 2] * 1000.0

    ref = torch.addcmul(input, tensor1, tensor2, value=value)

    diff = (out - ref).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    ref_abs = ref.abs().mean().item()
    mare = max_abs / (ref_abs + 1e-12)
    mere = mean_abs / (ref_abs + 1e-12)
    atol = THRESHOLDS[dtype]
    passed = mere < atol and mare < 10 * atol
    print(
        f"[{name:24s}] shape={str(shape):14s} dtype={str(dtype).split('.')[-1]:8s} "
        f"MERE={mere:.3e} MARE={mare:.3e} max_abs={max_abs:.4f} "
        f"time={elapsed_ms:.2f}ms  {'PASS' if passed else 'FAIL'}"
    )
    return passed, mere, mare, max_abs, elapsed_ms


def main():
    cases = [
        ("test_addcmul_basic", (128, 128), torch.float32, 1.0),
        ("test_addcmul_basic", (64, 64), torch.float32, 1.0),
        ("test_addcmul_fp16", (1024, 1024), torch.float16, 10.0),
        ("test_addcmul_large", (4096, 4096), torch.float16, 3.0),
    ]
    results = []
    for name, shape, dtype, value in cases:
        results.append((name, *run_one(name, shape, dtype, value=value)))

    print("\n=== summary ===")
    for name, passed, mere, mare, max_abs, ms in results:
        print(f"{name:24s} {'PASS' if passed else 'FAIL':4s}  MERE={mere:.3e}  MARE={mare:.3e}  {ms:.2f}ms")


if __name__ == "__main__":
    main()
