import torch
import time
import ntops
from ntops.kernels.copysign import premake
from ntops.torch.utils import _cached_make

TILE_CANDIDATES = [256, 512, 1024, 2048, 4096, 8192]


def build_kernel(block_size, ndim, dtype):
    return _cached_make(premake, ndim, dtype, block_size)


def bench_kernel(kernel, input, other, out, n_warmup=20, n_iter=200):
    for _ in range(n_warmup):
        kernel(input, other, out)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        kernel(input, other, out)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n_iter


def sweep(shape, dtype):
    input = torch.randn(shape, dtype=dtype, device="cuda")
    other = torch.randn(shape, dtype=dtype, device="cuda")
    out = torch.empty_like(input)
    nbytes = input.numel() * input.element_size() * 3

    print(f"\n=== sweep shape={shape} dtype={str(dtype).split('.')[-1]} ===")
    print(f"{'block':>8s} | {'ms':>10s} | {'BW(GB/s)':>10s} | status")
    best_ms, best_tile = float("inf"), None

    for bs in TILE_CANDIDATES:
        try:
            kernel = build_kernel(bs, input.ndim, dtype)
            ms = bench_kernel(kernel, input, other, out)
            bw = nbytes / (ms * 1e-3) / 1e9
            if ms < best_ms:
                best_ms, best_tile = ms, bs
            status = "BEST" if ms == best_ms else f"{ms/best_ms:.2f}x"
            print(f"{bs:>8d} | {ms:>10.4f} | {bw:>10.1f} | {status}")
        except Exception as e:
            print(f"{bs:>8d} | {'FAIL':>10s} | {'---':>10s} | {type(e).__name__}: {str(e)[:60]}")

    print(f"\nOptimal: block={best_tile}, GPU={best_ms:.4f}ms")

    # torch reference
    for _ in range(20):
        torch.copysign(input, other, out=out)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(200):
        torch.copysign(input, other, out=out)
    end.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(end) / 200
    torch_bw = nbytes / (torch_ms * 1e-3) / 1e9
    print(f"torch:        {torch_ms:.4f}ms, BW={torch_bw:.1f}GB/s, speedup(best/torch)={torch_ms/best_ms:.2f}x")


if __name__ == "__main__":
    for shape in [(1024,), (1024, 1024), (4096, 4096)]:
        for dtype in [torch.float32, torch.float16]:
            sweep(shape, dtype)
