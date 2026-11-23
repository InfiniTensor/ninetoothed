import time
import torch
import ntops
from ntops.kernels.copysign import premake
from ntops.torch.utils import _cached_make


def bench(shape, dtype, block_size, n_warmup=20, n_iter=200):
    input = torch.randn(shape, dtype=dtype, device="cuda")
    other = torch.randn(shape, dtype=dtype, device="cuda")
    out = torch.empty_like(input)
    nbytes = input.numel() * input.element_size() * 3

    kernel = _cached_make(premake, input.ndim, dtype, block_size)

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
    nt_ms = start.elapsed_time(end) / n_iter
    nt_bw = nbytes / (nt_ms * 1e-3) / 1e9

    for _ in range(n_warmup):
        torch.copysign(input, other, out=out)
    torch.cuda.synchronize()
    start.record()
    for _ in range(n_iter):
        torch.copysign(input, other, out=out)
    end.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(end) / n_iter
    torch_bw = nbytes / (torch_ms * 1e-3) / 1e9

    speedup = torch_ms / nt_ms
    print(f"shape={str(shape):20s} dtype={str(dtype).split('.')[-1]:8s} block={block_size:>5d} "
          f"nt={nt_ms:7.4f}ms BW={nt_bw:7.1f} | torch={torch_ms:7.4f}ms BW={torch_bw:7.1f} | "
          f"speedup={speedup:.2f}x")


if __name__ == "__main__":
    print("=== copysign: block_size=1024 vs 4096 ===")
    for shape in [(1024,), (1024, 1024), (4096, 4096)]:
        for dtype in [torch.float32, torch.float16]:
            bench(shape, dtype, 1024)
            bench(shape, dtype, 4096)
