import time
import torch
import ntops


def bench(shape, dtype, n_warmup=20, n_iter=200):
    input = torch.randn(shape, dtype=dtype, device="cuda")
    other = torch.randn(shape, dtype=dtype, device="cuda")
    out = torch.empty_like(input)

    nbytes = input.numel() * input.element_size() * 3

    # --- ntops ---
    for _ in range(n_warmup):
        ntops.torch.copysign(input, other, out=out)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_iter):
        ntops.torch.copysign(input, other, out=out)
    end.record()
    torch.cuda.synchronize()
    nt_ms = start.elapsed_time(end) / n_iter
    nt_bw = nbytes / (nt_ms * 1e-3) / 1e9

    t0 = time.perf_counter()
    for _ in range(n_iter):
        ntops.torch.copysign(input, other, out=out)
    torch.cuda.synchronize()
    nt_e2e = (time.perf_counter() - t0) * 1000 / n_iter

    # --- torch ---
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

    t0 = time.perf_counter()
    for _ in range(n_iter):
        torch.copysign(input, other, out=out)
    torch.cuda.synchronize()
    torch_e2e = (time.perf_counter() - t0) * 1000 / n_iter

    speedup_gpu = torch_ms / nt_ms
    speedup_e2e = torch_e2e / nt_e2e
    host_pct = (nt_e2e - nt_ms) / nt_e2e * 100

    print(f"shape={shape} dtype={str(dtype).split('.')[-1]:8s} "
          f"nt={nt_ms:7.4f}ms(E2E {nt_e2e:7.4f} host {host_pct:4.1f}%) BW={nt_bw:6.1f}GB/s | "
          f"torch={torch_ms:7.4f}ms(E2E {torch_e2e:7.4f}) BW={torch_bw:6.1f}GB/s | "
          f"speedup GPU={speedup_gpu:.2f}x E2E={speedup_e2e:.2f}x")


if __name__ == "__main__":
    print("=== copysign benchmark: ntops vs torch ===")
    for shape in [(1024,), (1024, 1024), (4096, 4096)]:
        for dtype in [torch.float32, torch.float16]:
            bench(shape, dtype)
