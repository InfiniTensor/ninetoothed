#!/usr/bin/env python3

import argparse
import time


def _measure(torch, fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iters


def _report(torch, name, shape, dtype, ntops_fn, torch_fn, warmup, iters):
    ntops_ms = _measure(torch, ntops_fn, warmup, iters)
    torch_ms = _measure(torch, torch_fn, warmup, iters)
    print(
        f"{name}\tshape={shape}\tdtype={dtype}\t"
        f"ntops_ms={ntops_ms:.4f}\ttorch_ms={torch_ms:.4f}\t"
        f"ratio={ntops_ms / torch_ms:.4f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark selected official ntops operators against PyTorch."
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F

    import ntops

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for ntops benchmark.")

    torch.manual_seed(20260607)
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"warmup={args.warmup} iters={args.iters}")

    for dtype in [torch.float16, torch.float32]:
        x = torch.randn((1048576,), device="cuda", dtype=dtype)
        y = torch.randn((1048576,), device="cuda", dtype=dtype)
        _report(
            torch,
            "gelu",
            (1048576,),
            dtype,
            lambda x=x: ntops.torch.gelu(x),
            lambda x=x: F.gelu(x),
            args.warmup,
            args.iters,
        )
        _report(
            torch,
            "relu",
            (1048576,),
            dtype,
            lambda x=x: ntops.torch.relu(x),
            lambda x=x: torch.relu(x),
            args.warmup,
            args.iters,
        )
        _report(
            torch,
            "silu",
            (1048576,),
            dtype,
            lambda x=x: ntops.torch.silu(x),
            lambda x=x: F.silu(x),
            args.warmup,
            args.iters,
        )
        _report(
            torch,
            "add",
            (1048576,),
            dtype,
            lambda x=x, y=y: ntops.torch.add(x, y),
            lambda x=x, y=y: torch.add(x, y),
            args.warmup,
            args.iters,
        )
        _report(
            torch,
            "mul",
            (1048576,),
            dtype,
            lambda x=x, y=y: ntops.torch.mul(x, y),
            lambda x=x, y=y: torch.mul(x, y),
            args.warmup,
            args.iters,
        )

    x = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    _report(
        torch,
        "softmax",
        (1024, 1024),
        torch.float16,
        lambda x=x: ntops.torch.softmax(x, dim=-1),
        lambda x=x: F.softmax(x, dim=-1),
        args.warmup,
        args.iters,
    )

    m = n = k = 512
    input_tensor = torch.randn((m, n), device="cuda", dtype=torch.float16)
    mat1 = torch.randn((m, k), device="cuda", dtype=torch.float16)
    mat2 = torch.randn((k, n), device="cuda", dtype=torch.float16)
    _report(
        torch,
        "addmm",
        (m, n, k),
        torch.float16,
        lambda: ntops.torch.addmm(input_tensor, mat1, mat2),
        lambda: torch.addmm(input_tensor, mat1, mat2),
        args.warmup,
        args.iters,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
