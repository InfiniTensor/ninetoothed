#!/usr/bin/env python3

import argparse
import time


def _shape(value):
    return tuple(int(part) for part in value.split(",") if part)


def _measure(torch, fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def _print_result(operator, shape, dtype, gpu, ntops_latency, torch_latency):
    print(f"operator={operator}")
    print(f"shape={shape}")
    print(f"dtype={dtype}")
    print(f"gpu={gpu}")
    print(f"ntops_ms={ntops_latency * 1000:.4f}")
    print(f"torch_ms={torch_latency * 1000:.4f}")
    print(f"relative_to_torch={ntops_latency / torch_latency:.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark selected ntops operators.")
    parser.add_argument("operator", choices=["softmax", "addmm", "matmul", "maximum"])
    parser.add_argument("--shape", default=None, help="Operator-specific comma shape.")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument(
        "--scalar-other",
        action="store_true",
        help="Use a zero-dimensional second input for maximum.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()

    import torch
    import torch.nn.functional as F

    import ntops

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    dtype = getattr(torch, args.dtype)
    gpu = torch.cuda.get_device_name(0)

    if args.operator == "softmax":
        shape = _shape(args.shape or "1024,1024")
        x = torch.randn(shape, device="cuda", dtype=dtype)
        dim = -1
        ntops_latency = _measure(
            torch, lambda: ntops.torch.softmax(x, dim=dim), args.warmup, args.iters
        )
        torch_latency = _measure(
            torch, lambda: F.softmax(x, dim=dim), args.warmup, args.iters
        )
    elif args.operator == "addmm":
        m, n, k = _shape(args.shape or "512,512,512")
        input_tensor = torch.randn((m, n), device="cuda", dtype=dtype)
        mat1 = torch.randn((m, k), device="cuda", dtype=dtype)
        mat2 = torch.randn((k, n), device="cuda", dtype=dtype)
        ntops_latency = _measure(
            torch,
            lambda: ntops.torch.addmm(input_tensor, mat1, mat2),
            args.warmup,
            args.iters,
        )
        torch_latency = _measure(
            torch,
            lambda: torch.addmm(input_tensor, mat1, mat2),
            args.warmup,
            args.iters,
        )
        shape = (m, n, k)
    elif args.operator == "matmul":
        m, n, k = _shape(args.shape or "512,512,512")
        mat1 = torch.randn((m, k), device="cuda", dtype=dtype)
        mat2 = torch.randn((k, n), device="cuda", dtype=dtype)
        ntops_latency = _measure(
            torch, lambda: ntops.torch.matmul(mat1, mat2), args.warmup, args.iters
        )
        torch_latency = _measure(
            torch, lambda: torch.matmul(mat1, mat2), args.warmup, args.iters
        )
        shape = (m, n, k)
    else:
        shape = _shape(args.shape or "1048576")
        input_tensor = torch.randn(shape, device="cuda", dtype=dtype)
        other_shape = () if args.scalar_other else shape
        other = torch.randn(other_shape, device="cuda", dtype=dtype)
        ntops_latency = _measure(
            torch,
            lambda: ntops.torch.maximum(input_tensor, other),
            args.warmup,
            args.iters,
        )
        torch_latency = _measure(
            torch,
            lambda: torch.maximum(input_tensor, other),
            args.warmup,
            args.iters,
        )

    _print_result(args.operator, shape, args.dtype, gpu, ntops_latency, torch_latency)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
