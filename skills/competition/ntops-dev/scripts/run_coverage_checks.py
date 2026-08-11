#!/usr/bin/env python3

import argparse
import time


def _require_cuda(torch):
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for ntops coverage checks.")


def _check_addmm():
    import torch

    import ntops

    _require_cuda(torch)
    torch.manual_seed(20260607)
    m, n, k = 64, 48, 80
    cases = []

    input_contig = torch.randn((m, n), device="cuda", dtype=torch.float16)
    mat1_contig = torch.randn((m, k), device="cuda", dtype=torch.float16)
    mat2_contig = torch.randn((k, n), device="cuda", dtype=torch.float16)
    cases.append(("contiguous", input_contig, mat1_contig, mat2_contig))

    input_nc = torch.randn((n, m), device="cuda", dtype=torch.float16).t()
    mat1_nc = torch.randn((k, m), device="cuda", dtype=torch.float16).t()
    mat2_nc = torch.randn((n, k), device="cuda", dtype=torch.float16).t()
    cases.append(("all_non_contiguous_transpose", input_nc, mat1_nc, mat2_nc))

    print("[addmm_non_contiguous]")
    for name, input_tensor, mat1, mat2 in cases:
        output = ntops.torch.addmm(input_tensor, mat1, mat2, beta=0.75, alpha=1.25)
        reference = torch.addmm(input_tensor, mat1, mat2, beta=0.75, alpha=1.25)
        torch.cuda.synchronize()
        max_abs = (output - reference).abs().max().item()
        ok = torch.allclose(output, reference, rtol=1e-2, atol=1e-2)
        print(
            f"{name} ok={ok} "
            f"input_contig={input_tensor.is_contiguous()} "
            f"mat1_contig={mat1.is_contiguous()} "
            f"mat2_contig={mat2.is_contiguous()} "
            f"input_stride={tuple(input_tensor.stride())} "
            f"mat1_stride={tuple(mat1.stride())} "
            f"mat2_stride={tuple(mat2.stride())} "
            f"max_abs={max_abs}"
        )


def _check_softmax():
    import torch
    import torch.nn.functional as F

    import ntops

    _require_cuda(torch)
    torch.manual_seed(20260607)
    cases = [
        ("single_last_dim_fp16", (7, 1), torch.float16, -1),
        ("wide_fp16", (2, 1024), torch.float16, -1),
        ("middle_dim_fp32", (3, 5, 7), torch.float32, 1),
        ("large_values_fp32", (4, 33), torch.float32, -1),
    ]

    print("[softmax_boundary]")
    for name, shape, dtype, dim in cases:
        if name == "large_values_fp32":
            input_tensor = torch.randn(shape, device="cuda", dtype=dtype) * 40
        else:
            input_tensor = torch.randn(shape, device="cuda", dtype=dtype)

        output = ntops.torch.softmax(input_tensor, dim=dim)
        reference = F.softmax(input_tensor, dim=dim)
        torch.cuda.synchronize()
        max_abs = (output - reference).abs().max().item()
        ok = torch.allclose(output, reference, rtol=1e-2, atol=1e-2)
        print(
            f"{name} ok={ok} shape={shape} dtype={dtype} dim={dim} max_abs={max_abs}"
        )


def _check_matmul():
    import torch

    import ntops

    _require_cuda(torch)
    seed = 0
    shape_a = (1, 394, 724)
    shape_b = (1, 724, 388)

    print("[matmul_fixed_repro]")
    print(f"seed={seed} shape_a={shape_a} shape_b={shape_b}")
    torch.manual_seed(seed)
    input_tensor = torch.randn(shape_a, dtype=torch.float16, device="cuda")
    other = torch.randn(shape_b, dtype=torch.float16, device="cuda")

    start = time.perf_counter()
    output = ntops.torch.matmul(input_tensor, other)
    reference = torch.matmul(input_tensor, other)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    diff = (output - reference).abs()
    max_abs = diff.max().item()
    max_ref = reference.abs().max().item()
    max_rel = (diff / reference.abs().clamp_min(1e-6)).max().item()
    ok = torch.allclose(output, reference, rtol=1e-2, atol=1e-2)
    print(
        f"ok={ok} elapsed_s={elapsed:.4f} max_abs={max_abs} "
        f"max_rel={max_rel} max_ref={max_ref}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ntops self-test coverage checks.")
    parser.add_argument("target", choices=["addmm", "softmax", "matmul"])
    args = parser.parse_args()

    if args.target == "addmm":
        _check_addmm()
    elif args.target == "softmax":
        _check_softmax()
    else:
        _check_matmul()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
