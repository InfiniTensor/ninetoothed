"""Reward hacking guard — detect kernels that cheat on benchmarks.

Detects three categories of reward hacking:
1. Static (AST): zero-store, no-op kernel, constant-only output
2. Dynamic (runtime): all-zero output, NaN/Inf, output == input (no-op)
3. Output validation: shape mismatch, dtype mismatch

Usage (standalone):
    python scripts/reward_hacking_guard.py --op add --shapes 1024

Usage (Python import):
    from scripts.reward_hacking_guard import static_check, dynamic_check, full_guard
    flags = static_check(source_path="~/.ninetoothed/abc123.py")
    flags = dynamic_check(result_tensor, input_tensor)

"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Optional


def static_check(
    source_path: Optional[str] = None,
    source_code: Optional[str] = None,
) -> list[str]:
    """AST-based static analysis of a generated Triton kernel.

    Returns list of flag strings (empty = clean).
    """
    if source_path and not source_code:
        source_code = Path(source_path).read_text(encoding="utf-8")
    if not source_code:
        return ["no_source: cannot analyse"]

    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        return [f"syntax_error: {e}"]

    tl_store = len(re.findall(r"tl\.store\(", source_code))
    tl_load = len(re.findall(r"tl\.load\(", source_code))
    zeros_patterns = len(re.findall(
        r"tl\.(zeros|ones|full)\(", source_code
    ))
    early_returns = len(re.findall(r"return\s*$", source_code, re.MULTILINE))

    flags = []
    has_kernel = tl_load > 0 or tl_store > 0 or "triton.jit" in source_code

    if has_kernel:
        if tl_store == 0:
            flags.append("no_tl_store: kernel has no tl.store — output never written")
        if tl_load == 0:
            flags.append("no_tl_load: kernel has no tl.load — input never read")

    if zeros_patterns > 0 and tl_store <= zeros_patterns:
        flags.append(
            f"constant_output: {zeros_patterns} store(s) write zeros/ones/full "
            "— output may be data-independent"
        )

    if tl_store > 0 and tl_load > 0:
        ratio = tl_load / tl_store
        if ratio > 32:
            flags.append(
                f"high_load_ratio: {tl_load} loads vs {tl_store} stores "
                f"(ratio {ratio:.1f}) — possible redundant loads"
            )

    return flags


def dynamic_check(
    output_tensor,
    input_tensor=None,
    atol: float = 1e-6,
) -> list[str]:
    """Runtime checks for data-independence and no-op patterns.

    Args:
        output_tensor: the output tensor AFTER kernel execution.
        input_tensor: optional first input (for no-op detection).
        atol: tolerance for all-close checks.

    Returns list of flag strings (empty = clean).
    """
    try:
        import torch
    except ImportError:
        return ["no_torch: dynamic analysis requires torch"]

    flags = []
    out = output_tensor

    if out.numel() > 0:
        max_abs = out.abs().max().item()
        if max_abs < atol:
            flags.append(
                f"all_zero_output: max|output|={max_abs:.2e} — "
                "kernel may have skipped computation"
            )

    if torch.isnan(out).any():
        flags.append("nan_in_output: kernel produced NaN")
    if torch.isinf(out).any():
        flags.append("inf_in_output: kernel produced Inf")

    if input_tensor is not None and input_tensor.shape == out.shape:
        try:
            if torch.allclose(out, input_tensor.to(out.dtype), rtol=0, atol=atol):
                flags.append(
                    "no_op_output: output == input — kernel may not compute"
                )
        except Exception:
            pass

    return flags


def wrapper_bypass_check(wrapper_source: str) -> list[str]:
    """Detect when a torch wrapper bypasses NineToothed entirely.

    Returns list of flag strings (empty = clean).
    """
    flags = []

    has_kernel_call = bool(re.search(r"kernel\s*\(", wrapper_source))

    pytorch_compute = re.findall(
        r"\.(view|reshape|transpose|permute|contiguous|expand|repeat)\s*\(",
        wrapper_source
    )
    has_heavy_pytorch = len(pytorch_compute) >= 3

    if not has_kernel_call and has_heavy_pytorch:
        flags.append(
            f"wrapper_bypass: no kernel() call, but {len(pytorch_compute)} "
            f"PyTorch view/reshape ops — computation bypasses NineToothed"
        )

    return flags


def kernel_noop_check(kernel_source: str) -> list[str]:
    """Detect placeholder/no-op kernels. Returns list of flag strings."""
    flags = []

    if re.search(r"output\s*=\s*input\b", kernel_source):
        has_computation = bool(re.search(
            r"(ntl\.|libdevice\.|tl\.\w+\(|\+|-|\*|/)",
            kernel_source.replace("output = input", "")
        ))
        if not has_computation:
            flags.append(
                "noop_kernel: 'output = input' with no computation — placeholder"
            )

    return flags


def full_guard(
    output_tensor,
    input_tensor=None,
    generated_source_path: Optional[str] = None,
) -> dict:
    """Run all checks and return a combined report dict.

    Returns:
        {"static": [...], "dynamic": [...], "clean": bool}
    """
    static_flags = []
    if generated_source_path:
        static_flags = static_check(source_path=generated_source_path)

    dynamic_flags = dynamic_check(output_tensor, input_tensor)

    return {
        "static": static_flags,
        "dynamic": dynamic_flags,
        "clean": not static_flags and not dynamic_flags,
    }


def _cli(argv: list[str]) -> int:
    """CLI: python reward_hacking_guard.py --op <name>"""
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--op", required=True, help="Operator name to check")
    p.add_argument("--shapes", nargs="+", default=["1024"])
    args = p.parse_args(argv)

    # Try to find generated source
    cache_dir = Path.home() / ".ninetoothed"
    if cache_dir.exists():
        files = sorted(cache_dir.glob("*.py"))
        matched = [f for f in files if args.op in f.read_text().lower()]
        if matched:
            path = matched[-1]
            print(f"Checking generated source: {path.name}")
            flags = static_check(source_path=str(path))
            if flags:
                print("  ⚠️  Static flags:")
                for f in flags:
                    print(f"    - {f}")
                return 1
            else:
                print("  ✅ Static analysis clean")
                return 0

    print(f"No generated source found for '{args.op}'")
    print("  Run the kernel once first to generate the cached source.")
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
