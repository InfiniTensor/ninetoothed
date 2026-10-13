"""Correctness validation script for nt-devskill examples.

Compares ninetoothed kernel output against a reference (PyTorch) implementation.
Usage:
    python scripts/validate.py                  # run all examples
    python scripts/validate.py --op softmax      # run a single example
"""

import argparse
import sys
from pathlib import Path

import torch

SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT))


def _rand(shape, dtype=torch.float16, device="cuda"):
    return torch.randn(shape, dtype=dtype, device=device)


# Each test case: (module_import_path, func_name, args_fn, ref_fn, kwargs, tolerances)
_TEST_CASES = [
    {
        "name": "add",
        "import_path": "examples.add",
        "func_name": "add",
        "args_fn": lambda: (_rand(98432), _rand(98432)),
        "ref_fn": lambda a, b: torch.add(a, b),
        "tolerances": {"atol": 0, "rtol": 0},
    },
    {
        "name": "softmax",
        "import_path": "examples.softmax",
        "func_name": "softmax",
        "args_fn": lambda: (_rand((1823, 781)),),
        "ref_fn": lambda x: torch.softmax(x, dim=-1),
        "tolerances": {"atol": 0.001, "rtol": 0},
    },
    {
        "name": "matmul",
        "import_path": "examples.matmul",
        "func_name": "mm",
        "args_fn": lambda: (_rand((512, 512)), _rand((512, 512))),
        "ref_fn": lambda a, b: torch.mm(a, b),
        "tolerances": {"atol": 0.01, "rtol": 0},
    },
    {
        "name": "fused_rms_norm",
        "import_path": "examples.fused_rms_norm",
        "func_name": "fused_rms_norm",
        "args_fn": lambda: (
            _rand((1151, 8192)),
            _rand(8192),
            1e-5,
        ),
        "ref_fn": lambda x, w, eps: torch.nn.functional.rms_norm(
            x, x.shape[-1:], w, eps
        ),
        "tolerances": {"atol": 0.001, "rtol": 0.005},
    },
    {
        "name": "silu",
        "import_path": "examples.silu",
        "func_name": "silu",
        "args_fn": lambda: (_rand((8, 256, 512)),),
        "ref_fn": lambda x: torch.nn.functional.silu(x),
        "tolerances": {"atol": 0.001, "rtol": 0.001},
    },
    {
        "name": "bmm",
        "import_path": "examples.bmm",
        "func_name": "bmm",
        "args_fn": lambda: (_rand((4, 512, 1024)), _rand((4, 1024, 2028))),
        "ref_fn": lambda a, b: torch.bmm(a, b),
        "tolerances": {"atol": 0.01, "rtol": 0},
    },
    {
        "name": "addmm",
        "import_path": "examples.addmm",
        "func_name": "addmm",
        "args_fn": lambda: (_rand((512, 512)), _rand((512, 512)), _rand((512, 512))),
        "ref_fn": lambda c, a, b: torch.addmm(c, a, b),
        "tolerances": {"atol": 0.01, "rtol": 0.01},
    },
    {
        "name": "scaled_dot_product_attention",
        "import_path": "examples.scaled_dot_product_attention",
        "func_name": "scaled_dot_product_attention",
        "args_fn": lambda: (
            _rand((2, 8, 1024, 64)),
            _rand((2, 8, 1024, 64)),
            _rand((2, 8, 1024, 64)),
        ),
        "ref_fn": lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(
            q, k, v
        ),
        "tolerances": {"atol": 0.01, "rtol": 0},
    },
    {
        "name": "swiglu",
        "import_path": "examples.swiglu",
        "func_name": "swiglu",
        "args_fn": lambda: (_rand(4096), _rand(4096)),
        "ref_fn": lambda a, b: a * (b * torch.sigmoid(b.float()).half()),
        "tolerances": {"atol": 0.001, "rtol": 0.001},
    },
    {
        "name": "conv2d",
        "import_path": "examples.conv2d",
        "func_name": "conv2d",
        "args_fn": lambda: (
            _rand((1, 3, 32, 32)),
            _rand((16, 3, 3, 3)),
        ),
        "ref_fn": lambda i, f: torch.nn.functional.conv2d(i, f),
        "tolerances": {"atol": 0.01, "rtol": 0.01},
    },
    {
        "name": "max_pool2d",
        "import_path": "examples.max_pool2d",
        "func_name": "max_pool2d",
        "args_fn": lambda: (_rand((1, 3, 32, 32)), (2, 2)),
        "ref_fn": lambda x, w: torch.nn.functional.max_pool2d(x, w, stride=w),
        "tolerances": {"atol": 0, "rtol": 0},
    },
    {
        "name": "rotary_position_embedding",
        "import_path": "examples.rotary_position_embedding",
        "func_name": "rotary_position_embedding",
        "args_fn": lambda: _make_rope_args(),
        "ref_fn": lambda x, sin, cos: _rope_ref(x, sin, cos),
        "tolerances": {"atol": 0.01, "rtol": 0.01},
    },
]


def _make_rope_args():
    """Generate test inputs for rotary position embedding."""
    batch, seq_len, num_heads, head_dim = 2, 32, 4, 32
    device = "cuda"
    input = _rand((batch, seq_len, num_heads, head_dim))
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim // 2, dtype=torch.float32, device=device) / (head_dim // 2)))
    angles = positions[:, None] * freqs[None, :]
    sin_table = torch.sin(angles)
    cos_table = torch.cos(angles)
    return (input, sin_table, cos_table)


def _rope_ref(x, sin_table, cos_table):
    """Reference implementation of interleaved rotary position embedding.

    Matches the kernel's interleaved pattern: pairs adjacent elements
    (x[0],x[1]), (x[2],x[3]), ... and rotates each pair using
    sin_table[k], cos_table[k] for the k-th pair.
    """
    batch_size, seq_len, num_heads, head_dim = x.shape
    half = head_dim // 2
    # Reshape to pairs: (..., half, 2)
    x_pairs = x.reshape(batch_size, seq_len, num_heads, half, 2)
    x0 = x_pairs[..., 0]  # even elements
    x1 = x_pairs[..., 1]  # odd elements
    # sin/cos shape: (seq_len, half) → broadcast to (batch, seq, heads, half)
    sin = sin_table.to(x.dtype).unsqueeze(0).unsqueeze(2).expand(batch_size, -1, num_heads, -1)
    cos = cos_table.to(x.dtype).unsqueeze(0).unsqueeze(2).expand(batch_size, -1, num_heads, -1)
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    # Interleave back: (..., half, 2) → (..., head_dim)
    out = torch.stack([out0, out1], dim=-1).reshape(batch_size, seq_len, num_heads, head_dim)
    return out


def validate_one(case):
    import importlib

    mod = importlib.import_module(case["import_path"])
    func = getattr(mod, case["func_name"])
    args = case["args_fn"]()
    ref_out = case["ref_fn"](*args)
    nt_out = func(*args)

    atol = case["tolerances"].get("atol", 1e-3)
    rtol = case["tolerances"].get("rtol", 1e-3)

    if torch.allclose(nt_out, ref_out, atol=atol, rtol=rtol):
        print(f"  [PASS] {case['name']}")
        return True
    else:
        max_diff = (nt_out - ref_out).abs().max().item()
        print(f"  [FAIL] {case['name']}  (max_diff={max_diff:.6f}, atol={atol}, rtol={rtol})")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", type=str, default=None, help="Run only this operator")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("[SKIP] CUDA is not available")
        return

    cases = _TEST_CASES if args.op is None else [c for c in _TEST_CASES if c["name"] == args.op]

    if not cases:
        print(f"[ERROR] Unknown operator: {args.op}")
        sys.exit(1)

    print(f"Validating {len(cases)} operator(s) ...")
    results = [validate_one(c) for c in cases]
    passed = sum(results)
    failed = len(results) - passed

    print(f"\n{'=' * 30}")
    print(f"  Total: {len(results)}  Passed: {passed}  Failed: {failed}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
