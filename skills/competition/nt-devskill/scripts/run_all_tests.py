#!/usr/bin/env python3
"""nt-devskill 全套测试脚本 — 适配 MetaX GPU 服务器

包含四个测试阶段：
  1. 环境检测 (doctor)
  2. 正确性验证 (correctness)
  3. 性能基准测试 (benchmark)
  4. 多形状/多数据类型鲁棒性测试 (robustness)

用法：
    python run_all_tests.py                 # 运行全部测试
    python run_all_tests.py --phase doctor  # 只跑环境检测
    python run_all_tests.py --phase correctness
    python run_all_tests.py --phase benchmark
    python run_all_tests.py --phase robustness
    python run_all_tests.py --phase all     # 等价于不传 --phase

输出：
    - 终端实时打印结果
    - 生成 test_results.json (机器可读汇总)
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup: 将 Skill 根目录加入 sys.path
# ---------------------------------------------------------------------------
SKILL_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SKILL_ROOT))

RESULTS = {
    "meta": {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "skill_root": str(SKILL_ROOT),
    },
    "doctor": {},
    "correctness": [],
    "benchmark": [],
    "robustness": [],
}


# ===================================================================
# Phase 1: 环境检测
# ===================================================================
def phase_doctor():
    print("=" * 60)
    print("  Phase 1: 环境检测 (Doctor)")
    print("=" * 60)

    checks = {}

    # Python
    checks["python"] = sys.version.split()[0]
    print(f"  [OK] Python {checks['python']}")

    # PyTorch
    try:
        import torch
        checks["torch"] = torch.__version__
        print(f"  [OK] PyTorch {checks['torch']}")
    except ImportError:
        checks["torch"] = None
        print("  [FAIL] PyTorch not installed")

    # CUDA / MetaX GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            cuda_ver = torch.version.cuda or "N/A (MetaX/ROCm)"
            props = torch.cuda.get_device_properties(0)
            checks["gpu"] = gpu_name
            checks["gpu_memory_gb"] = round(props.total_memory / 1024**3, 1)
            checks["gpu_sm_count"] = props.multi_processor_count
            checks["cuda_version"] = cuda_ver
            print(f"  [OK] GPU: {gpu_name}")
            print(f"       Memory: {checks['gpu_memory_gb']} GB, SMs: {checks['gpu_sm_count']}")
            print(f"       CUDA/Driver: {cuda_ver}")
        else:
            checks["gpu"] = None
            print("  [FAIL] CUDA not available")
    except Exception as e:
        checks["gpu"] = None
        print(f"  [FAIL] GPU check error: {e}")

    # Triton
    try:
        import triton
        checks["triton"] = triton.__version__
        print(f"  [OK] Triton {checks['triton']}")
    except ImportError:
        checks["triton"] = None
        print("  [WARN] Triton not installed (may be bundled with ninetoothed)")

    # NineToothed
    try:
        import ninetoothed
        ver = getattr(ninetoothed, "__version__", "unknown")
        checks["ninetoothed"] = ver
        print(f"  [OK] ninetoothed {ver}")
    except ImportError:
        checks["ninetoothed"] = None
        print("  [FAIL] ninetoothed not installed")

    # pytest
    try:
        import pytest
        checks["pytest"] = pytest.__version__
        print(f"  [OK] pytest {checks['pytest']}")
    except ImportError:
        checks["pytest"] = None
        print("  [WARN] pytest not installed")

    # 导入测试 examples
    example_ops = ["add", "softmax", "matmul", "fused_rms_norm", "silu", "bmm", "addmm", "scaled_dot_product_attention", "swiglu", "conv2d", "rotary_position_embedding", "max_pool2d"]
    importable = []
    for op in example_ops:
        try:
            __import__(f"examples.{op}")
            importable.append(op)
            print(f"  [OK] examples.{op} importable")
        except Exception as e:
            print(f"  [FAIL] examples.{op}: {e}")

    checks["importable_examples"] = importable
    checks["total_examples"] = len(example_ops)

    RESULTS["doctor"] = checks
    print(f"\n  总结: {len(importable)}/{len(example_ops)} 算子可导入\n")

    return checks.get("gpu") is not None and checks.get("ninetoothed") is not None


# ===================================================================
# Phase 2: 正确性验证
# ===================================================================
_CORRECTNESS_CASES = [
    {
        "name": "add",
        "import_path": "examples.add",
        "func_name": "add",
        "args_fn": lambda torch: (
            torch.randn(98432, dtype=torch.float16, device="cuda"),
            torch.randn(98432, dtype=torch.float16, device="cuda"),
        ),
        "ref_fn": lambda torch, args: torch.add(*args),
        "tolerances": {"atol": 0, "rtol": 0},
    },
    {
        "name": "softmax",
        "import_path": "examples.softmax",
        "func_name": "softmax",
        "args_fn": lambda torch: (
            torch.randn(1823, 781, dtype=torch.float16, device="cuda"),
        ),
        "ref_fn": lambda torch, args: torch.softmax(args[0], dim=-1),
        "tolerances": {"atol": 0.001, "rtol": 0},
    },
    {
        "name": "matmul",
        "import_path": "examples.matmul",
        "func_name": "mm",
        "args_fn": lambda torch: (
            torch.randn(512, 512, dtype=torch.float16, device="cuda"),
            torch.randn(512, 512, dtype=torch.float16, device="cuda"),
        ),
        "ref_fn": lambda torch, args: torch.mm(*args),
        "tolerances": {"atol": 0.01, "rtol": 0},
    },
    {
        "name": "fused_rms_norm",
        "import_path": "examples.fused_rms_norm",
        "func_name": "fused_rms_norm",
        "args_fn": lambda torch: (
            torch.randn(1151, 8192, dtype=torch.float16, device="cuda"),
            torch.randn(8192, dtype=torch.float16, device="cuda"),
            1e-5,
        ),
        "ref_fn": lambda torch, args: torch.nn.functional.rms_norm(
            args[0], args[0].shape[-1:], args[1], args[2]
        ),
        "tolerances": {"atol": 0.001, "rtol": 0.005},
    },
    {
        "name": "silu",
        "import_path": "examples.silu",
        "func_name": "silu",
        "args_fn": lambda torch: (
            torch.randn(8, 256, 512, dtype=torch.float16, device="cuda"),
        ),
        "ref_fn": lambda torch, args: torch.nn.functional.silu(args[0]),
        "tolerances": {"atol": 0.001, "rtol": 0.001},
    },
    {
        "name": "bmm",
        "import_path": "examples.bmm",
        "func_name": "bmm",
        "args_fn": lambda torch: (
            torch.randn(4, 512, 1024, dtype=torch.float16, device="cuda"),
            torch.randn(4, 1024, 2028, dtype=torch.float16, device="cuda"),
        ),
        "ref_fn": lambda torch, args: torch.bmm(*args),
        "tolerances": {"atol": 0.01, "rtol": 0},
    },
    {
        "name": "addmm",
        "import_path": "examples.addmm",
        "func_name": "addmm",
        "args_fn": lambda torch: (
            torch.randn(512, 512, dtype=torch.float16, device="cuda"),
            torch.randn(512, 512, dtype=torch.float16, device="cuda"),
            torch.randn(512, 512, dtype=torch.float16, device="cuda"),
        ),
        "ref_fn": lambda torch, args: torch.addmm(*args),
        "tolerances": {"atol": 0.01, "rtol": 0.01},
    },
    {
        "name": "scaled_dot_product_attention",
        "import_path": "examples.scaled_dot_product_attention",
        "func_name": "scaled_dot_product_attention",
        "args_fn": lambda torch: (
            torch.randn(2, 8, 1024, 64, dtype=torch.float16, device="cuda"),
            torch.randn(2, 8, 1024, 64, dtype=torch.float16, device="cuda"),
            torch.randn(2, 8, 1024, 64, dtype=torch.float16, device="cuda"),
        ),
        "ref_fn": lambda torch, args: torch.nn.functional.scaled_dot_product_attention(*args),
        "tolerances": {"atol": 0.01, "rtol": 0},
    },
    {
        "name": "swiglu",
        "import_path": "examples.swiglu",
        "func_name": "swiglu",
        "args_fn": lambda torch: (
            torch.randn(4096, dtype=torch.float16, device="cuda"),
            torch.randn(4096, dtype=torch.float16, device="cuda"),
        ),
        "ref_fn": lambda torch, args: args[0] * (args[1] * torch.sigmoid(args[1].float()).half()),
        "tolerances": {"atol": 0.001, "rtol": 0.001},
    },
    {
        "name": "conv2d",
        "import_path": "examples.conv2d",
        "func_name": "conv2d",
        "args_fn": lambda torch: (
            torch.randn(1, 3, 32, 32, dtype=torch.float16, device="cuda"),
            torch.randn(16, 3, 3, 3, dtype=torch.float16, device="cuda"),
        ),
        "ref_fn": lambda torch, args: torch.nn.functional.conv2d(*args),
        "tolerances": {"atol": 0.01, "rtol": 0.01},
    },
    {
        "name": "max_pool2d",
        "import_path": "examples.max_pool2d",
        "func_name": "max_pool2d",
        "args_fn": lambda torch: (
            torch.randn(1, 3, 32, 32, dtype=torch.float16, device="cuda"),
            (2, 2),
        ),
        "ref_fn": lambda torch, args: torch.nn.functional.max_pool2d(args[0], args[1], stride=args[1]),
        "tolerances": {"atol": 0, "rtol": 0},
    },
    {
        "name": "rotary_position_embedding",
        "import_path": "examples.rotary_position_embedding",
        "func_name": "rotary_position_embedding",
        "args_fn": lambda torch: _make_rope_args(torch),
        "ref_fn": lambda torch, args: _rope_ref(torch, *args),
        "tolerances": {"atol": 0.01, "rtol": 0.01},
    },
]


def _make_rope_args(torch):
    batch, seq_len, num_heads, head_dim = 2, 32, 4, 32
    input = torch.randn(batch, seq_len, num_heads, head_dim, dtype=torch.float16, device="cuda")
    positions = torch.arange(seq_len, dtype=torch.float32, device="cuda")
    freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim // 2, dtype=torch.float32, device="cuda") / (head_dim // 2)))
    angles = positions[:, None] * freqs[None, :]
    sin_table = torch.sin(angles).to(torch.float16)
    cos_table = torch.cos(angles).to(torch.float16)
    return (input, sin_table, cos_table)


def _rope_ref(torch, x, sin_table, cos_table):
    batch_size, seq_len, num_heads, head_dim = x.shape
    half = head_dim // 2
    x_pairs = x.reshape(batch_size, seq_len, num_heads, half, 2)
    x0 = x_pairs[..., 0]
    x1 = x_pairs[..., 1]
    sin = sin_table.to(x.dtype).unsqueeze(0).unsqueeze(2).expand(batch_size, -1, num_heads, -1)
    cos = cos_table.to(x.dtype).unsqueeze(0).unsqueeze(2).expand(batch_size, -1, num_heads, -1)
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos
    return torch.stack([out0, out1], dim=-1).reshape(batch_size, seq_len, num_heads, head_dim)


def phase_correctness():
    import torch
    import importlib

    print("=" * 60)
    print("  Phase 2: 正确性验证 (Correctness)")
    print("=" * 60)

    torch.manual_seed(0)
    results = []

    for case in _CORRECTNESS_CASES:
        name = case["name"]
        try:
            mod = importlib.import_module(case["import_path"])
            func = getattr(mod, case["func_name"])
            args = case["args_fn"](torch)
            ref_out = case["ref_fn"](torch, args)
            nt_out = func(*args)

            atol = case["tolerances"]["atol"]
            rtol = case["tolerances"]["rtol"]

            passed = torch.allclose(nt_out, ref_out, atol=atol, rtol=rtol)
            max_diff = (nt_out - ref_out).abs().max().item()
            mean_diff = (nt_out - ref_out).abs().mean().item()

            record = {
                "name": name,
                "passed": passed,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "atol": atol,
                "rtol": rtol,
                "output_shape": list(nt_out.shape),
            }
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name:40s} max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")

        except Exception as e:
            record = {"name": name, "passed": False, "error": str(e), "traceback": traceback.format_exc()}
            print(f"  [ERR ] {name:40s} {e}")

        results.append(record)

    passed = sum(1 for r in results if r["passed"])
    print(f"\n  总计: {len(results)}  通过: {passed}  失败: {len(results) - passed}\n")
    RESULTS["correctness"] = results
    return results


# ===================================================================
# Phase 3: 性能基准
# ===================================================================
def _bench_fn(fn, args, warmup=10, trials=50):
    import torch
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
    return start.elapsed_time(end) / trials  # ms


def phase_benchmark():
    import torch
    import importlib

    print("=" * 60)
    print("  Phase 3: 性能基准 (Benchmark)")
    print("=" * 60)

    torch.manual_seed(0)
    results = []

    for case in _CORRECTNESS_CASES:
        name = case["name"]
        try:
            mod = importlib.import_module(case["import_path"])
            func = getattr(mod, case["func_name"])
            args = case["args_fn"](torch)

            nt_ms = _bench_fn(func, args)
            ref_ms = _bench_fn(case["ref_fn"], (torch, args))
            speedup = ref_ms / nt_ms if nt_ms > 0 else float("inf")

            record = {
                "name": name,
                "nt_ms": round(nt_ms, 4),
                "ref_ms": round(ref_ms, 4),
                "speedup": round(speedup, 3),
            }
            print(f"  {name:40s} nt={nt_ms:.4f}ms  torch={ref_ms:.4f}ms  speedup={speedup:.2f}x")

        except Exception as e:
            record = {"name": name, "error": str(e)}
            print(f"  [ERR] {name:40s} {e}")

        results.append(record)

    RESULTS["benchmark"] = results
    print()
    return results


# ===================================================================
# Phase 4: 鲁棒性测试 (多形状 + 多数据类型)
# ===================================================================
_ROBUSTNESS_SPECS = {
    "add": {
        "import_path": "examples.add",
        "func_name": "add",
        "shapes": [(1024,), (65536,), (98432,), (1048576,)],
        "ref_fn": lambda torch, a, b: torch.add(a, b),
        "nargs": 2,
    },
    "silu": {
        "import_path": "examples.silu",
        "func_name": "silu",
        "shapes": [(1024,), (4096, 4096), (8, 256, 512)],
        "ref_fn": lambda torch, x: torch.nn.functional.silu(x),
        "nargs": 1,
    },
    "matmul": {
        "import_path": "examples.matmul",
        "func_name": "mm",
        "shapes": [(128, 128), (512, 512), (1024, 1024), (256, 2048)],
        "ref_fn": lambda torch, a, b: torch.mm(a, b),
        "nargs": 2,
    },
    "softmax": {
        "import_path": "examples.softmax",
        "func_name": "softmax",
        "shapes": [(32, 32), (1024, 1024), (1823, 781), (4096, 4096)],
        "ref_fn": lambda torch, x: torch.softmax(x, dim=-1),
        "nargs": 1,
    },
    "swiglu": {
        "import_path": "examples.swiglu",
        "func_name": "swiglu",
        "shapes": [(1024,), (4096,), (4096, 4096)],
        "ref_fn": lambda torch, a, b: a * (b * torch.sigmoid(b.float()).to(b.dtype)),
        "nargs": 2,
    },
}


def phase_robustness():
    import torch
    import importlib

    print("=" * 60)
    print("  Phase 4: 鲁棒性测试 (Robustness)")
    print("=" * 60)

    torch.manual_seed(42)
    dtypes = [torch.float16, torch.float32]
    results = []

    for op_name, spec in _ROBUSTNESS_SPECS.items():
        try:
            mod = importlib.import_module(spec["import_path"])
            func = getattr(mod, spec["func_name"])
        except Exception as e:
            print(f"  [SKIP] {op_name}: import failed ({e})")
            continue

        for shape in spec["shapes"]:
            for dt in dtypes:
                dt_name = str(dt).split(".")[-1]
                try:
                    if spec["nargs"] == 1:
                        x = torch.randn(shape, dtype=dt, device="cuda")
                        ref = spec["ref_fn"](torch, x)
                        out = func(x)
                    else:
                        a = torch.randn(shape, dtype=dt, device="cuda")
                        b = torch.randn(shape, dtype=dt, device="cuda")
                        ref = spec["ref_fn"](torch, a, b)
                        out = func(a, b)

                    # 使用宽松容差
                    atol = 0.02 if dt == torch.float16 else 0.01
                    passed = torch.allclose(out, ref, atol=atol, rtol=0.01)
                    max_diff = (out - ref).abs().max().item()

                    record = {
                        "op": op_name,
                        "shape": list(shape),
                        "dtype": dt_name,
                        "passed": passed,
                        "max_diff": max_diff,
                    }
                    status = "PASS" if passed else "FAIL"
                    print(f"  [{status}] {op_name:15s} shape={str(shape):25s} {dt_name:8s} max_diff={max_diff:.6f}")

                except Exception as e:
                    record = {
                        "op": op_name,
                        "shape": list(shape),
                        "dtype": dt_name,
                        "passed": False,
                        "error": str(e),
                    }
                    print(f"  [ERR ] {op_name:15s} shape={str(shape):25s} {dt_name:8s} {e}")

                results.append(record)

    passed = sum(1 for r in results if r["passed"])
    print(f"\n  总计: {len(results)}  通过: {passed}  失败: {len(results) - passed}\n")
    RESULTS["robustness"] = results
    return results


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="nt-devskill 全套测试")
    parser.add_argument(
        "--phase",
        choices=["doctor", "correctness", "benchmark", "robustness", "all"],
        default="all",
    )
    parser.add_argument("--output", type=str, default="test_results.json")
    args = parser.parse_args()

    t0 = time.time()

    # Phase 1 always runs first to check environment
    ok = phase_doctor()
    if not ok and args.phase != "doctor":
        print("[ABORT] 环境检测未通过，请先安装缺失依赖。")
        RESULTS["meta"]["aborted"] = True
        _save_results(args.output)
        sys.exit(1)

    if args.phase in ("correctness", "all"):
        phase_correctness()

    if args.phase in ("benchmark", "all"):
        phase_benchmark()

    if args.phase in ("robustness", "all"):
        phase_robustness()

    elapsed = time.time() - t0
    RESULTS["meta"]["total_seconds"] = round(elapsed, 2)

    # Summary
    print("=" * 60)
    print("  最终汇总")
    print("=" * 60)
    if RESULTS["correctness"]:
        cp = sum(1 for r in RESULTS["correctness"] if r["passed"])
        print(f"  正确性: {cp}/{len(RESULTS['correctness'])} 通过")
    if RESULTS["benchmark"]:
        geos = [r["speedup"] for r in RESULTS["benchmark"] if "speedup" in r]
        if geos:
            import math
            geo_mean = math.exp(sum(math.log(s) for s in geos) / len(geos))
            print(f"  性能几何均值: {geo_mean:.2f}x (vs PyTorch)")
    if RESULTS["robustness"]:
        rp = sum(1 for r in RESULTS["robustness"] if r["passed"])
        print(f"  鲁棒性: {rp}/{len(RESULTS['robustness'])} 通过")
    print(f"  总耗时: {elapsed:.1f}s")
    print()

    _save_results(args.output)


def _save_results(path):
    with open(path, "w") as f:
        json.dump(RESULTS, f, indent=2, ensure_ascii=False, default=str)
    print(f"  结果已保存到 {path}")


if __name__ == "__main__":
    main()
