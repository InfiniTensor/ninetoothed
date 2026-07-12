#!/usr/bin/env python3
"""nt-devskill 统一流水线编排 (Pipeline)

五阶段流水线: PLAN → SCAFFOLD → VALIDATE → BENCHMARK → REPORT

用法:
    python scripts/pipeline.py run --op add           # 单算子全流程
    python scripts/pipeline.py run --op add --from validate  # 从验证阶段开始
    python scripts/pipeline.py list                    # 列出所有算子规格
    python scripts/pipeline.py gate                    # 对所有算子执行全流程
    python scripts/pipeline.py diagnose --error "illegal memory"  # 故障诊断
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parent.parent
SPECS_DIR = SKILL_ROOT / "specs"
LOG_FILE = SKILL_ROOT / "docs" / "pipeline_runs.jsonl"

sys.path.insert(0, str(SKILL_ROOT))

STAGES = ["plan", "scaffold", "validate", "benchmark", "report"]

# Safe reference function lookup — avoids eval()/exec() for compliance.
# Maps spec YAML reference strings to callable functions.
_REF_LOOKUP = {
    "torch.add(a, b)": lambda a, b: __import__("torch").add(a, b),
    "torch.mm(a, b)": lambda a, b: __import__("torch").mm(a, b),
    "torch.bmm(a, b)": lambda a, b: __import__("torch").bmm(a, b),
    "torch.addmm(c, a, b)": lambda c, a, b: __import__("torch").addmm(c, a, b),
    "torch.softmax(x, dim=-1)": lambda x: __import__("torch").softmax(x, dim=-1),
    "torch.nn.functional.silu(x)": lambda x: __import__("torch").nn.functional.silu(x),
    "torch.nn.functional.conv2d(input, filter)": lambda i, f: __import__("torch").nn.functional.conv2d(i, f),
    "torch.nn.functional.scaled_dot_product_attention(q, k, v)": lambda q, k, v: __import__("torch").nn.functional.scaled_dot_product_attention(q, k, v),
}


def _build_ref_fn(reference_str):
    """Build a reference function from a spec YAML reference string.

    Uses a lookup table for known references. Falls back to raising
    NotImplementedError for unknown or manual references.
    """
    if reference_str in _REF_LOOKUP:
        return _REF_LOOKUP[reference_str]
    raise NotImplementedError(
        f"No safe reference builder for: {reference_str!r}. "
        f"Add it to _REF_LOOKUP in pipeline.py."
    )


def load_spec(op_name):
    import yaml
    spec_path = SPECS_DIR / f"{op_name}.yaml"
    if not spec_path.exists():
        return None
    with open(spec_path) as f:
        return yaml.safe_load(f)


def log_run(record):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ===================================================================
# Stage 1: PLAN — 读取规格卡，确认算子信息和策略
# ===================================================================
def stage_plan(op_name):
    print(f"\n{'='*50}")
    print(f"  [PLAN] {op_name}")
    print(f"{'='*50}")

    spec = load_spec(op_name)
    if spec is None:
        print(f"  [FAIL] 未找到 specs/{op_name}.yaml")
        return None

    print(f"  名称:     {spec['name']}")
    print(f"  族:       {spec['family']}")
    print(f"  模式:     {spec['pattern']}")
    print(f"  描述:     {spec['description']}")
    print(f"  模板:     {spec['template']}")
    print(f"  参考实现: {spec['reference']}")

    if "composition" in spec:
        print(f"  组合复用: {spec['composition']}")

    tile = spec.get("tile_strategy", {})
    print(f"  Tile 策略: {tile.get('type', 'N/A')}")

    return spec


# ===================================================================
# Stage 2: SCAFFOLD — 检查模板文件是否存在
# ===================================================================
def stage_scaffold(spec):
    op_name = spec["name"]
    print(f"\n{'='*50}")
    print(f"  [SCAFFOLD] {op_name}")
    print(f"{'='*50}")

    checks = {}

    kernel_path = SKILL_ROOT / spec["template"]
    checks["kernel"] = kernel_path.exists()
    status = "OK" if checks["kernel"] else "MISSING"
    print(f"  [{status}] kernel: {spec['template']}")

    torch_path = SKILL_ROOT / spec.get("torch_wrapper", "")
    checks["torch_wrapper"] = torch_path.exists()
    status = "OK" if checks["torch_wrapper"] else "MISSING"
    print(f"  [{status}] torch:  {spec.get('torch_wrapper', 'N/A')}")

    init_path = kernel_path.parent / "__init__.py"
    checks["init"] = init_path.exists()
    status = "OK" if checks["init"] else "MISSING"
    print(f"  [{status}] __init__.py")

    if all(checks.values()):
        print(f"  [PASS] 所有文件就绪")
        return True
    else:
        missing = [k for k, v in checks.items() if not v]
        print(f"  [FAIL] 缺少文件: {missing}")
        return False


# ===================================================================
# Stage 3: VALIDATE — 正确性验证
# ===================================================================
def stage_validate(spec):
    import torch
    import importlib

    op_name = spec["name"]
    print(f"\n{'='*50}")
    print(f"  [VALIDATE] {op_name}")
    print(f"{'='*50}")

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return None

    torch.manual_seed(0)

    try:
        mod_path = spec["template"].replace("/", ".").replace(".py", "")
        # Import torch wrapper
        wrapper_path = spec.get("torch_wrapper", "")
        if wrapper_path:
            mod = importlib.import_module(wrapper_path.replace("/", ".").replace(".py", ""))
        else:
            mod = importlib.import_module(mod_path)

        func = getattr(mod, spec["name"])

        # Build test args from spec
        test_shapes = spec["acceptance"]["correctness"]["shapes"]
        atol = spec["acceptance"]["correctness"]["atol"]
        rtol = spec["acceptance"]["correctness"]["rtol"]

        # Generic arg builder
        args = []
        for inp in spec.get("inputs", []):
            if inp.get("type") == "scalar":
                args.append(inp.get("default", 1e-5))
            elif isinstance(inp.get("ndim"), int):
                shape = test_shapes[0] if inp["ndim"] <= len(test_shapes[0]) else test_shapes[0][:inp["ndim"]]
                dt = getattr(torch, inp.get("dtype", "float16"))
                args.append(torch.randn(shape, dtype=dt, device="cuda"))
            else:
                shape = test_shapes[0]
                dt = getattr(torch, inp.get("dtype", "float16"))
                args.append(torch.randn(shape, dtype=dt, device="cuda"))

        # Run kernel
        nt_out = func(*args)

        # Run reference
        ref_fn = _build_ref_fn(spec['reference'])
        ref_out = ref_fn(*args)

        # Compare
        passed = torch.allclose(nt_out, ref_out, atol=atol, rtol=rtol)
        max_diff = (nt_out - ref_out).abs().max().item()
        mean_diff = (nt_out - ref_out).abs().mean().item()

        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")
        print(f"          atol={atol}  rtol={rtol}")

        return {
            "passed": passed,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "atol": atol,
            "rtol": rtol,
        }

    except Exception as e:
        print(f"  [ERR] {e}")
        traceback.print_exc()
        return {"passed": False, "error": str(e)}


# ===================================================================
# Stage 4: BENCHMARK — 性能基准
# ===================================================================
def stage_benchmark(spec):
    import torch
    import importlib

    op_name = spec["name"]
    print(f"\n{'='*50}")
    print(f"  [BENCHMARK] {op_name}")
    print(f"{'='*50}")

    if not torch.cuda.is_available():
        print("  [SKIP] CUDA not available")
        return None

    torch.manual_seed(0)

    try:
        wrapper_path = spec.get("torch_wrapper", "")
        mod = importlib.import_module(wrapper_path.replace("/", ".").replace(".py", ""))
        func = getattr(mod, spec["name"])

        bench_shapes = spec["acceptance"]["benchmark"]["shapes"]
        warmup = spec["acceptance"]["benchmark"].get("warmup", 10)
        trials = spec["acceptance"]["benchmark"].get("trials", 50)

        args = []
        for inp in spec.get("inputs", []):
            if inp.get("type") == "scalar":
                args.append(inp.get("default", 1e-5))
            elif isinstance(inp.get("ndim"), int):
                shape = bench_shapes[0] if inp["ndim"] <= len(bench_shapes[0]) else bench_shapes[0][:inp["ndim"]]
                dt = getattr(torch, inp.get("dtype", "float16"))
                args.append(torch.randn(shape, dtype=dt, device="cuda"))
            else:
                shape = bench_shapes[0]
                dt = getattr(torch, inp.get("dtype", "float16"))
                args.append(torch.randn(shape, dtype=dt, device="cuda"))

        def bench(fn, args, warmup=warmup, trials=trials):
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
            return start.elapsed_time(end) / trials

        nt_ms = bench(func, args)

        ref_fn = _build_ref_fn(spec['reference'])
        ref_ms = bench(ref_fn, args)

        speedup = ref_ms / nt_ms if nt_ms > 0 else float("inf")

        print(f"  nt:    {nt_ms:.4f} ms")
        print(f"  torch: {ref_ms:.4f} ms")
        print(f"  speedup: {speedup:.2f}x")

        return {"nt_ms": round(nt_ms, 4), "ref_ms": round(ref_ms, 4), "speedup": round(speedup, 3)}

    except Exception as e:
        print(f"  [ERR] {e}")
        traceback.print_exc()
        return {"error": str(e)}


# ===================================================================
# Stage 5: REPORT — 汇总报告
# ===================================================================
def stage_report(op_name, results):
    print(f"\n{'='*50}")
    print(f"  [REPORT] {op_name}")
    print(f"{'='*50}")

    validate = results.get("validate", {})
    benchmark = results.get("benchmark", {})

    v_status = "PASS" if validate and validate.get("passed") else "FAIL"
    print(f"  正确性: {v_status}")
    if validate and "max_diff" in validate:
        print(f"    max_diff: {validate['max_diff']:.6f}")

    if benchmark and "speedup" in benchmark:
        print(f"  性能: {benchmark['speedup']:.2f}x (vs PyTorch)")
        print(f"    nt: {benchmark['nt_ms']:.4f} ms  torch: {benchmark['ref_ms']:.4f} ms")

    overall = "PASS" if validate and validate.get("passed") else "FAIL"
    print(f"\n  总体: {overall}")
    return overall == "PASS"


# ===================================================================
# Commands
# ===================================================================
def cmd_run(args):
    op_name = args.op
    start_from = args.start_from or "plan"
    start_idx = STAGES.index(start_from)

    record = {
        "op": op_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stages": {},
    }
    results = {}
    t0 = time.time()

    # PLAN
    if start_idx <= 0:
        spec = stage_plan(op_name)
        if spec is None:
            record["status"] = "FAIL"
            record["error"] = "spec not found"
            log_run(record)
            return
        record["stages"]["plan"] = "PASS"
    else:
        spec = load_spec(op_name)

    # SCAFFOLD
    if start_idx <= 1:
        scaffold_ok = stage_scaffold(spec)
        record["stages"]["scaffold"] = "PASS" if scaffold_ok else "FAIL"
        if not scaffold_ok and not args.continue_on_fail:
            record["status"] = "FAIL"
            log_run(record)
            return

    # VALIDATE
    if start_idx <= 2:
        val_result = stage_validate(spec)
        results["validate"] = val_result
        if val_result:
            record["stages"]["validate"] = "PASS" if val_result.get("passed") else "FAIL"
            record["stages"]["max_diff"] = val_result.get("max_diff")

    # BENCHMARK
    if start_idx <= 3:
        bench_result = stage_benchmark(spec)
        results["benchmark"] = bench_result
        if bench_result:
            record["stages"]["benchmark"] = "PASS" if "error" not in bench_result else "FAIL"
            record["stages"]["speedup"] = bench_result.get("speedup")

    # REPORT
    if start_idx <= 4:
        passed = stage_report(op_name, results)
        record["status"] = "PASS" if passed else "FAIL"

    record["total_seconds"] = round(time.time() - t0, 2)
    log_run(record)
    print(f"\n  日志已写入 {LOG_FILE}")


def cmd_list(args):
    print(f"{'='*50}")
    print(f"  算子规格列表")
    print(f"{'='*50}\n")

    specs = sorted(SPECS_DIR.glob("*.yaml"))
    if not specs:
        print("  没有找到规格文件")
        return

    for spec_path in specs:
        spec = load_spec(spec_path.stem)
        if spec:
            print(f"  {spec['name']:40s} family={spec['family']:20s} pattern={spec['pattern']}")

    print(f"\n  共 {len(specs)} 个算子")


def cmd_gate(args):
    """对所有算子执行全流程"""
    specs = sorted(SPECS_DIR.glob("*.yaml"))
    if not specs:
        print("没有找到规格文件")
        return

    results = {}
    for spec_path in specs:
        op = spec_path.stem
        print(f"\n{'#'*60}")
        print(f"# 算子: {op}")
        print(f"{'#'*60}")

        gate_args = argparse.Namespace(op=op, start_from="plan", continue_on_fail=True)
        try:
            cmd_run(gate_args)
            results[op] = "DONE"
        except Exception as e:
            results[op] = f"ERR: {e}"
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  Gate Summary")
    print(f"{'='*60}")
    for op, status in results.items():
        print(f"  {op:40s} {status}")


def cmd_diagnose(args):
    """根据错误信息匹配 Fix Card"""
    error_msg = args.error.lower()

    cards = {
        "illegal memory": "FC-01: Partial tile 越界 → 使用 Tensor(other=...) 或 ntl.where mask",
        "cannot squeeze": "FC-02: dtype 维度追踪错误 → 手动追踪 tile 后的维度变化",
        "returns zeros": "FC-03: application 中未赋值 → 确保 output = result",
        "autotuning": "FC-04: 搜索空间过大 → 添加 upper_bound 或使用固定值",
        "size must match": "FC-05: 广播维度不匹配 → 使用 [:, None] 扩展维度",
        "launch fails": "FC-06: Grid 维度为 0 → 检查输入张量和 BLOCK_SIZE",
        "nan": "FC-07: 数值不稳定 → 使用 float32 累加器和 ntl.cast",
        "inf": "FC-07: 数值不稳定 → 使用 float32 累加器和 ntl.cast",
        "dtype": "FC-08: dot 类型不匹配 → 确保输入为 float16",
        "import": "FC-09: 模块找不到 → 检查 sys.path 和 __init__.py",
        "slow": "FC-10: tile 大小非 2 的幂 → 使用 next_power_of_2",
    }

    print(f"\n{'='*50}")
    print(f"  故障诊断")
    print(f"{'='*50}\n")

    matched = False
    for key, card in cards.items():
        if key in error_msg:
            print(f"  匹配: {card}")
            matched = True

    if not matched:
        print("  未匹配到已知故障卡片。")
        print("  请查阅 references/FIX_CARDS.md 获取完整诊断流程。")
        print(f"\n  错误信息: {args.error}")


def main():
    parser = argparse.ArgumentParser(description="nt-devskill Pipeline")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="对单个算子执行全流程")
    run_p.add_argument("--op", required=True, help="算子名称")
    run_p.add_argument("--from", dest="start_from", choices=STAGES, help="从指定阶段开始")
    run_p.add_argument("--continue-on-fail", action="store_true", help="失败时继续后续阶段")

    sub.add_parser("list", help="列出所有算子规格")

    sub.add_parser("gate", help="对所有算子执行全流程")

    diag_p = sub.add_parser("diagnose", help="故障诊断")
    diag_p.add_argument("--error", required=True, help="错误信息关键词")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "gate":
        cmd_gate(args)
    elif args.command == "diagnose":
        cmd_diagnose(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
