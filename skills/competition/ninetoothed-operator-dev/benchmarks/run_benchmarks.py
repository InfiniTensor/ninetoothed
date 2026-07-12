#!/usr/bin/env python3
"""
九齿算子完整 benchmark 套件。

覆盖全部 9 个算子的 ninetoothed vs PyTorch 性能对比，
以及 strided vs contiguous 的非连续访存开销分析。

用法：
    # 快速模式（每个算子 3 个规模点，< 1 分钟）
    python benchmarks/run_benchmarks.py --quick

    # 运行全部（正常 sweep，约 5-10 分钟）
    python benchmarks/run_benchmarks.py

    # 仅逐元素类
    python benchmarks/run_benchmarks.py --category elementwise --quick

    # 保存图表
    python benchmarks/run_benchmarks.py --save-path ./bench_results

依赖：
    - triton >= 3.0
    - torch >= 2.0 (CUDA)
    - ninetoothed
"""
import os
import sys
import argparse

# 必须在 import matplotlib 之前设置，避免 headless 环境阻塞
import matplotlib
matplotlib.use("Agg")

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.bench import benchmark


DTYPE = torch.float16
DEVICE = "cuda"


# ---- 1D sweep 规模 ----
# 快速: 3 点，正常: 9 点
SWEEP_1D_QUICK = [2**i for i in range(10, 19, 3)]   # 1K, 8K, 64K
SWEEP_1D_FULL  = [2**i for i in range(10, 24)]       # 1K ~ 8M

# ---- 2D sweep 规模 (n × n) ----
SWEEP_2D_QUICK = [2**i for i in range(5, 12, 3)]     # 32, 256, 2048
SWEEP_2D_FULL  = [2**i for i in range(5, 13)]         # 32 ~ 4096

# ---- Softmax/RMS sweep 规模 (512 × n) ----
SWEEP_COL_QUICK = [2**i for i in range(5, 13, 3)]    # 32, 256, 2048
SWEEP_COL_FULL  = [2**i for i in range(5, 14)]        # 32 ~ 8192


# ============================================================
# 辅助：算子包装器（延迟导入）
# ============================================================

def _check_cuda():
    if not torch.cuda.is_available():
        print("错误: 需要 CUDA 环境。")
        sys.exit(1)


_wrappers_cache = None


def _setup_wrappers():
    global _wrappers_cache
    if _wrappers_cache is not None:
        return _wrappers_cache

    from operators import (
        create_add_kernel,
        create_2d_add_kernel,
        create_relu_kernel,
        create_sigmoid_kernel,
        create_gelu_kernel,
        create_softmax_kernel,
        create_sum_kernel,
        create_strided_add_kernel,
        create_2d_strided_add_kernel,
        create_rms_norm_kernel,
    )

    class W: pass
    w = W()

    add_nt = create_add_kernel()
    w.nt_add = lambda x, y, out: add_nt(x, y, out)

    relu_nt = create_relu_kernel()
    w.nt_relu = lambda x, out: relu_nt(x, out)

    sigmoid_nt = create_sigmoid_kernel()
    w.nt_sigmoid = lambda x, out: sigmoid_nt(x, out)

    gelu_nt = create_gelu_kernel()
    w.nt_gelu = lambda x, out: gelu_nt(x, out)

    add_2d_nt = create_2d_add_kernel()
    w.nt_add_2d = lambda x, y, out: add_2d_nt(x, y, out)

    softmax_nt = create_softmax_kernel()
    w.nt_softmax = lambda x, out: softmax_nt(x, out, BLOCK_SIZE=x.shape[-1])

    sum_nt = create_sum_kernel()
    w.nt_sum = lambda x, out: sum_nt(x, out, BLOCK_SIZE=x.numel())

    strided_add_nt = create_strided_add_kernel()
    w.nt_strided_add = lambda x, y, out: strided_add_nt(x, y, out, BLOCK_SIZE=x.shape[0])

    strided_add_2d_nt = create_2d_strided_add_kernel()
    w.nt_strided_add_2d = lambda x, y, out: strided_add_2d_nt(
        x, y, out,
        BLOCK_SIZE_ROW=min(x.shape[0], 128),
        BLOCK_SIZE_COL=min(x.shape[1] // 2, 256))

    rms_norm_nt = create_rms_norm_kernel()
    w.nt_rms_norm = lambda x, wt, out: rms_norm_nt(
        x, wt.expand_as(x), 1e-5, out, BLOCK_SIZE=x.shape[-1])

    _wrappers_cache = w
    return w


# ============================================================
# 逐元素类
# ============================================================

def bench_add_1d(save_path=None, quick=False):
    print("\n--- Benchmark: Add 1D (逐元素) ---")
    w = _setup_wrappers()
    sweep = SWEEP_1D_QUICK if quick else SWEEP_1D_FULL

    def make_args(size):
        x = torch.randn((size,), dtype=DTYPE, device=DEVICE)
        y = torch.randn((size,), dtype=DTYPE, device=DEVICE)
        return (x, y, torch.empty_like(x))

    benchmark(
        impls={"ninetoothed": w.nt_add, "torch": lambda x, y, out: out.copy_(x + y)},
        make_inputs=lambda size: (make_args(size), {}),
        x_names=["size"],
        x_vals=sweep,
        plot_name="add_1d",
        save_path=save_path,
        x_log=True,
    )


def bench_relu(save_path=None, quick=False):
    print("\n--- Benchmark: ReLU (逐元素) ---")
    w = _setup_wrappers()
    sweep = SWEEP_1D_QUICK if quick else SWEEP_1D_FULL

    def make_args(size):
        x = torch.randn((size,), dtype=DTYPE, device=DEVICE)
        return (x, torch.empty_like(x))

    benchmark(
        impls={"ninetoothed": w.nt_relu,
               "torch": lambda x, out: out.copy_(torch.nn.functional.relu(x))},
        make_inputs=lambda size: (make_args(size), {}),
        x_names=["size"],
        x_vals=sweep,
        plot_name="relu",
        save_path=save_path,
        x_log=True,
    )


def bench_sigmoid(save_path=None, quick=False):
    print("\n--- Benchmark: Sigmoid (逐元素) ---")
    w = _setup_wrappers()
    sweep = SWEEP_1D_QUICK if quick else SWEEP_1D_FULL

    def make_args(size):
        x = torch.randn((size,), dtype=DTYPE, device=DEVICE)
        return (x, torch.empty_like(x))

    benchmark(
        impls={"ninetoothed": w.nt_sigmoid,
               "torch": lambda x, out: out.copy_(torch.sigmoid(x))},
        make_inputs=lambda size: (make_args(size), {}),
        x_names=["size"],
        x_vals=sweep,
        plot_name="sigmoid",
        save_path=save_path,
        x_log=True,
        tolerances={"torch": {"atol": 1e-2, "rtol": 1e-2}},
    )


def bench_gelu(save_path=None, quick=False):
    print("\n--- Benchmark: GELU (逐元素) ---")
    w = _setup_wrappers()
    sweep = SWEEP_1D_QUICK if quick else SWEEP_1D_FULL

    def make_args(size):
        x = torch.randn((size,), dtype=DTYPE, device=DEVICE)
        return (x, torch.empty_like(x))

    benchmark(
        impls={"ninetoothed": w.nt_gelu,
               "torch": lambda x, out: out.copy_(torch.nn.functional.gelu(x))},
        make_inputs=lambda size: (make_args(size), {}),
        x_names=["size"],
        x_vals=sweep,
        plot_name="gelu",
        save_path=save_path,
        x_log=True,
        tolerances={"torch": {"atol": 1e-2, "rtol": 1e-2}},
    )


def bench_add_2d(save_path=None, quick=False):
    print("\n--- Benchmark: Add 2D (逐元素) ---")
    w = _setup_wrappers()
    sweep = SWEEP_2D_QUICK if quick else SWEEP_2D_FULL

    def make_args(n):
        x = torch.randn((n, n), dtype=DTYPE, device=DEVICE)
        y = torch.randn((n, n), dtype=DTYPE, device=DEVICE)
        return (x, y, torch.empty_like(x))

    benchmark(
        impls={"ninetoothed": w.nt_add_2d, "torch": lambda x, y, out: out.copy_(x + y)},
        make_inputs=lambda n: (make_args(n), {}),
        x_names=["n"],
        x_vals=sweep,
        plot_name="add_2d",
        save_path=save_path,
        x_log=True,
    )


# ============================================================
# 归约类
# ============================================================

def bench_softmax(save_path=None, quick=False):
    print("\n--- Benchmark: Softmax (归约) ---")
    w = _setup_wrappers()
    sweep = SWEEP_COL_QUICK if quick else SWEEP_COL_FULL

    def make_args(n):
        x = torch.randn((512, n), dtype=DTYPE, device=DEVICE)
        return (x, torch.empty_like(x))

    benchmark(
        impls={"ninetoothed": w.nt_softmax,
               "torch": lambda x, out: out.copy_(torch.softmax(x, dim=-1))},
        make_inputs=lambda n: (make_args(n), {}),
        x_names=["n"],
        x_vals=sweep,
        plot_name="softmax",
        save_path=save_path,
        x_log=True,
        tolerances={"torch": {"atol": 1e-2, "rtol": 1e-2}},
    )


def bench_sum(save_path=None, quick=False):
    print("\n--- Benchmark: Sum (归约) ---")
    w = _setup_wrappers()
    sweep = SWEEP_1D_QUICK if quick else SWEEP_1D_FULL

    def make_args(size):
        x = torch.randn((size,), dtype=DTYPE, device=DEVICE)
        return (x, torch.empty(1, dtype=DTYPE, device=DEVICE))

    benchmark(
        impls={"ninetoothed": w.nt_sum, "torch": lambda x, out: out.copy_(x.sum())},
        make_inputs=lambda size: (make_args(size), {}),
        x_names=["size"],
        x_vals=sweep,
        plot_name="sum",
        save_path=save_path,
        x_log=True,
        tolerances={"torch": {"atol": 5e-1, "rtol": 1e-1}},
    )


# ============================================================
# 非连续 / 步长
# ============================================================

def bench_strided_add_1d(save_path=None, quick=False):
    print("\n--- Benchmark: Strided Add 1D (非连续) ---")
    w = _setup_wrappers()
    sweep = SWEEP_1D_QUICK if quick else SWEEP_1D_FULL

    def make_args(size):
        x = torch.randn((size,), dtype=DTYPE, device=DEVICE)
        y = torch.randn((size,), dtype=DTYPE, device=DEVICE)
        return (x, y, torch.zeros_like(x))

    def torch_strided(x, y, out):
        out.zero_()
        out[::2] = x[::2] + y[::2]

    benchmark(
        impls={"ninetoothed": w.nt_strided_add, "torch": torch_strided},
        make_inputs=lambda size: (make_args(size), {}),
        x_names=["size"],
        x_vals=sweep,
        plot_name="strided_add_1d",
        save_path=save_path,
        x_log=True,
    )


def bench_strided_add_2d(save_path=None, quick=False):
    print("\n--- Benchmark: Strided Add 2D (非连续) ---")
    w = _setup_wrappers()
    sweep = SWEEP_2D_QUICK if quick else SWEEP_2D_FULL

    def make_args(n):
        x = torch.randn((512, n), dtype=DTYPE, device=DEVICE)
        y = torch.randn((512, n), dtype=DTYPE, device=DEVICE)
        return (x, y, torch.zeros_like(x))

    def torch_strided_2d(x, y, out):
        out.zero_()
        out[:, ::2] = x[:, ::2] + y[:, ::2]

    benchmark(
        impls={"ninetoothed": w.nt_strided_add_2d, "torch": torch_strided_2d},
        make_inputs=lambda n: (make_args(n), {}),
        x_names=["n"],
        x_vals=sweep,
        plot_name="strided_add_2d",
        save_path=save_path,
        x_log=True,
    )


# ============================================================
# 融合算子
# ============================================================

def bench_rms_norm(save_path=None, quick=False):
    print("\n--- Benchmark: RMS Norm (融合归约) ---")
    w = _setup_wrappers()
    sweep = SWEEP_COL_QUICK if quick else SWEEP_COL_FULL

    def make_args(n):
        x = torch.randn((1024, n), dtype=DTYPE, device=DEVICE)
        wt = torch.randn((n,), dtype=DTYPE, device=DEVICE)
        return (x, wt, torch.empty_like(x))

    def torch_rms(x, wt, out):
        out.copy_(torch.nn.functional.rms_norm(
            x.float(), (x.shape[-1],), wt.float(), 1e-5).half())

    benchmark(
        impls={"ninetoothed": w.nt_rms_norm, "torch": torch_rms},
        make_inputs=lambda n: (make_args(n), {}),
        x_names=["n"],
        x_vals=sweep,
        plot_name="rms_norm",
        save_path=save_path,
        x_log=True,
        tolerances={"torch": {"atol": 1e-2, "rtol": 1e-2}},
    )


# ============================================================
# 专项：Stride vs Contiguous 开销分析
# ============================================================

def bench_stride_vs_contiguous(save_path=None, quick=False):
    """
    对比 stride=2 vs contiguous 的延迟差异。
    直接量化非连续访存对 GPU 性能的影响。
    """
    print("\n--- Benchmark: Stride vs Contiguous 开销分析 ---")
    w = _setup_wrappers()
    sweep = SWEEP_1D_QUICK if quick else SWEEP_1D_FULL

    def make_args(size):
        x = torch.randn((size,), dtype=DTYPE, device=DEVICE)
        y = torch.randn((size,), dtype=DTYPE, device=DEVICE)
        return (x, y, torch.zeros_like(x), torch.empty_like(x))

    benchmark(
        impls={
            "strided (stride=2)": lambda x, y, out_s, out_c: w.nt_strided_add(x, y, out_s),
            "contiguous": lambda x, y, out_s, out_c: w.nt_add(x, y, out_c),
        },
        make_inputs=lambda size: (make_args(size), {}),
        x_names=["size"],
        x_vals=sweep,
        plot_name="stride_vs_contiguous",
        save_path=save_path,
        x_log=True,
        ylabel="ms (stride=2 vs contiguous)",
        skip_correctness=True,
    )


# ============================================================
# 主入口
# ============================================================

ALL_BENCHMARKS = {
    "elementwise": {
        "add_1d": bench_add_1d,
        "relu": bench_relu,
        "sigmoid": bench_sigmoid,
        "gelu": bench_gelu,
        "add_2d": bench_add_2d,
    },
    "reduction": {
        "softmax": bench_softmax,
        "sum": bench_sum,
    },
    "noncontiguous": {
        "strided_add_1d": bench_strided_add_1d,
        "strided_add_2d": bench_strided_add_2d,
    },
    "fused": {
        "rms_norm": bench_rms_norm,
    },
    "analysis": {
        "stride_vs_contiguous": bench_stride_vs_contiguous,
    },
}


def main():
    parser = argparse.ArgumentParser(description="九齿算子 Benchmark 套件")
    parser.add_argument(
        "--category",
        choices=["all"] + list(ALL_BENCHMARKS.keys()),
        default="all",
        help="运行哪个类别（默认 all）",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="快速模式：每个算子仅 3 个规模点，适合快速验证",
    )
    parser.add_argument(
        "--save-path", default=None,
        help="图表保存目录",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="列出所有可用的 benchmark",
    )
    args = parser.parse_args()

    _check_cuda()

    if args.list:
        print("\n可用的 benchmark 类别和子项:")
        for cat, items in ALL_BENCHMARKS.items():
            print(f"\n  [{cat}]")
            for name in items:
                print(f"    - {name}")
        return

    if args.category == "all":
        selected = {}
        for cat in ALL_BENCHMARKS:
            selected.update(ALL_BENCHMARKS[cat])
    else:
        selected = ALL_BENCHMARKS[args.category]

    save_path = args.save_path
    if save_path:
        save_path = os.path.join(save_path, args.category)
        os.makedirs(save_path, exist_ok=True)

    total = len(selected)
    print(f"\n{'=' * 60}")
    print(f"九齿算子 Benchmark 套件")
    print(f"  类别: {args.category}")
    print(f"  模式: {'快速 (3 规模点)' if args.quick else f'正常 ({len(SWEEP_1D_FULL)} 规模点)'}")
    print(f"  子项: {total} 个")
    print(f"  dtype: {DTYPE}, device: {DEVICE}")
    print(f"{'=' * 60}")

    passed = 0
    failed = 0
    for i, (name, bench_fn) in enumerate(selected.items(), 1):
        print(f"\n[{i}/{total}]", end=" ", flush=True)
        try:
            bench_fn(save_path=save_path, quick=args.quick)
            passed += 1
            print(f"  ✓ {name} 完成")
        except Exception as e:
            print(f"\n  ✗ [{name}] 失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"完成: {passed}/{total} 通过" +
          (f", {failed} 失败" if failed else ""))
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
