"""Performance / diagnosis proxy tasks (6): train 4, holdout 2.

These are diagnosis tasks: the agent is given a failing or slow kernel and must
produce a diagnosis hitting the expected findings, plus a minimal fix. Scored by
how many expected_findings the diagnosis covers, plus whether the fix passes
correctness (when a fix is required).
"""

from __future__ import annotations

from schema import TaskSpec

TASKS = [
    TaskSpec(
        id="pd01",
        category="perf_diag",
        split="train",
        difficulty="medium",
        name="softmax_no_maxsub",
        kind="diagnosis",
        prompt="给定一个 softmax kernel，在 fp16 大值输入下输出 NaN。定位根因并给出最小修复。",
        scenario="softmax 的 application 直接对 x 取 exp，未先减去行最大值；fp16 大值时 exp 溢出为 inf，归一化得到 NaN。",
        expected_findings=(
            "缺少减去行最大值的数值稳定步骤",
            "exp 在 fp16 大值下溢出",
            "修复：先计算 row max 再 exp(x - row_max)",
        ),
        buggy_snippet="numerator = ntl.exp(x); out = numerator / ntl.sum(numerator)",
    ),
    TaskSpec(
        id="pd02",
        category="perf_diag",
        split="train",
        difficulty="medium",
        name="mean_no_upcast",
        kind="diagnosis",
        prompt="给定一个 mean reduction kernel，fp16 大 N 输入下精度不达标。定位根因并修复。",
        scenario="reduction 在 fp16 中直接累加，N 较大时累积误差超过容差。",
        expected_findings=(
            "fp16 中累积导致精度损失",
            "需在 fp32 中累积",
            "修复：ntl.cast 到 float32 后求和，再 cast 回",
        ),
        buggy_snippet="total = ntl.sum(x)  # x 为 fp16，未升精度",
    ),
    TaskSpec(
        id="pd03",
        category="perf_diag",
        split="train",
        difficulty="easy",
        name="add_bench_memorybound",
        kind="diagnosis",
        prompt="为一个 add kernel 设计 benchmark 并判定瓶颈类型。",
        scenario="add 是纯逐元素算子，每元素 1 FLOP、读两个写一个，算术强度极低。需用 benchmark 数据判定 memory-bound。",
        expected_findings=(
            "给出 benchmark 模板（warmup + 多次测量 + CUDA Event）",
            "计算有效带宽 GB/s 或与峰值带宽对比",
            "算术强度远低于脊点，判定为 memory-bound",
        ),
    ),
    TaskSpec(
        id="pd04",
        category="perf_diag",
        split="train",
        difficulty="medium",
        name="inspect_tile_config",
        kind="diagnosis",
        prompt="检查某 kernel 的生成源码，报告其 tile 配置与并行参数。",
        scenario="需要从 NineToothed 生成的 Triton 源码中提取 tile 形状、num_warps、num_stages 作为优化判断依据。",
        expected_findings=(
            "从 ~/.ninetoothed 缓存读取生成源码",
            "报告 tile 形状与 block 大小",
            "报告 num_warps / num_stages",
        ),
    ),
    TaskSpec(
        id="pd05",
        category="perf_diag",
        split="holdout",
        difficulty="hard",
        name="noncontig_regression",
        kind="diagnosis",
        prompt="某 kernel 在连续输入下正确，非连续输入下结果错误。定位根因并给出修复路径。",
        scenario="kernel 假设输入按连续存储计算偏移；传入转置等非连续张量时偏移错误，结果不对。",
        expected_findings=(
            "根因是 kernel 假设了连续存储",
            "wrapper 层 .contiguous() 快速路径，或用 tile(strides=) 表达布局",
            "在测试矩阵中加入非连续输入用例验证",
        ),
    ),
    TaskSpec(
        id="pd06",
        category="perf_diag",
        split="holdout",
        difficulty="hard",
        name="aot_numwarps_mismatch",
        kind="diagnosis",
        prompt="JIT 路径正常，但 AOT 构建失败。定位根因并给出修复。",
        scenario="AOT 构建使用了与目标不匹配的 num_warps / num_stages 默认值，导致构建失败。",
        expected_findings=(
            "AOT 与 JIT 的默认 num_warps / num_stages 不一致",
            "向 make(caller='cuda', ...) 显式传入 num_warps / num_stages",
            "用 aot_build_smoke 验证产物 .py 与 .h 完整",
        ),
    ),
]
