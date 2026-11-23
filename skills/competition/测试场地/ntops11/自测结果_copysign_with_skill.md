# 自测任务记录：copysign（Skill 辅助版）

> 对照版（无 Skill）：`测试场地/ntops12/copysign_no_skills.txt`
> 工程目录：`测试场地/ntops11/`
> 对话记录：`测试场地/ntops11/copysign_with_skills.txt`

## 输入任务说明

```
我需要实现一个九齿算子：copysign(input, other)

要求：
- 返回一个与 input 绝对值相同、但符号取自 other 的 tensor
- torch 参考：torch.copysign(input, other)
- 支持 float32 和 float16
- 必须正确处理 IEEE 754 特殊值：copysign(-0.0, +0.0) == +0.0，copysign(+0.0, -0.0) == -0.0
- input 和 other shape 相同
```

## AI 智能体执行记录摘要

- **Step 0 — Skill 激活与工程勘探**
  - 激活 Skill `nt-devskill`，派生 Explore 子代理摸清 Skill 目录结构：
    `examples/`(14)、`references/`(8)、`scripts/`(16)、`specs/`(12)、`agents/`、`evaluation/`、`tests/`。
  - 扫描 `src/ntops/kernels/`（78 个算子）与 `tests/`（79 个测试），确认 `copysign` 尚未存在。
  - 通过 MCP `ping` → `run_python` 确认远端环境：MetaX C500、torch 2.6.0+metax3.2.1.3、ninetoothed 已装。

- **Step 1 — 分类（Taxonomy）**
  - 依据 Skill `references/TAXONOMY.md` 与 `CODE_TEMPLATES.md`，`copysign` 被归为
    **Application = ElementWise**（双输入逐元素算子），复用 `ntops.kernels.element_wise.arrangement`。
  - 依据 `references/API_REFERENCE.md` § libdevice.copysign 条目确认：
    - `libdevice.copysign(x, y)` 仅对 float32/float64 定义，float16 行为未定义；
    - 禁用 `ntl.where(y >= 0, abs(x), -abs(x))`，无法区分 ±0.0（与 OPTIMIZATION_GUIDE 一致）。

- **Step 2 — 代码生成**
  - 参考 `cosh.py`（libdevice 用法）、`signbit.py`（bitcast 模板）、`fmax.py`（wrapper+test 结构）、
    `pow.py`（dtype cast 模板）生成 3 个新文件。
  - **关键决策**：分 dtype 双路径
    - float16：`ntl.cast → uint16 bitcast` → `abs_bits = in & 0x7FFF`，`sign_bits = ot & 0x8000`，
      `result = abs | sign` → bitcast 回 float16。保证 ±0.0 语义；
    - float32/64：cast 到 fp32 → `libdevice.copysign` → cast 回原 dtype。
  - Wrapper 加 `input.shape == other.shape` 与 `input.dtype == other.dtype` 双 assert，
    采用 `_cached_make(premake, ndim, dtype, block_size)`，初始 `block_size=1024`。

- **Step 2.5 — 测试循环（首次上传即通过，无失败轮次）**
  - 一次上传 `src/ntops/{kernels,torch}/copysign.py`、`tests/test_copysign.py`、两个 `__init__.py` 增量改动 →
    `/data/ntops11`，`pip install -e .`。
  - 首轮 `pytest tests/test_copysign.py -v`：**12/12 PASSED in 14.19s**
    （8 个 shape×dtype 参数化 + 2 个 signed-zero + 2 个特殊值含 inf/nan/-0.0）。
  - **未触发 FIX_CARDS 修复流程**，一次性通过。

- **Step 2.8 — 性能优化（Tile Sweep + Block Size 切换）**
  - 首轮 `bench_copysign.py` 暴露：默认 `block_size=1024` 在 `(4096,4096)` 上仅 0.89x/0.50x（fp32/fp16）。
  - 撰写 `bench_tile_sweep_copysign.py`，扫描 `[256, 512, 1024, 2048, 4096, 8192]`。结论：
    - 大 shape `(4096,4096)` 最优 `block=4096`（fp32 1427 GB/s，fp16 1369 GB/s，≈torch）；
    - 小 shape `(1024,)` 受 kernel launch 开销主导（host ~3%），tile 调整无法弥补。
  - 修改 `src/ntops/torch/copysign.py` 把默认 `_BLOCK_SIZE = 4096`，重新跑测试 + benchmark。

## 产出补丁摘要

- **新增文件**：
  - `src/ntops/kernels/copysign.py`
  - `src/ntops/torch/copysign.py`
  - `tests/test_copysign.py`
  - `bench_copysign.py` / `bench_tile_sweep_copysign.py` / `bench_compare_blocks.py`（优化诊断）
- **修改文件**：
  - `src/ntops/kernels/__init__.py`：`import copysign` + `__all__` 登记
  - `src/ntops/torch/__init__.py`：同上
  - `.gitignore`：新增 `.qoder`
- **关键代码片段（kernel application）**：

```python
# src/ntops/kernels/copysign.py
def application(input, other, output):
    if input.dtype is ntl.float16:
        in_bits  = ntl.cast(input, ntl.uint16, bitcast=True)
        ot_bits  = ntl.cast(other, ntl.uint16, bitcast=True)
        abs_bits  = in_bits & 0x7FFF
        sign_bits = ot_bits & 0x8000
        result_bits = abs_bits | sign_bits
        output = ntl.cast(result_bits, ntl.float16, bitcast=True)  # noqa: F841
    else:
        in_f32 = ntl.cast(input, ntl.float32)
        ot_f32 = ntl.cast(other, ntl.float32)
        result_f32 = libdevice.copysign(in_f32, ot_f32)
        output = ntl.cast(result_f32, input.dtype)  # noqa: F841
```

## Correctness 测试

命令：`python -m pytest tests/test_copysign.py -v`（MetaX C500）

- **首轮**（block_size=1024）：**12/12 passed** in 14.19s
- **优化后**（block_size=4096）：**12/12 passed** in 8.57s
- float32：8/8 参数化 + signed-zero + 特殊值 PASS
- float16：8/8 参数化 + signed-zero + 特殊值 PASS
- 非连续输入：**未测**（`arrangement` 使用 `flatten()`，要求 contiguous；这是 ntops elementwise 的通用约定，wrapper 未做 `.contiguous()`）
- IEEE 754 ±0.0：`torch.signbit` 逐位比对 PASS
- 特殊值（±inf, ±nan, ±0.0）：符号位 + NaN mask 双重校验 PASS

## Benchmark（block_size=4096，MetaX C500，200 iters）

| 输入规模 | dtype | nt GPU (ms) | torch GPU (ms) | Speedup | 带宽 nt (GB/s) | 带宽 torch (GB/s) | 瓶颈类型 |
|---------|--------|------------|----------------|---------|----------------|-------------------|----------|
| (1024,) | fp32 | 0.0359 | 0.0065 | **0.18x** | 0.3 | 1.9 | launch-overhead |
| (1024,) | fp16 | 0.0368 | 0.0063 | **0.17x** | 0.2 | 1.0 | launch-overhead |
| (1024,1024) | fp32 | 0.0432 | 0.0129 | **0.30x** | 291 | 979 | memory-bound（未充分占满） |
| (1024,1024) | fp16 | 0.0429 | 0.0086 | **0.20x** | 147 | 734 | memory-bound（未充分占满） |
| (4096,4096) | fp32 | 0.1410 | 0.1396 | **0.99x** ✓ | 1428 | 1442 | memory-bound（饱和） |
| (4096,4096) | fp16 | 0.0737 | 0.0720 | **0.98x** ✓ | 1367 | 1399 | memory-bound（饱和） |

命令：`python bench_copysign.py`（默认 block=4096）；`python bench_tile_sweep_copysign.py`（扫描 block）

性能结论：
- **memory-bound**：大 shape 带宽利用 1367–1428 GB/s，与 torch 的 1399–1442 GB/s 持平，达到 ≈0.99x；
- **小 shape 未达标**：shape ≤ 1K 时 speedup 仅 0.17–0.30x，瓶颈是 kernel launch 固定开销，与 tile 大小无关；
- **优化前后对比**：block_size 由 1024 → 4096 使大 shape fp32 从 0.89x → 0.99x，fp16 从 0.50x → 0.98x，分别提升 ~11% 与 ~96%。

## 失败诊断（如有）

**无 correctness 失败**。首轮上传即 12/12 PASS，未触发 FIX_CARDS。

性能侧的"伪失败"：
- 失败现象：默认 `block_size=1024` 在 `(4096,4096) fp16` 仅 0.50x。
- 根因判断：tile 太小导致 wave 利用率不足（memory-bound 算子的常见现象），无 FC 卡片对应，
  属于 OPTIMIZATION_GUIDE 中 "tile sweep" 经典场景。
- 修复方案：wrapper 内 `_BLOCK_SIZE = 4096`，通过 `_cached_make(premake, ..., _BLOCK_SIZE)` 传入。
- 验证闭环：同表最后一行 0.98x ✓。

## 不支持用例

- **dtype**：仅 float32 / float16，不支持 bfloat16 / int / bool / float64（未注册路径）。
- **广播**：`input` 与 `other` 必须 shape、dtype 都完全相同（torch.copysign 本身支持广播，本实现不支持）。
- **非连续输入**：arrangement 使用 `flatten()`，wrapper 未做 `.contiguous()`，非连续输入结果不可信
  （与 ntops 所有 elementwise 算子约定一致）。
- **小 shape 性能**：shape ≤ 1K 时相对 torch speedup 仅 0.17–0.30x，属 kernel launch 固定开销，无法通过 tile 优化消除。
- **硬件**：仅在 MetaX C500 上验证；`libdevice.copysign` 路径在 NVIDIA 上行为预期一致但未实测。

## Skill 效能小结（对照 A/B 用）

| 维度 | Skill 辅助（ntops11） |
|---|---|
| 首次通过轮次 | 1（12/12 PASS） |
| FIX_CARDS 触发次数 | 0 |
| 分类准确性 | 正确归到 ElementWise + libdevice 模板 |
| 关键决策引导 | Skill 明确禁止 `ntl.where` 实现（±0.0 错误），强制走 libdevice/bitcast |
| 优化脚本 | 直接复用 `diag_tile_sweep.py` 模式，自写 sweep 脚本 |
| 总耗时 | ≈ 1 轮对话（生成 + 1 次测试 + 1 轮性能调优） |
