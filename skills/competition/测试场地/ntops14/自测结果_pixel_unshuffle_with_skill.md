# 自测任务记录：pixel_unshuffle（Skill 辅助版）

> 工程目录：`测试场地/ntops14/`
> 对话记录：`测试场地/ntops14/pixel_unshuffle.txt`
> 前序对照：`测试场地/ntops13/自测结果_scatter_add_with_skill.md`（scatter_add 曾出现"绕过九齿走纯 Triton"事故，本任务用来验证 Step 0.8 方案先行确认机制的有效性）

## 输入任务说明

```
我需要一个九齿算子：pixel_unshuffle(input, downscale_factor)

功能是 torch.nn.functional.pixel_unshuffle 的等价实现：
  把 (..., C, H*r, W*r) 重排成 (..., C*r*r, H, W)，其中 r = downscale_factor。

要求：
- 支持 3D (C, H, W) 和 4D (N, C, H, W) 输入
- downscale_factor 是正整数
- H 和 W 必须能被 downscale_factor 整除
- 支持 float32 和 float16
- 请在 MetaX C500 GPU 上通过 MCP 实现和测试
- 和 torch.nn.functional.pixel_unshuffle 做正确性对比
- 跑个 benchmark 对比下 torch 的速度
```

## AI 智能体执行记录摘要

- **Step 0 — 环境分析（MANDATORY 全部命中）**
  - 本地勘探：`ls` / `find src/`（159 个 `.py`）/ 读 `__init__.py` 与多个示例算子（`add` / `rot90` / `msort` / `select_copy`）与 `conftest.py`、`skippers.py`。
  - MCP：`ping → pong`；`remote_ls("")` 列出 `/data` 下已部署 ntops5/6/10/11/13。
  - GPU 探测：`run_python` 确认 MetaX C500、CUDA available、`F.pixel_unshuffle` 在 MetaX 上**已返回 contiguous**（关键发现 —— 意味着底层已在做数据拷贝，必须有真 kernel）。
  - 参考资料：派生子代理读 `CODE_TEMPLATES.md`（Pattern 13：1D copy）、`TAXONOMY.md`（layout/view 族）、`API_REFERENCE.md`（tile/ravel/flatten）、`LAYOUT.md`（space-to-depth / pixel_unshuffle 指引）、`SKILL.md`（Step 0.8 + LIMITATION REPORT 模板）。
  - 本地无 GPU，全程走 MCP 远程。

- **Step 0.5 — 能力可行性判定**
  - 未输出独立的 `CAPABILITY REPORT` 块（因为九齿能表达本算子），但**在 thinking 中显式给出能力判定**：
    - 不在 "无法表达" 清单（无 scatter atomic、无动态控制流、无跨 tensor 索引）
    - 主操作虽是 view-only（O(1) 元数据），但 MetaX 上 `F.pixel_unshuffle` 已经返回 contiguous，意味着必须做真数据搬运
  - 结论：九齿**能表达**，走 Pattern 13（1D copy kernel）+ wrapper-heavy 路径。

- **Step 0.8 — 实现方案确认（📋 IMPLEMENTATION PROPOSAL）**
  - 按新加的 Step 0.8 输出了 5 项简报：
    1. **分类结论**：layout/view 族（重排类）；依据 TAXONOMY §9 + LAYOUT.md
    2. **倾向方案（Plan A）**：wrapper-heavy + Pattern 13 1D copy kernel；wrapper 做 shape 校验 + 3D/4D + view+permute，kernel 做 `output = source`
    3. **GPU kernel 判断**：**需要**（非纯 view，要做真数据搬运）
    4. **潜在风险**：
       - MetaX 非连续 tensor flatten + 1D copy 可能 stride 处理有问题
       - 大 `downscale_factor`（r=8，r*r=64 channels）可能 memory access 不友好
       - fp16 精度对齐不需要（纯数据搬运）
    5. **备选方案**：
       - A. Wrapper-heavy + Pattern 13（推荐）
       - B. 纯九齿 arrangement（tile+ravel+permute），更"九齿原生"但复杂
       - C. 纯 PyTorch 回退（无 kernel，直接 `F.pixel_unshuffle`）
  - **用户选择 Plan A**，Agent 才开始写代码。

- **Step 1 — 分类路由**
  - TAXONOMY 家族：`layout/view`（重排类）
  - 依据：数学本质是 `reshape + permute + reshape`；与 `channel_shuffle` 同族；命中 LAYOUT.md §"pixel_unshuffle 指导"。

- **Step 2 — 代码生成**
  - **使用九齿**：标准的 `arrangement` + `application` + `premake` 模式（Pattern 13）；**没有绕过九齿走纯 Triton**（与 scatter_add 事故形成对照）。
  - **kernel 核心**：`def application(source, output): output = source  # noqa: F841`
  - **arrangement**：`source.flatten().tile((block_size,)), output.flatten().tile((block_size,))`
  - **wrapper**：3D/4D 分支，`view + permute` 得到非连续中间 tensor，再交给 Pattern 13 kernel 做真数据搬运；`block_size=4096`（优化后）。
  - **新增文件**：
    - `src/ntops/kernels/pixel_unshuffle.py`（30 行）
    - `src/ntops/torch/pixel_unshuffle.py`（54 行，block_size=4096）
    - `tests/test_pixel_unshuffle.py`（52 行，12 个 test case）
  - **修改文件**：
    - `src/ntops/kernels/__init__.py`（注册 `pixel_unshuffle`）
    - `src/ntops/torch/__init__.py`（同上）

- **Step 2.5 — 测试循环（2 轮失败 → 第 3 轮 PASS）**
  - **R1（FAIL）**：
    - 症状：`element_wise.arrangement` 中 `assert all(tensor.ndim == ndim ...)` 失败（source 5D vs output 4D）
    - 根因：初版误引了 `element_wise.arrangement`，但 view+permute 后的中间 tensor ndim 不等于输出 ndim
    - 修复：写自定义 `arrangement`，先 `flatten()` 到 1D 再 `tile`
  - **R2（FAIL）**：
    - 症状：shape 匹配但数值错，`max_diff = 4.08577299118042`
    - 根因：`permute` 维度顺序写错
      - 4D 错误：`view(N,C,r,r,H,W).permute(0,1,4,2,5,3)`
      - 3D 错误：`view(C,r,r,H,W).permute(0,3,1,4,2)`
    - 修复：用 `(1,1,4,4)` 小 tensor 做逐元素对比，trace 索引映射后推导出正确 permute：
      - 4D 正确：`view(N,C,H,r,W,r).permute(0,1,3,5,2,4)`
      - 3D 正确：`view(C,H,r,W,r).permute(0,2,4,1,3)`
  - **R3（PASS）**：quick test `match=True, max_diff=0.000000` 对全部 4 组快速用例
  - **优化后复测**：block_size=4096，**12/12 PASSED in 10.29s**
  - **未触发 FIX_CARDS**：R1/R2 都是"代码级 bug"而非"九齿语法 / 平台 API / 精度陷阱"，直接修即可

- **Step 2.8 — 性能优化（block_size sweep + host/GPU 拆分）**
  - **host overhead 诊断**：host 占比 0–6%，**瓶颈在 GPU kernel 本身**，不是 Python dispatch
  - **block_size sweep**（medium shape 1M elements）：
    | block | ntops (ms) | speedup |
    |---|---:|---:|
    | 128 | 0.1646 | 0.24x |
    | 256 | 0.0861 | 0.46x |
    | 512 | 0.0478 | 0.83x |
    | 1024 | 0.0468 | 0.85x |
    | 2048 | 0.0467 | 0.85x |
    | **4096** | **0.0470** | **0.84x** |
    | 8192 | 0.0568 | 0.70x |
  - 选定 `block_size=4096` 作为默认（最优区间的上限端）

## 产出补丁摘要

- **新增文件**：
  - `src/ntops/kernels/pixel_unshuffle.py` — Pattern 13 1D copy kernel
  - `src/ntops/torch/pixel_unshuffle.py` — PyTorch wrapper（3D/4D + view+permute）
  - `tests/test_pixel_unshuffle.py` — pytest 正确性测试（12 case）
- **修改文件**：
  - `src/ntops/kernels/__init__.py`（import + `__all__` 注册）
  - `src/ntops/torch/__init__.py`（同上）
- **关键代码片段（kernel application）**：

```python
# src/ntops/kernels/pixel_unshuffle.py
def arrangement(source, output, block_size=None):
    if block_size is None:
        block_size = ninetoothed.block_size()
    return (
        source.flatten().tile((block_size,)),
        output.flatten().tile((block_size,)),
    )

def application(source, output):
    output = source  # noqa: F841
```

```python
# src/ntops/torch/pixel_unshuffle.py（4D 分支节选）
intermediate = input.view(N, C, H, r, W, r).permute(0, 1, 3, 5, 2, 4)
kernel = _cached_make(
    ntops.kernels.pixel_unshuffle.premake,
    source_ndim, output_ndim, input.dtype, 4096,
)
kernel(intermediate, out)
```

## Correctness 测试

命令：`python -m pytest tests/test_pixel_unshuffle.py -v`（MetaX C500，`/data/ntops14`）

- **R1/R2**：FAIL（arrangement ndim / permute 顺序）
- **R3**：12/12 PASSED in 13.13s
- **优化后（block_size=4096）**：**12/12 PASSED in 10.29s**

测试矩阵覆盖：

| 维度 | 覆盖范围 |
|---|---|
| 输入维度 | 3D `(C,H,W)` / 4D `(N,C,H,W)` |
| dtype | float32 / float16 |
| downscale_factor | 2 / 3 / 4 |
| Shape | 4D `(2,3,H*r,W*r)`、3D `(4,H*r,W*r)` |
| 异常输入 | H/W 不整除 r（assert）、5D 输入（ValueError） |
| 比对 | `torch.allclose(result, F.pixel_unshuffle(input, r), rtol=1e-3, atol=1e-3)` |
| 连续性 | `result.is_contiguous() == True` |

结果：

- 4D × 3 r × 2 dtype = **6/6 PASS**
- 3D × 2 r × 2 dtype = **4/4 PASS**
- 异常输入（invalid_shape / invalid_ndim）= **2/2 PASS**
- **合计 12/12**；`max_diff = 0.000000`（纯数据搬运，无精度损失）

## Benchmark（MetaX C500，CUDA events，block_size=4096）

命令：`python tests/bench_pixel_unshuffle.py`

| 配置 | dtype | nt (ms) | torch (ms) | Speedup | nt BW | torch BW |
|---|---|---:|---:|---:|---:|---:|
| (2,3,64,64) r=2 4D | fp32 | 0.0561 | 0.0147 | 0.26x | 1.8 G/s | 6.7 G/s |
| (2,3,64,64) r=4 4D | fp32 | 0.0564 | 0.0146 | 0.26x | 1.7 G/s | 6.7 G/s |
| (4,16,128,128) r=2 4D | fp32 | 0.0566 | 0.0382 | **0.68x** | 74.0 G/s | 109.7 G/s |
| (4,16,128,128) r=4 4D | fp32 | 0.0557 | 0.0382 | **0.69x** | 75.3 G/s | 109.8 G/s |
| **(8,64,256,256) r=2 4D** | fp32 | **0.9261** | **0.9994** | **1.08x ✓** | **144.9 G/s** | 134.3 G/s |
| (1,3,512,512) r=2 4D | fp32 | 0.0563 | 0.0301 | 0.54x | 55.9 G/s | 104.4 G/s |
| (2,3,64,64) r=2 4D | fp16 | 0.0554 | 0.0152 | 0.27x | 0.9 G/s | 3.2 G/s |
| (4,16,128,128) r=2 4D | fp16 | 0.0561 | 0.0392 | **0.70x** | 37.4 G/s | 53.4 G/s |
| **(8,64,256,256) r=2 4D** | fp16 | **0.9181** | **1.0001** | **1.09x ✓** | **73.1 G/s** | 67.1 G/s |
| (3,32,32) r=2 3D | fp32 | 0.0500 | 0.0136 | 0.27x | 0.2 G/s | 0.9 G/s |
| (16,128,128) r=2 3D | fp32 | 0.0500 | 0.0144 | 0.29x | 21.0 G/s | 72.6 G/s |

**性能结论**：

- **memory-bound**：纯数据搬运，带宽利用率直接反映性能
- **小 shape（<100K elements）**：受 MetaX GPU kernel launch 最低延迟（~0.04ms）主导，speedup 0.26–0.29x
- **大 shape（33M elements，`(8,64,256,256)`）**：ntops **反超 torch**（1.08x fp32 / 1.09x fp16），带宽利用率 144.9 vs 134.3 GB/s（fp32）
- **结论**：小 shape 未达标（launch overhead 框架上限），大 shape 达标且反超

## 失败诊断

### R1：arrangement ndim 不匹配（已修复）

- **失败现象**：`AssertionError` in `ntops.kernels.element_wise.arrangement`，assertion `all(tensor.ndim == ndim or tensor.ndim == 0 for tensor in tensors)` 失败
- **根因判断**：代码级 bug（非 FC 卡片），初版错用了 `element_wise.arrangement`，但 view+permute 后的中间 tensor ndim（5）不等于输出 ndim（4）
- **修复方案**：写自定义 `arrangement`，先 `flatten()` 到 1D 再 `tile`，彻底与 ndim 解耦
- **验证闭环**：R2 通过 ndim 校验，但暴露 R2 permute 顺序错

### R2：permute 维度顺序错误（已修复）

- **失败现象**：shape 匹配但数值错，`max_diff = 4.08577299118042`
- **根因判断**：代码级 bug（非 FC 卡片），推导 pixel_unshuffle 的 reshape+permute 索引映射时出错
  - 4D 错：`view(N,C,r,r,H,W).permute(0,1,4,2,5,3)`
  - 3D 错：`view(C,r,r,H,W).permute(0,3,1,4,2)`
- **修复方案**：用 `(1,1,4,4)` 小 tensor 做逐元素 trace，对比 `F.pixel_unshuffle` 真值，反推正确映射：
  - 4D 对：`view(N,C,H,r,W,r).permute(0,1,3,5,2,4)`
  - 3D 对：`view(C,H,r,W,r).permute(0,2,4,1,3)`
- **验证闭环**：R3 quick test `match=True, max_diff=0.0`；完整 pytest 12/12 PASS

### 未触发 FIX_CARDS

两次失败都是"代码级推导错误"（permute 维度顺序、arrangement 选型），不是九齿语法陷阱 / 平台 API / 精度问题，因此 **没有 FC-XX 卡片被触发**。Agent 直接根据测试反馈修复，未走诊断流程。

### 未重新触发 Step 0.8

R1/R2 失败后 Agent **没有重新输出 `📋 IMPLEMENTATION PROPOSAL`**，因为失败都在 Plan A 框架内（wrapper-heavy + Pattern 13），不涉及方案切换。这符合 Step 0.8 设计意图（只在"换方案"时才重新确认）。

## 不支持用例

- **输入维度**：仅支持 3D / 4D；5D+ 不支持（wrapper 会抛 `ValueError`）
- **dtype**：仅 float32 / float16；bfloat16 / int / bool 不支持
- **downscale_factor**：正整数；非整数 / ≤0 由 wrapper assert 拦截
- **H/W 不整除 r**：wrapper assert 拦截（`pytest invalid_shape` PASS）
- **非连续输入**：未测（pixel_unshuffle 语义上要求 contiguous input）
- **小 shape 性能**：<100K elements 时 0.26–0.29x（MetaX GPU launch 最低延迟框架上限，无法通过 tile 消除）

## Skill 效能小结

| 维度 | 表现 |
|---|---|
| **Step 0.8 方案先行确认** | ✅ 输出完整 5 项简报，等用户选 Plan A 后才动笔；与 scatter_add 事故形成对照（那次是 silently 切换方案） |
| **TAXONOMY 分类命中** | layout/view 族 → Pattern 13 1D copy + wrapper-heavy，命中 LAYOUT.md 指引 |
| **避免绕过九齿** | ✅ kernel 是真正的 Pattern 13 九齿 kernel（`output = source`），不是纯 Triton `@triton.jit` |
| **MetaX 平台特性探测** | Step 0 发现 `F.pixel_unshuffle` 在 MetaX 上已返回 contiguous，避免了"view-only wrapper"死路 |
| **失败修复效率** | R1/R2 两轮代码级 bug 修复，未走 FC 诊断流程；R3 即 PASS |
| **性能优化闭环** | block_size sweep 7 档 + host/GPU 拆分，识别 launch overhead 框架上限 + 大 shape 反超 |
| **与 scatter_add 事故的关键差异** | scatter_add 因 TAXONOMY 未内置"scatter 族"而悄悄换方案；pixel_unshuffle 命中 layout/view 族，Agent 老老实实走 Step 0.8 + Pattern 13 |

> **结论**：本任务是 Step 0.8 方案先行确认机制的**正面验证** —— Agent 在动笔前输出完整 5 项简报，用户选 Plan A 后才开始写代码，全程没有"绕过九齿"或"silent fallback"。与 `ntops13/scatter_add` 形成对照，证明 Step 0.8 对 layout/view 族算子（之前被绕过九齿的重灾区）的防护有效。
