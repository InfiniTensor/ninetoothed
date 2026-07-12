# NineToothed - 九齿编译优化 T1-2-1

## 项目信息

- **项目**: NineToothed — 基于 Triton 的 DSL，通过 tensor-oriented meta-programming 编写 GPU kernel
- **赛题**: 九齿编译优化 T1-2-1 — NineToothed 代码生成特化增强挑战
- **比赛**: 2026 春季人工智能大赛
- **仓库**: https://github.com/InfiniTensor/ninetoothed
- **文档**: https://ninetoothed.org/
- **Python 版本**: >= 3.10
- **CUDA**: 驱动版本 570.124.06 (CUDA 12.8)；PyTorch 需匹配驱动版本，不可使用 CUDA 13.0 编译的 PyTorch
- **依赖**: triton>=3.0.0, sympy>=1.13.0, numpy>=1.26.4
- **环境**: 不使用 `.venv`，直接使用系统 `/usr/bin/python`。`torch` 和 `triton` 已安装在系统 site-packages 中
- **赛题文档**: `/data/ninetoothed/instructions.txt`

## 环境变量（必要）

```bash
# torch 链接了 /opt/hpcx/ucc/lib/libucc.so.1，需要 /opt/hpcx/ucx/lib 中的 libucs.so 提供 ucs_config_doc_nop 符号
export LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:/usr/local/cuda/lib64:/usr/local/nccl/lib:/opt/hpcx/ucc/lib
export PYTHONPATH=/data/ninetoothed/src
```

## 关键文件

| 文件 | 用途 |
|------|------|
| `src/ninetoothed/generation.py` | **代码生成主逻辑** — `CodeGenerator` AST 转换器，生成 Triton 源码 |
| `src/ninetoothed/aot.py` | **AOT 编译** — 变体枚举、dispatcher 生成、编译 |
| `src/ninetoothed/tensor.py` | **Tensor 类** — 符号张量的 tile/expand/squeeze/permute 等操作 |
| `src/ninetoothed/symbol.py` | **Symbol 类** — AST 节点符号包装，支持算术运算、范围约束 |
| `src/ninetoothed/language.py` | 语言层辅助 — `call`, `attribute` 等 |
| `src/ninetoothed/naming.py` | 命名工具 — `auto_generate`, `make_constexpr`, `make_meta`, prefix 管理 |
| `src/ninetoothed/jit.py` | JIT 编译入口 — 调用 `CodeGenerator` 生成并加载 kernel |
| `src/ninetoothed/make.py` | `make()` 入口 — 整合 arrangement/application/tensors |
| `src/ninetoothed/build.py` | `build()` 入口 — 多配置构建 |
| `src/ninetoothed/dtype.py` | 数据类型定义 |
| `src/ninetoothed/torchifier.py` | Torch 调用适配 |
| `src/ninetoothed/cudaifier.py` | CUDA 调用适配 |
| `src/ninetoothed/utils.py` | 工具函数 — `calculate_default_configs()` |

## 开发命令

```bash
# 环境设置（每次新 shell 需要）
export LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:/usr/local/cuda/lib64:/usr/local/nccl/lib:/opt/hpcx/ucc/lib
export PYTHONPATH=/data/ninetoothed/src

# 测试（核心 — 代码生成，无需 nvcc，~9s）
python -m pytest tests/test_generation.py -v

# 特化测试（~2s）
python -m pytest tests/test_specialization.py -v

# AOT 测试（需要 nvcc，慢，~90s+ per case，非必要不运行）
# python -m pytest tests/test_aot.py -v --timeout=300

# 格式化/检查
ruff format
ruff check
python scripts/check_contributing_style.py
python scripts/check_contributing_style.py --fix  # 自动修复

# 完整测试（所有非 AOT 测试）
python -m pytest tests/ --ignore=tests/test_aot.py --ignore=tests/test_ipynb.py -v

# 完整 CI（使用 tee 同时输出到终端和日志）
python scripts/check_contributing_style.py --fix
ruff format
ruff check
python scripts/check_contributing_style.py
python -m pytest tests/test_generation.py tests/test_specialization.py -v 2>&1 | tee /data/ninetoothed/non-deliverable/logs/test_ci.log
```

## 交付物目录

| 路径 | 用途 |
|------|------|
| `deliverable/final_report.md` | **赛题报告** |
| `deliverable/REFERENCE.md` | 参考资料披露 |
| `deliverable/HONOR_CODE.md` | 诚信守则 |

## 非交付物目录

| 路径 | 用途 |
|------|------|
| `non-deliverable/reports/` | 阶段报告、参考分析等 (phase1, phase2, helion_reference, v0.0.1) |
| `non-deliverable/benchmarks/` | Benchmark 输出 (JSON/CSV) |
| `non-deliverable/logs/` | **所有日志** — 测试日志、benchmark 日志、运行日志等 |
| `non-deliverable/tutorials/` | 学习 notebook (tile/arrangement, AST basics, codegen trace) |

## 日志规范

- 所有测试输出、benchmark 输出、脚本运行日志均保存到 `/data/ninetoothed/non-deliverable/logs/`
- 使用 `2>&1 | tee <logfile>` 将 stdout+stderr 同时输出到终端和日志文件
- 日志文件名应包含时间戳或明确的用例标识，便于追溯

## 代码生成流程

1. `make()` → `jit()` or `aot()`
2. `CodeGenerator` 接收 `application` 函数 → 解析 AST → AST 变换
3. 生成 Triton `@triton.jit` kernel + launch function
4. 缓存到 `~/.ninetoothed/<sha256>.py`
5. 通过 `importlib` 加载或 AOT 编译为共享库

### 关键代码生成方法 (`generation.py`)

- `_generate_load()` — 生成 `tl.load(pointers, mask=mask, other=other)`
- `_generate_store()` — 生成 `tl.store(pointers, value, mask=mask)`
- `_generate_pointers_and_mask()` — 生成 pointer 表达式和 mask（特化入口）
- `_generate_overall_offsets_and_mask()` — 计算 `sum(offsets[dim] * stride[dim])` 和边界 mask
- `_generate_offsets_and_mask()` — 递归遍历 tensor 层级，生成 offset 和 mask 约束
- `_is_effectively_zero_stride()` — 检测 size-1 维度或广播零偏移
- `_try_get_constant_int()` — 编译期常数提取（divisible tile 检测用）
- `_generate_autotune()` — 生成 auto-tuning configs
- `_generate_launch()` — 生成 launch function

### AOT 编译流程 (`aot.py`)

1. `CodeGenerator` 生成通用 Triton 源码（获取结构性信息）
2. `_enumerate_variant_specs()` — 枚举所有 divisibility/contiguity 变体组合
3. 每个变体用 `divisibility_hints` / `contiguity_hints` 重新调用 `CodeGenerator` 生成变体专用源码
4. `_build_variant()` — 编译每个变体
5. `_generate_dispatcher()` — 生成 C++ dispatcher，运行时根据 tensor 属性选择变体

## 赛题 T1-2-1：代码生成特化增强

详细规则见 `/data/ninetoothed/instructions.txt`。

### 允许的特化类别

1. **contiguous fast path** — 当 tensor 连续时，生成线性 pointer 表达式，减少多维 stride/offset 展开
2. **divisible tile fast path** — 当 tile 覆盖无尾块时，生成无 mask 或更少 mask 的 `tl.load`/`tl.store`
3. **broadcast/scalar fast path** — 当广播维度、size-1 维度或标量输入可判定时，简化表达式
4. **layout-known AOT variant** — 利用 AOT 已知的 contiguity/divisibility/stride/shape 约束选择更专门的 variant

### 当前进度 (v0.0.3)

全部 4 个类别已实现，详见 `deliverable/final_report.md`。

| 类别 | 状态 | 关键改动 |
|------|:----:|------|
| Divisible tile | ✅ v0.0.1–v0.0.3 | `skip_lower_bound`/`skip_upper_bound` per-dim 控制; AOT divisibility hints |
| Broadcast/scalar | ✅ v0.0.1–v0.0.2 | `_BinOpSimplifier`; size-1 stride skip; `_is_effectively_zero_stride` |
| Contiguous | ✅ v0.0.1–v0.0.3 | `x*1→x`; `tl.max_contiguous` hint; AOT contiguous 线性化 |
| AOT variant | ✅ v0.0.3 | per-variant `CodeGenerator` 调用; hints 传入 mask/pointer 生成 |

### 测试

| 测试文件 | 数量 | 状态 |
|---------|:----:|:----:|
| `test_generation.py` | 76 | ✅ |
| `test_specialization.py` | 6 | ✅ |
| 其他 (pad/matmul/conv2d/attention 等) | ~114 | ✅ |

### 交付物

- `deliverable/final_report.md` — 赛题报告
- `deliverable/REFERENCE.md` — 参考资料披露
- `deliverable/HONOR_CODE.md` — 诚信守则

### 隐藏评测

| 评测 | 用例数 | 说明 |
|------|--------|------|
| Correctness | 30 | 需 ≥ 29/30 过门槛 |
| Specialization Coverage | 12 | 按有效命中比例计分，误命中扣分 |
| Generated Code Metric | 8 | 按 mask/stride/pointer 改善比例计分 |
| Benchmark | 8 | 按 speedup 线性计分 |
