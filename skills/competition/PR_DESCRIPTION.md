# [2026春季][T3-1-1] 王一鸣 — NineToothed 算子开发 Skill

## 一、基本信息

- **.skill 名称**：ninetoothed-skill
- **赛题编号**：T3-1-1（NineToothed 算子开发 Skill）
- **小组名称**：王一鸣
- **成员**：王一鸣（1 人）
- **提交目录**：`skills/competition/ninetoothed-skill/`

## 二、适用任务范围与不适用范围

### 适用范围

安装本 .skill 后，AI 智能体可完成以下类型的 NineToothed 算子开发任务：

- **逐元素 / 广播算子**：覆盖 dtype 约束、broadcast 约束、mask 约束
- **归约 / 分块算子**：覆盖 reduce、softmax 子任务、分块统计、直方图（`ninetoothed.language.histogram`）
- **布局敏感算子**：覆盖 non-contiguous、stride、offset 输入
- **性能 / 诊断 / 集成任务**：benchmark 设计、generated source 检查、AOT build 配置、性能回退定位
- **9 种 Arrangement 模式**：Element-wise, Reduction, Matmul, BMM, Addmm, Conv2D, Attention, Pooling, RoPE

### 不适用范围

- 动态 shape（NineToothed 的 premake 需要固定 ndim）
- 递归算法
- 数据依赖 while 循环（已提供固定 range 规避方案，有性能代价）
- view vs copy 类算子（flatten, chunk, unbind 等使用 kernel 拷贝时性能无法匹敌 PyTorch 零拷贝 view）
- 大规模 matmul（Windows 环境 Triton 优化不足，Linux 下预期接近 PyTorch）
- Triton `sort` 算子（无 Python 源码，code generator 无法内联）
- 联网依赖或需要 API key 的服务

## 三、安装与使用方式

### 安装

```bash
# 方式 1：复制到 Claude Code skills 目录
cp -r skills/competition/ninetoothed-skill/ ~/.claude/skills/ninetoothed-skill/

# 方式 2：直接向 AI 智能体提供 SKILL.md
```

### 依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| Python | ≥3.10 | 运行环境 |
| PyTorch | ≥2.0 | CPU 参考 + torch 层 |
| Triton | ≥3.0 | GPU kernel 编译后端 |
| NineToothed | ≥0.25 | DSL 框架 |
| ntops | ≥0.1 | 算子库 |

### 使用

向 AI 智能体提供 CPU 参考实现，Skill 自动完成六阶段闭环：分析 → 生成 → 编译 → 验证 → 优化 → 报告。

## 四、自测任务运行记录

### 自测任务总览

| # | 任务 | 类型 | 文件 | Benchmark | 结果 |
|---|------|------|------|:--:|:--:|
| 1 | leaky_relu | Element-wise + 标量参数 | `tests/test_leaky_relu.py` | ✅ | 全部通过 |
| 2 | log_softmax | Reduction + 数值稳定性 | `tests/test_log_softmax.py` | ✅ | 全部通过 |
| 3 | 非连续输入 | 转置/步幅/偏移 | `tests/test_non_contiguous.py` | — | 全部通过 |
| 4 | 性能对比 | Benchmark + 回退分析 | `tests/test_benchmark.py` | — | 已完成 |
| 5 | meshgrid | 1D→2D 广播 | `tests/test_meshgrid.py` | — | 全部通过 |
| 6 | deg2rad | Element-wise + constexpr | `tests/test_deg2rad.py` | — | 全部通过 |

### 运行命令

```bash
# Skill 自身验证（55 项检查）
python ninetoothed-skill/scripts/verify_skill.py

# 运行全部自测
python ninetoothed-skill/scripts/run_tests.py

# 运行性能基准测试
python ninetoothed-skill/scripts/run_benchmark.py
```

### 扩展验证

除 6 个自测任务外，额外完成了 24 个算子的端到端开发和验证，总计 **30 个算子**全部精度测试通过。每个算子均有独立的开发报告（`test_log/` 目录下）。

## 五、AI 智能体使用 Skill 前后对比

| 维度 | 无 .skill（预估） | 有 .skill（实测 30 算子） | 改进 |
|------|:--:|:--:|:--:|
| 分类准确率 | ~60% | 100% | +67% |
| 首次编译通过率 | ~30% | 80%（24/30） | +167% |
| 平均迭代次数 | 5-10 次 | 1.6 次 | -70% |
| 精度测试通过率 | ~60% | 100%（30/30） | +67% |
| 性能意识 | 无 | 有（100%） | 从无到有 |

详细对比数据见 `BEFORE_AFTER.md`。

## 六、附件说明

- **HONOR_CODE.md** — 诚信守则（已签署）
- **REFERENCE.md** — 引用披露
- **BEFORE_AFTER.md** — AI 智能体使用 Skill 前后对比
- **Proposal** — `王一鸣_九齿skill创新挑战_proposal.pdf`
- **中期报告** — `王一鸣_九齿skill创新挑战_中期报告.pdf`
- **自测计划** — `王一鸣_九齿skill创新挑战_自测计划.pdf`
- **最终赛题报告** — `王一鸣_九齿skill创新挑战_T3-1-1_赛题报告.md`
- **算子开发报告** — `test_log/` 目录（34 份）

## 七、目录结构

```
skills/competition/
├── ninetoothed-skill/              .skill 包（20 个文件）
│   ├── SKILL.md                    主 Skill 文件
│   ├── README.md
│   ├── references/（4 个参考文件）
│   ├── examples/（6 个文档）
│   ├── scripts/（3 个脚本）
│   └── tests/（6 个测试文件）
├── HONOR_CODE.md
├── REFERENCE.md
├── BEFORE_AFTER.md
├── proposal.pdf / 中期报告.pdf / 自测计划.pdf / 赛题报告.md
└── test_log/（34 份算子开发报告）
```
