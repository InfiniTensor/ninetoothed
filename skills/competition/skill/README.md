# .skill — ninetoothed DSL Agent Workspace

> 本 `.skill` 工作区是 AI Agent 的 **技能包 (Skill Package)**，使 agent 能够高效实现、测试、基准分析和诊断基于 **ninetoothed DSL** 的 GPU 算子。所有文档、模板、脚本、示例均已内置，agent 可自主完成从实作到报告的全流程。

## 概览

| 目录 | 用途 |
|------|------|
| `references/` | DSL 模式、测试模式、Benchmark 模式、Repo 索引、AOT 指南、故障诊断 |
| `scripts/` | 正确性测试、Benchmark、源码检查、日志收集的可执行脚本 |
| `templates/` | Agent 任务报告、Benchmark 报告、故障诊断报告模板 |
| `examples/` | 4 个完整示例项目（含源码 + benchmark） |
| `tests/` | Agent 触发 prompt、自校验任务、期望输出参考 |

## 快速开始

### 1. 实现一个算子

```bash
# 1a. 参考 DSL 模式
cat references/dsl_patterns.md

# 1b. 参考已有示例（如 elementwise 加法）
cat examples/elementwise_broadcast_add/run.py

# 1c. 实现自己的 kernel
```

### 2. 运行正确性测试

```bash
scripts/run_correctness.sh examples/elementwise_broadcast_add
```

### 3. 运行 Benchmark

```bash
scripts/run_benchmark.sh examples/elementwise_broadcast_add
```

### 4. 查看生成源

```bash
scripts/inspect_generated_source.sh examples/elementwise_broadcast_add/run.py
```

### 5. 收集日志

```bash
python scripts/collect_task_log.py --dir . --output task_log.json
```

## 文件结构

```
.skill/
├── README.md                        ← 本文档
├── SKILL.md                         ← DSL 完整 API 参考
├── references/
│   ├── repo_index.md                ← ninetoothed 仓库结构索引
│   ├── dsl_patterns.md              ← 7 种 DSL 实现模式
│   ├── testing_patterns.md          ← 4 维度测试覆盖策略
│   ├── benchmark_patterns.md        ← 8 元素 Benchmark 设计
│   ├── generated_source_and_aot.md  ← Codegen 查看 + AOT 编译
│   └── failure_diagnosis.md        ← 4 类故障诊断指南
├── scripts/
│   ├── validate_skill_package.py    ← 结构完整性检查
│   ├── run_correctness.sh           ← 正确性测试运行器
│   ├── run_benchmark.sh             ← Benchmark 运行器
│   ├── inspect_generated_source.sh  ← 生成源码查看器
│   └── collect_task_log.py          ← 任务日志收集器
├── templates/
│   ├── operator_task_report_template.md   ← 算子任务报告模板
│   ├── benchmark_report_template.md       ← Benchmark 报告模板
│   └── failure_diagnosis_template.md      ← 故障诊断模板
├── examples/
│   ├── elementwise_broadcast_add/   ← 加法 kernel (elementwise_1d)
│   ├── reduction_softmax/           ← Softmax kernel (reduction_2d)
│   ├── non_contiguous_stride_case/  ← 非连续 stride 测试
│   └── performance_regression_case/ ← BLOCK_SIZE 退化诊断
└── tests/
    ├── trigger_prompts.md           ← Agent 触发 prompt
    ├── selftest_tasks.md            ← 自我校验任务
    └── expected_outputs.md          ← 期望输出参考
```

## Agent 工作流程

当 agent 收到"实现一个 XX 算子"的请求时，典型工作流如下：

```
1. 理解需求 ──→ 打开 references/dsl_patterns.md，匹配模式
                    │
2. 查看模板 ──→ 打开 templates/operator_task_report_template.md
                    │
3. 参考示例 ──→ 查看 examples/ 下相同模式的实现
                    │
4. 实现代码 ──→ 编写 kernel.py + run.py + benchmark.py
                    │
5. 正确性测试 ──→ scripts/run_correctness.sh 验证
                    │
6. Benchmark  ──→ scripts/run_benchmark.sh 性能对比
                    │
7. 查看源码  ──→ scripts/inspect_generated_source.sh 检查
                    │
8. 故障诊断  ──→ (如遇错误) 参考 failure_diagnosis.md
                    │
9. 生成报告  ──→ 填写 operator_task_report_template 完成
```

## 环境要求

- Python 3.10+
- PyTorch 2.0+ (CUDA)
- ninetoothed (git@github.com:QuantumIntelligence/ninetoothed.git)
- NVIDIA GPU with CUDA support

## 结构校验

```bash
python scripts/validate_skill_package.py
```

预期输出：
```
✅ .skill structure OK (所有 5 个目录和核心文件均存在)
```

## License

Internal — Qiyuan Competition
