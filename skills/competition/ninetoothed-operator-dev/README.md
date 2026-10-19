# ninetoothed-operator-dev

九齿算子开发 Skill — 指导 AI 智能体完成 NineToothed 算子编写、验证、优化和调试。

> 赛题编号 T3-1-1 · 2026 春季人工智能大赛 · 九齿 .skill 创新挑战赛道

## 安装

将此 skill 目录安装到 AI 智能体的 skills 目录中。包内无外部依赖、无 API key、无联网要求，可离线使用。

```
skills/ninetoothed-operator-dev/
├── SKILL.md                   # 主文件：触发条件、工作流、约束
├── README.md                  # 本文件
│
├── operators/                 # 9 个算子参考实现
│   ├── add.py                 #   一维逐元素加法
│   ├── add_2d.py              #   二维逐元素加法
│   ├── relu.py                #   ReLU 激活
│   ├── sigmoid.py             #   Sigmoid 激活
│   ├── gelu.py                #   GELU（tanh 近似，sigmoid 恒等式）
│   ├── softmax.py             #   二维 Softmax 归约
│   ├── sum.py                 #   一维 Sum 归约
│   ├── strided_add.py         #   非连续/步长加法 (1D + 2D)
│   └── rms_norm.py            #   RMS Normalization 融合算子
│
├── run_all_operator_tests.py  # 统一 correctness 测试（25 用例）
│
├── benchmarks/                # 性能验证
│   ├── bench.py               #   封装 triton.testing.Benchmark
│   └── run_benchmarks.py      #   5 类 11 项 benchmark
│
├── tests/                     # 自测材料
│   ├── validate_all.py        #   一键验证脚本
│   ├── bench_light.py         #   轻量 benchmark（CUDA Event 计时）
│   └── self_eval_*.md         #   5 个自测任务文档
│
├── examples/
│   └── usage_example.py       #   完整使用示例
│
├── assets/                    # 新算子开发模板
│   ├── operator_template.py
│   ├── test_template.py
│   └── benchmark_template.py
│
└── references/                # 知识库
    ├── nine_toothed_api.md    #   API 速查
    ├── operator_patterns.md   #   设计模式
    └── migration_guide.md     #   Triton → 九齿 迁移指南
```

## 触发条件

AI 智能体收到以下指令时激活此 skill：

- 实现一个 NineToothed 算子（逐元素、归约、非连续、融合）
- 编写 correctness 测试
- 运行 benchmark 并分析性能
- 检查 generated source
- 诊断测试失败或性能回退
- 将 Triton kernel 迁移到九齿

## 快速验证

### 仅检查（不需要 GPU）

```bash
python tests/validate_all.py --check-only
```

检查：项目结构 (28 文件) → Python 语法 (16 文件) → 模块导入 (10 个算子) → 代码规范 (9 个算子)

### 完整验证（需要 GPU）

```bash
python tests/validate_all.py --quick
```

检查：结构 + 语法 + 导入 + 规范 + correctness 测试 (25 用例) + benchmark (stride vs contiguous)

### 单项验证

```bash
# 仅 correctness
python run_all_operator_tests.py

# 仅 benchmark（快速）
python tests/bench_light.py

# 完整 benchmark 套件
python benchmarks/run_benchmarks.py --quick

# 仅 stride vs contiguous 分析
python benchmarks/run_benchmarks.py --category analysis --quick
```

## 算子覆盖

| 类别 | 算子 | 维度 | 测试规模 |
|------|------|------|---------|
| 逐元素 | add | 1D | (100,)(512,)(1024,) |
| 逐元素 | add_2d | 2D | (100,50)(512,512)(1024,1024) |
| 逐元素 | relu | 1D | (100,)(512,)(1024,) |
| 逐元素 | sigmoid | 1D | (100,)(512,)(1024,) |
| 逐元素 | gelu | 1D | (100,)(512,)(1024,) |
| 归约 | softmax | 2D | (512,256)(1024,512) |
| 归约 | sum | 1D | (100,)(512,)(1024,) |
| 非连续 | strided_add | 1D+2D | 6 规模, stride=2 |
| 融合 | rms_norm | 2D | (128,256)(512,512)(1024,512) |

## 环境要求

- CUDA GPU（已在 RTX 4060 8 GB 上验证）
- PyTorch >= 2.0 + Triton >= 3.0
- ninetoothed（从源码安装: `pip install git+https://github.com/InfiniTensor/ninetoothed.git`）

## 自测任务

详见 `tests/` 目录：

| 编号 | 任务 | 类型 |
|------|------|------|
| 1 | GELU 激活函数 | 逐元素 + benchmark |
| 2 | Softmax 归约 | 归约 + benchmark |
| 3 | Strided Add | 非连续 + benchmark |
| 4 | 性能回退分析 | 诊断 + benchmark |
| 5 | Triton Add 迁移 | 迁移 |

## 关于九齿

[NineToothed](https://github.com/InfiniTensor/ninetoothed) 是基于 Triton 的高层 DSL，通过面向张量的元编程 (Tensor-Oriented Metaprogramming) 简化 GPU 算子开发。核心概念：

- **Symbol**: 编译期符号变量（BLOCK_SIZE 等）
- **Symbolic Tensor**: 不存储数据、仅描述 shape/stride 的符号张量
- **tile / expand / squeeze / permute**: 编译期元操作
- **Arrange-and-Apply**: 九齿编程范式——arrangement 定义分块排布，application 定义计算逻辑

## 作者

- **独钓寒江雪** ([@73fc](https://github.com/73fc)) · 542591851@qq.com
- 赛题 T3-1-1 · 2026 春季人工智能大赛 · 九齿 .skill 创新挑战赛道

## 许可证

Apache-2.0
