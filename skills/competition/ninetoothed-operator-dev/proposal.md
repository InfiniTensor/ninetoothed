# NineToothed 算子开发 Skill — Proposal（更新版）

**小组名称**: 独钓寒江雪  
**选手**: 文博（73fc）  
**赛题编号**: T3-1-1  
**Skill 名称**: ninetoothed-operator-dev  
**规范**: Agent Skills 开放格式  

---

## 1. 目标用户与目标任务

**目标用户**: AI 智能体（DeepSeek 等大模型，通过 IDE / Agent 框架调度）

**目标任务**: 指导 AI 智能体完整闭环 NineToothed 算子开发：

1. **算子需求分析**: 从数学定义 / API 规范提取输入输出、shape、dtype、广播和边界条件
2. **DSL 编码**: 使用 Arrange-and-Apply 范式编写算子实现，参考 NineToothed 官方代码风格
3. **正确性验证**: 编写 correctness test，与 PyTorch 参考实现对齐，覆盖多种规模和布局
4. **性能分析**: 运行 benchmark，检查 generated source，分析 stride/dtype/融合对性能的影响
5. **调试与诊断**: 定位测试失败、性能回退，给出最小修复和验证闭环

---

## 2. 覆盖算子类型

覆盖赛题要求的四大类算子，共 9 个完整实现：

| 类别 | 算子 | 技术要点 |
|------|------|---------|
| 逐元素/广播 | add、add_2d、relu、sigmoid、gelu | tile 分块 + elementwise application、dtype 处理(cast f32→f16)、BLOCK_SIZE 定义方式对比(block_size() vs constexpr) |
| 归约/分块 | softmax、sum | 归约维度处理、constexpr 约束（归约算子必须）、max→exp→sum 数值稳定性 |
| 布局敏感 | strided_add (1D+2D) | [::2] 切片 + tile 分块、非合并访存对性能的影响 |
| 融合算子 | rms_norm | tile→expand→归约→逐元素 完整融合流程、weight 广播 |

**技术边界**: 不覆盖动态 shape、float64、>2D 的归约算子、需要 scratch buffer 的矩阵乘法

---

## 3. Skill 包结构

```
ninetoothed-operator-dev/
├── SKILL.md                          # 主文件: 触发条件、工作流、约束
├── README.md                         # 使用说明
│
├── operators/                        # 9 个算子参考实现
│   ├── add.py                        #   一维逐元素
│   ├── add_2d.py                     #   二维逐元素
│   ├── relu.py                       #   ReLU
│   ├── sigmoid.py                    #   Sigmoid
│   ├── gelu.py                       #   GELU (tanh 近似，sigmoid 恒等式)
│   ├── softmax.py                    #   二维归约
│   ├── sum.py                        #   一维归约
│   ├── strided_add.py                #   非连续(1D+2D)
│   └── rms_norm.py                   #   融合归约
│
├── run_all_operator_tests.py         # 统一 correctness 测试 (25 用例)
│
├── benchmarks/                       # 性能验证
│   ├── bench.py                      #   封装 triton.testing.Benchmark
│   └── run_benchmarks.py             #   5 类 11 项 benchmark
│
├── tests/                            # 自测材料
│   ├── validate_all.py               #   一键验证脚本 (结构/语法/导入/correctness/benchmark)
│   ├── bench_light.py                #   轻量 benchmark (CUDA Event 计时)
│   └── self_eval_{1..5}_*.md         #   5 个自测任务文档
│
├── examples/
│   └── usage_example.py              # 完整使用示例
│
├── assets/                           # 开发模板
│   ├── operator_template.py
│   ├── test_template.py
│   └── benchmark_template.py
│
├── references/                       # 知识库
│   ├── nine_toothed_api.md           #   API 速查
│   ├── operator_patterns.md          #   设计模式
│   └── migration_guide.md            #   Triton→九齿 迁移指南
│
└── REFERENCE.md                      # 参考资源披露
```

---

## 4. 核心工作流

### 4.1 触发条件

AI 智能体收到以下指令时激活本 skill:

- "实现一个 GELU/Softmax/... 算子"
- "把这个 Triton kernel 改写成九齿"
- "补充 correctness 测试"
- "运行 benchmark 并分析性能"
- "检查 generated source"
- "测试失败了，诊断原因"

### 4.2 执行流程

```
[用户输入算子需求]
     │
     ▼
① 需求分析
  - 提取 input/output shape、dtype、归约轴
  - 识别广播、非连续布局、边界条件
  - 查看 references/ 中的模式参考
     │
     ▼
② Arrangement 设计
  - 选择 BLOCK_SIZE 定义方式
    · 逐元素 1D: block_size() (meta) 或 Symbol(constexpr=True)
    · 归约算子: 必须 Symbol(constexpr=True)
    · match SKILL.md §2.3 tile 决策速查表
  - tile 各参数张量，确保最外层维度对齐
     │
     ▼
③ Application 编写
  - 使用 ninetoothed.language 原语
  - fp16 运算中 cast 到 fp32 保证精度
  - 输出赋值加 # noqa: F841
     │
     ▼
④ ninetoothed.make() → kernel
     │
     ▼
⑤ Correctness 测试
  - python run_all_operator_tests.py
  - 与 PyTorch 参考实现 torch.allclose 对齐
     │
     ▼
⑥ 性能验证 (需 GPU)
  - 快速: python tests/bench_light.py
  - 完整: python benchmarks/run_benchmarks.py --quick
  - 检查 generated source (~/.ninetoothed/)
  - 对比 PyTorch 基线
     │
     ▼
⑦ 诊断闭环 (失败时)
  - 记录现象 → 定位根因 → 最小修复 → 验证通过
```

---

## 5. 自测任务设计

| 编号 | 任务 | 类型 | 关键验证点 | Benchmark |
|------|------|------|-----------|-----------|
| 1 | GELU 算子 | 逐元素 | sigmoid 恒等式替代 libdevice.tanh、fp32 cast 精度 | 有 (vs torch.gelu) |
| 2 | Softmax 算子 | 归约 | constexpr BLOCK_SIZE 约束、归约维度正确性 | 有 (vs torch.softmax) |
| 3 | Strided Add | 非连续 | [::2] stride 切片、2D tile 分块、边界处理 | 有 (stride vs contiguous) |
| 4 | 性能回退分析 | 诊断 | generated source 检查、非合并访存根因、优化建议 | 有 (4 项对比) |
| 5 | Triton Add 迁移 | 迁移 | 原语对照表、arrangement 替代 pid/offsets、correctness 对齐 | 无 |

每个自测均包含:
- 任务说明与执行记录
- 算子代码 / 修复补丁摘要
- correctness 测试命令和结果
- 失败诊断 (根因 → 修复 → 验证闭环)
- (至少 4 个含) benchmark 基线、命令和结论

详见 `tests/self_eval_*.md`。

---

## 6. Benchmark 设计

### 6.1 测试环境

| 项目 | 值 |
|------|----|
| GPU | NVIDIA GeForce RTX 4060 Laptop (8 GB) |
| CUDA | 13.0 (Driver 581.83) |
| PyTorch | 2.5.1+cu124 |
| AI 智能体 | DeepSeek |

### 6.2 Benchmark 覆盖

| 类别 | 项目数 | 代表性对比 |
|------|--------|-----------|
| elementwise | 5 | add_1d, relu, sigmoid, gelu, add_2d |
| reduction | 2 | softmax, sum |
| noncontiguous | 2 | strided_add_1d, strided_add_2d |
| fused | 1 | rms_norm |
| analysis | 1 | stride vs contiguous 开销分析 |

### 6.3 运行方式

```bash
# 快速验证
python tests/bench_light.py
python benchmarks/run_benchmarks.py --quick

# 一键全量
python tests/validate_all.py --quick
```

---

## 7. 预期量化指标

| 指标 | 目标值 |
|------|--------|
| 自测任务 correctness 通过率 | 100% (25/25) |
| 算子数量 | ≥4 (实际 9 个) |
| Benchmark 覆盖 | 5 类 11 项 |
| 性能 vs PyTorch | 不低于 80% |
| 性能诊断闭环 | 完整 (Root Cause → Fix → Verify) |
| 参考文档 | 3 篇 (API 速查 + 设计模式 + 迁移指南) |
| 模板 | 3 个 (算子/测试/benchmark) |

---

## 8. 风险与边界

**不覆盖**:
- 动态 shape（ninetoothed 要求编译期确定）
- float64 dtype（未测试）
- 矩阵乘法、卷积（需要 scratch buffer / multi-stage tiling）
- 多 GPU / 分布式
- 自动 Triton → 九齿转换（采用文档驱动的手动迁移方式）

**依赖**:
- CUDA GPU + PyTorch ≥ 2.0 + Triton ≥ 3.0 + ninetoothed（从源码安装）
- 离线可复现，无联网依赖，无 API key
