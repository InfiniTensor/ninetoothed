# 九齿算子开发 Skill — 自测套件

本目录包含 5 个自测任务，验证 `ninetoothed-operator-dev` skill 在指导 AI 智能体完成九齿算子开发时的有效性。

## 自测任务覆盖

| 编号 | 算子 | 类型 | 赛题要求 | Benchmark |
|------|------|------|---------|-----------|
| 1 | GELU | 逐元素/激活函数 | 逐元素或广播类 | 有 |
| 2 | Softmax | 归约/分块 | 归约或分块类 | 有 |
| 3 | Strided Add | 非连续/步长 | 带非连续输入、步长或偏移量 | 有 |
| 4 | Stride vs Contiguous 分析 | 性能诊断 | 性能优化或回退分析 | 有 |
| 5 | Triton Add 迁移 | 迁移/诊断 | 补充完整性 | 无 |

## 运行方式

全部 correctness 测试:
```bash
python run_all_operator_tests.py
```

全部 benchmark:
```bash
python benchmarks/run_benchmarks.py --quick
```

仅性能分析:
```bash
python benchmarks/run_benchmarks.py --category analysis --quick
```

## 环境要求

- CUDA GPU（已在 RTX 4060 上验证）
- PyTorch >= 2.0
- Triton >= 3.0
- ninetoothed（从源码安装）

## 各任务详情

见对应 `self_eval_*.md` 文件。
