# 布局与表达式修订后的 RTX4090 核对

**15 项实际 GPU 差分和 108 项布局/表达式语义回归通过。**精确受测版本为 `07062137e599cfead66d510cede3c0a3c1bbdf6b`。本轮补充验证记录，计算源码与测试内容保持该版本。

## 范围

15 项 GPU 差分覆盖已记录的 float32/int32 尾块、广播、行归约、布尔比较、分支/循环、softmax、整数 floor division/remainder，以及一个 float32 多输出 tile 的矩阵乘案例。浮点采用 `rtol=atol=1e-3`，整数和布尔精确比较；输入、seed 和各项结果保存在原始 JSON。

近期新增的 67 项布局调用与 41 项表达式语义测试在同一服务器环境通过：

```text
108 passed in 0.77s
```

这 108 项执行解释器语义，不能称为 108 项 GPU 内核测试。此前无 Torch/Triton 环境中的完整 CPU 选择范围仍为 770 passed / 15 GPU deselected；本次没有重算为更多独立用例。

## 环境与源码

| 项目 | 实测值 |
|---|---|
| GPU | NVIDIA GeForce RTX4090，计算能力 8.9 |
| Python | 3.12.3 |
| NumPy / SymPy | 1.26.4 / 1.13.1 |
| Torch | 2.6.0a0+ecf3bae40a.nv25.01 |
| Triton / Torch CUDA | 3.1.0 / 12.8 |

当前提交的 **196 份文件**同时通过 Git blob 与远端冻结输入 SHA256 核对，其中包括全部 **70 份 src 文件**。运行前后源码一致，结果回收到本地逐文件校验后关闭实例。

[固定原始证据](https://github.com/a962695448-rgb/ninetoothed/tree/6e3df8a95a3aec9b32d61de343525100b11e46b8/docs/validation/layout-expression-gpu-20260917)包含逐项结果、JUnit、环境、源码清单和检出核对脚本。该证据来自已完成的手动实机运行；仓库自托管 GPU 自动任务排队不能记作通过。

## 复现与限制

在相应 CUDA 环境中，从固定提交的仓库根目录运行：

```bash
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python scripts/verify_interpreter_gpu.py --report /tmp/new-nine-gpu.json
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_interpreter_layout_calls.py tests/test_interpreter_expression_semantics.py -W error
```

这不是 GPU 性能基准，也不覆盖完整仓库、任意 layout/dtype、split-K 或 Tensor Core。旧 A100 和其他历史结果继续保持其原源码与环境范围。
