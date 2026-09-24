# 表达式语义回归与未采用的求值实验

本轮保留生产求值器，新增 **41 项表达式语义测试**。全部 70 份 `src/` 文件逐字节等于基线 `4d50b0424d963ac7ea3fd67fb06ff2c828fb3d92`；这次提交不声称新的执行性能提升。

## 保留的验证

新测试覆盖常量与动态符号组合、Python 精确结果类型、signed zero、固定种子的随机表达式树、异常类型/消息与先后顺序、NumPy 警告策略、数组更新、自定义算子/整数子类、超大整数和对象释放。随机树直接与原递归求值器比较；不预计算或缓存动态输入。

在没有 Torch/Triton 的 macOS arm64 / Python 3.12.14 / NumPy 2.3.5 / SymPy 1.14.0 环境，最终生产组合通过：

```text
770 passed, 15 deselected in 35.40s
```

这是文档约定的 CPU 选择范围。15 项 GPU 测试被取消选择，本轮未启动 GPU。历史硬件结果保持原有源码范围。

## 两项未采用的实验

两项都与同一个固定基线分别比较，三轮分别启动旧、新独立进程并交替顺序。目标是三种 prepared `interpret_program` matmul；控制包含向量、padding、softmax、小 matmul 及 trace 开关。每项 7 组、每组 3 次，要求每轮目标几何平均加速比至少 **1.10×**，任何稳态测试项耗时增加不超过 **5%**。

| 候选 | 第一轮 | 第二轮 | 第三轮 | 决定 |
|---|---:|---:|---:|---|
| 有界 Python 整数/布尔常量子树折叠 | 1.014061× | 1.002081× | 1.004981× | 未采用 |
| 计划直接持有常量叶节点、算子仍在运行时执行 | 1.039771× | 1.090193× | 1.039057× | 未采用 |

直接持值方案另有一个向量对照耗时增加约 **5.67%**。两项各 33 条配对及完整前端轨迹摘要核对均保留；冷启动和 tracemalloc 原始数据另列，不替代稳态门槛。通过正确性不能证明性能收益足够。

最初随机测试对 NumPy 不支持的布尔减法未处理参考异常，出现 8 failed / 43 passed。后续改为核对两条求值路径相同的异常类型和信息；未删除输入，初稿源码、日志和 JUnit 均归档。

## 复现与原始记录

[固定证据目录](https://github.com/a962695448-rgb/ninetoothed/tree/73a0fed3deeb9c7f7858f1d37e33bd571064e760/docs/validation/literal-plans-20260917)包含两个候选源码、冻结协议、六个进程的各自原始计时、所有测试日志、SHA256 清单和准备脚本。生产变更的 `production/prepare_inputs.py` 校验完整基线，再仅加入新增测试；候选复现脚本位于各自实验目录。

在仓库中复现已保留的 CPU 测试：

```bash
python scripts/run_cpu_tests.py --junitxml /tmp/nine-cpu-results.xml
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_interpreter_expression_semantics.py -W error
```
