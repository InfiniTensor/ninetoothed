# 解释器表达式求值优化：测量记录

基线提交：`d7e779642eab70164be849b2dcf52a5d02a7858a`。本轮仅修改解释器内部表达式求值，新增十项回归；SSA、编译器与 GPU 生成代码保持原样。

## 方法

先用 cProfile 与 tracemalloc 找热点，再进行三轮独立旧/新进程比较，运行顺序交替。每个用例含预热和五个无剖析器计时样本，单独测量内存；前端构建时间另列，正确性及轨迹序列化时间不计入执行计时。固定的三项程序均由实际前端与默认 Triton 目标管线生成 SSA。

预设门槛：矩阵乘、softmax 在每轮中位数上至少快 10%，对照程序不得慢 5%，开启轨迹的峰值内存不得增加 5%。所有输出符合独立 NumPy 公式，完整轨迹的序列化指纹在旧/新之间一致。结果：三轮全部满足门槛。

| 程序 | 开启轨迹 | 三轮中位数加速比范围 |
|---|---|---|
| elementwise_loop_16384 | False | 1.066–1.113× |
| elementwise_loop_16384 | True | 1.012–1.081× |
| softmax_32x128 | False | 1.513–1.550× |
| softmax_32x128 | True | 1.427–1.448× |
| matmul_12x16x12 | False | 1.624–1.661× |
| matmul_12x16x12 | True | 1.457–1.487× |

上述是 CPU 解释器执行耗时，不能解释为 Triton GPU 内核或模型端到端加速。预热后的轨迹分配峰值整体持平，未压缩或丢弃调试记录。tracemalloc 记录的是测量区间的新分配，不是进程 RSS；预热时已建立的计划缓存不计入这个峰值，因此该结果不表示完整进程内存或冷缓存开销减少。三个固定程序的结果不代表所有 SSA 程序；首次构建、极小程序与未支持语义保持各自边界。

## 实现与语义

对不可变 IndexExpr 的算术结构建立可复用求值计划，符号值仍在每次调用时读取。缓存只保留最近 256 个根表达式计划；简单常量和符号直接求值，避免挤占缓存。对象身份作为键，区分结构相等的 True、1、1.0 以及正负零；不可哈希常量继续原求值路径，不全局保留数组。未优化的表达式仍按原受支持操作求值，不执行 Python eval。

回归覆盖：常量类型与负零、符号与数组原地变化、输入释放、不可哈希常量、错误先后顺序、旧表达式缓存释放；原完整 CPU 选择范围通过 512 项，15 项实机测试按规则排除。

## 验收与复现

无 Torch/Triton 的 macOS arm64 / Python 3.12.14 / NumPy 2.3.5 / SymPy 1.14.0 环境完整 CPU 选择范围：**512 passed、15 GPU deselected，35.10 秒**。新增十项包含在 512 内。

RTX 4090 D / NGC Torch 2.6 / CUDA 12.8 上，最终源码 **15/15 真实 Triton GPU 差分通过**，十项表达式测试 **10 passed in 1.55s**。Ruff、format、贡献风格检查通过。首次实验驱动因漏设 PYTHONPATH 而收集失败；补上路径后通过，初始失败与修正记录均保留。

[固定原始证据](https://github.com/a962695448-rgb/ninetoothed/tree/bb284c1a9f7ce7cfd33683571936de9523822320/docs/validation/expression-plans-20260916)包含三轮六个独立进程的记录、源码与轨迹指纹、完整 CPU JUnit、实机结果和复现脚本；被弃用的原型亦保留在实验档案。

```bash
# 不含 Torch/Triton 的 CPU 环境
python scripts/run_cpu_tests.py
# GPU 环境
python scripts/verify_interpreter_gpu.py --report /tmp/new-gpu-report.json
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_interpreter_expression_plans.py
```

此次改变的是 CPU 解释器执行，不能沿用旧源码的完整 GPU 集合计数作为新版本的全库通过数。没有重新运行未变更的完整卷积集合；旧 A100 与 902 项集合继续保留各自版本范围。当前提交的安装回放与跨 Python 版本情况以 CI 记录为准。
