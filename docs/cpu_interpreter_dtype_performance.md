# dtype 解析缓存与多形状回归

基于 `64bd6958a610a0e5e61807aa141bf1cbc7881426`，将逐操作重复的 dtype 解析改为最多 64 项的描述符缓存。只保存规范化字符串和不可变 dtype，保留别名、字节序、None/fallback、原始错误拼写与动态用户对象的语义；不缓存张量数值。

## 三轮确认

使用实际前端和默认管线生成 SSA，准备过程不计时。三轮独立旧/新进程交替，每配置七个执行样本；独立 NumPy 输出、完整轨迹指纹和输入不变性全部匹配。

| 工作负载 | 不记录轨迹的三轮加速比 | 记录轨迹的三轮加速比 |
|---|---|---|
| 连续 FP32，4096 元素循环 | 1.385～1.447× | 1.056～1.095× |
| 跨步 FP32，16387 元素循环 | 1.352～1.429× | 1.057～1.091× |
| 反向 INT32，8193 元素循环 | 1.252～1.315× | 1.048～1.063× |
| softmax 17×257 尾部 | 1.068～1.114× | 0.992～1.079× |
| matmul 9×16×7 尾部 | 1.059～1.068× | 1.019～1.031× |

三个无轨迹循环是预定目标，其三轮几何平均分别为 1.386、1.329、1.383×。各目标至少快 5%、每轮目标平均至少快 10%、其余对照不得慢 5% 的门槛全部通过。较小或开启轨迹的程序收益有限，不将目标结果推广到所有程序，也不与上一轮不同形状的比值相乘。

另测预热后与清缓存后的 tracemalloc 分配峰值，记录结果释放和缓存释放后的分配。冷测量包括新建表达式/dtype 缓存；这些数值不是进程 RSS。

## 验收与复现

- 无 Torch/Triton 的 CPU 回归：535 passed、15 GPU deselected，36.18 秒。
- 新增 23 项 dtype 语义回归，包含在 535 内。
- RTX 4090 D 上 15/15 真实 GPU 差分通过；23 项新测试在该主机 CPU 上通过，0.42 秒。
- 本地 Ruff、format、贡献风格检查通过；最新提交的跨版本和安装回放以 CI 记录为准。

[固定原始证据](https://github.com/a962695448-rgb/ninetoothed/tree/98feddcdf50a9d3991707b96d471220c92b6b132/docs/validation/dtype-cache-20260917)保存协议、六个独立进程记录、冷/热内存、源码、JUnit 与准备脚本。绝对时间对应 macOS arm64 / Python 3.12.14 / NumPy 2.3.5 / SymPy 1.14.0；不同环境须重新配对。

仓库 CPU 入口仍为 `python scripts/run_cpu_tests.py`。GPU 验证使用 `python scripts/verify_interpreter_gpu.py --report /tmp/new-gpu-report.json`；新增测试可用 `PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_interpreter_dtype_cache.py`。

此次提升的是 CPU 解释器执行开销，GPU 生成器和核函数未改。旧完整 GPU 集合和 A100 记录保留各自受测源码范围，不能当作新版本全库测试计数。
