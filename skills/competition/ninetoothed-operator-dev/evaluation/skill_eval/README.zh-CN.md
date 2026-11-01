# skill_eval —— KernelSwift 风格的评估工具

[English](README.md) | **中文**

方法来源:上海人工智能实验室 KernelSwift Agent Infra 的公开技术分享。独立实现,
未复制外部源码。

## 借鉴点

| 概念 | 本模块 | 实现 |
|---|---|---|
| 固定计算图 | `robust_benchmark()` warmup 阶段 | 计时前先调用 `warmup=25` 次 |
| 多次测量 | `robust_benchmark(iters=100)` | 每次迭代用 CUDA Event |
| 离群值剔除 | `_remove_outliers_iqr()` | Tukey IQR 栅栏,k=1.5 |
| reward-hacking 防护(静态) | `static_analysis(source_path)` | AST:统计 tl.load/tl.store,检测常量写出 |
| reward-hacking 防护(动态) | `dynamic_analysis(kernel_fn, output)` | 全零 / NaN / Inf / no-op 检查 |
| reward-hacking 防护(ncu roofline) | `ncu_roofline_check()` | stub(需安装 ncu) |
| 测量缓存 | (TODO:加 SQLite 或 JSON 缓存) | 未实现 |
| Island-based evolution | (v0 范围外) | Stage 3 实验性 |

## 使用

把 `bench_compare.benchmark(...)` 换成 `robust_benchmark(...)`,一次调用即得离群值
剔除与 reward-hacking 检测:

```python
from skill_eval import robust_benchmark, full_guard, BenchResult

# 1. 先跑一次 kernel 得到输出张量。
out = torch.empty_like(x)
kernel(x, out, BLOCK_SIZE=512)

# 2. 跑稳健 benchmark。
result: BenchResult = robust_benchmark(
    kernel_fn=lambda: kernel(x, out, BLOCK_SIZE=512),
    bytes_moved=x.numel() * x.element_size() * 2,
    # gpu 自动探测实际设备；传 gpu= 或设 $NT_GPU 可手动指定。
)
print(result)

# 3. 跑完整防护(静态 + 动态)。
report = full_guard(
    kernel_fn=lambda: kernel(x, out, BLOCK_SIZE=512),
    output_tensor=out,
    input_tensor=x,
)
print(report)  # "reward_hacking: CLEAN" 或 "SUSPICIOUS: ..."
```

## 为什么需要离群值剔除

GPU 计时噪声来源:
- 热降频(前若干次迭代可能是冷态)
- 操作系统抖动(随机 L3 逐出、调度抢占)
- CUDA 驱动首次同步开销

IQR 栅栏 k=1.5 在稳定卡上约剔除 7% 样本,但能挡住偶发的 10x 尖峰(否则均值虚高
15-20%)。在嘈杂的服务器上提升更大。

## 与 bench_compare.py 的区别

| 指标 | bench_compare.py | robust_benchmark() |
|---|---|---|
| 离群值剔除 | 无 | IQR 栅栏 |
| reward-hacking | 无 | 带宽 + AST + 动态 |
| 固定计算图 | 隐式 | 显式 warmup 阶段 |
| 输出 | dict | BenchResult dataclass |
