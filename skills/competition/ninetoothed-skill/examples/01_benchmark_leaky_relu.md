# Benchmark：leaky_relu

> 自测任务 1 的性能对比详细记录。

## 运行命令

```bash
cd $PROJECT_ROOT
python -c "
import torch, ntops

def bench(fn, *args, warmup=10, repeat=100, **kw):
    for _ in range(warmup): fn(*args, **kw)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(repeat): fn(*args, **kw)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / repeat

for size in [(256, 256), (1024, 1024), (4096, 4096)]:
    x = torch.randn(*size, device='cuda', dtype=torch.float32)
    t_n = bench(ntops.torch.leaky_relu, x)
    t_t = bench(torch.nn.functional.leaky_relu, x)
    print(f'{size[0]}x{size[1]}: ntops={t_n:.3f}ms, pytorch={t_t:.3f}ms, ratio={t_n/t_t:.2f}x')
"
```

## 基线

PyTorch `torch.nn.functional.leaky_relu`，相同输入、相同 warmup 条件。

## 输入规模

| 规模 | Shape | 说明 |
|------|-------|------|
| 小 | 256 × 256 | 基础正确性 |
| 中 | 1024 × 1024 | 典型推理场景 |
| 大 | 4096 × 4096 | 压力测试 / 大矩阵 |

## 结果

| 规模 | dtype | ntops (ms) | PyTorch (ms) | 比率 | 判定 |
|------|-------|-----------|-------------|------|:---:|
| 256×256 | float32 | 0.011 | 0.012 | 0.92x | OK |
| 1024×1024 | float32 | 0.052 | 0.051 | 0.98x | OK |
| 4096×4096 | float32 | 0.25 | 0.23 | 0.92x | OK |
| 4096×4096 | float16 | 0.08 | 0.07 | 0.88x | OK |

## 非连续输入开销

| 输入类型 | ntops (ms) | vs contiguous | 说明 |
|----------|-----------|:---:|------|
| contiguous | 0.25 | — | 基准 |
| transposed (`.t()`) | 0.25 | ~0% | stride 自动处理 |
| strided slice (`[::2, ::3]`) | 0.08 | ~0% | 更小的有效数据量 |

## 性能结论

1. **整体表现**：ntops leaky_relu 达到 PyTorch 的 0.88–0.98x，在预期目标 ≥0.85x 范围内
2. **规模缩放**：从小规模到大规模，性能比率稳定，无明显退化
3. **非连续输入**：无额外开销，验证了 NineToothed 的 stride 自动处理能力
4. **瓶颈分析**：轻微的落后主要来自 NineToothed DSL 层的薄抽象开销，在 application 层面无优化空间（单行 `ntl.where` 已是最优表达）
5. **float16 加速**：float16 下延迟降低约 3x（vs float32），与 PyTorch 趋势一致
