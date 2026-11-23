# 执行记录

## 环境

- GPU：NVIDIA GeForce RTX 4090 D
- Python：3.10.8
- PyTorch：2.1.2+cu121
- CUDA：12.1
- Triton：3.0.0
- pytest：8.4.2

## 首次测试

```bash
cd ntops
python3 -m pytest tests/test_maximum.py -q
```

结果：

```text
8 failed, 8 passed in 10.92s
```

同 shape 用例全部通过；0 维 `other` 全部编译失败。核心报错：

```text
invalid operands of type pointer<fp32> and triton.language.float32
```

生成代码将 0 维 tensor 作为指针参数直接传给 `triton.language.maximum`，而不是加载后的标量值。

## 修复与回归

Wrapper 对 `other.ndim == 0` 使用 `other.item()`，其余路径保持 tensor 输入。随后运行：

```bash
python3 -m pytest \
  tests/test_maximum.py tests/test_add.py tests/test_mul.py -q
```

结果：

```text
32 passed in 26.48s
```

其中 `test_maximum.py` 包含 16 组：float16/float32、1D 至 4D、同 shape 和 0 维 `other`。

## Benchmark

```bash
cd <ntskill-repository>
python3 \
  skills/competition/ntops-dev/scripts/run_operator_benchmark.py \
  maximum --shape 1048576 --dtype float16 --warmup 10 --iters 100

python3 \
  skills/competition/ntops-dev/scripts/run_operator_benchmark.py \
  maximum --shape 1048576 --dtype float16 --scalar-other \
  --warmup 10 --iters 100
```

| 输入 | ntops | PyTorch | ntops / PyTorch |
| --- | ---: | ---: | ---: |
| 同 shape | 0.0515 ms | 0.0083 ms | 6.1974x |
| 0 维 `other` | 0.0633 ms | 0.0093 ms | 6.7984x |

## 结论

- 新增算子、导出、correctness 和相邻回归已完成。
- 两条性能结果均未达到 Proposal 目标。
- 0 维路径虽然结果正确，但 `.item()` 带来主机同步，不适合性能敏感热路径。
- 任意不同 shape 广播尚未支持，不能将本结果表述为完整 PyTorch broadcasting。
