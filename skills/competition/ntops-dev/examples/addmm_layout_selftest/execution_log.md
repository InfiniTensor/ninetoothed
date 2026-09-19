# AddMM 执行记录

环境：NVIDIA GeForce RTX 4090 D，Python 3.10.8，PyTorch 2.1.2+cu121，CUDA 12.1，Triton 3.0.0。

## Correctness

```bash
cd ntops
python3 -m pytest tests/test_addmm.py -q
```

```text
..                                                                       [100%]
2 passed in 10.46s
```

## Benchmark

```bash
cd <ntskill-repository>
python3 \
  skills/competition/ntops-dev/scripts/run_operator_benchmark.py \
  addmm --shape 512,512,512 --dtype float16 --warmup 10 --iters 50
```

```text
ntops_ms=0.1276
torch_ms=0.0272
relative_to_torch=4.6976
```

## Generated source

关键统计：

```text
language_load=3
language_store=1
language_dot=1
language_arange=40
autotune=True
```

后续检查目标是 tile/block、precision mode 和 autotune 对 `(512, 512, 512)` 的选择。当前记录没有声称已经修复性能。

## 非连续输入

```text
contiguous ok=True
input_stride=(48, 1) mat1_stride=(80, 1) mat2_stride=(48, 1)
max_abs=0.03125

all_non_contiguous_transpose ok=True
input_stride=(1, 64) mat1_stride=(1, 64) mat2_stride=(1, 80)
max_abs=0.0
```

已覆盖 transpose 产生的非连续 `input`、`mat1`、`mat2`。尚未覆盖任意 `as_strided`、negative stride 和广播矩阵输入。性能未达到 Proposal 目标。
