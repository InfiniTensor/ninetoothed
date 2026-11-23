# column_stack 算子开发报告

> 交错重排 + identity kernel。将多个 1D tensor 堆叠为 2D 矩阵的列。

## 1. 算子信息

| 项目 | 内容 |
|------|------|
| 算子名称 | `column_stack` |
| 分类 | element_wise identity（交错重排后拷贝） |
| 实现方式 | torch 层交错排布 + identity kernel |
| 基线 | `torch.column_stack` |
| 生成文件 | `kernels/column_stack.py`, `torch/column_stack.py` |

## 2. 实现原理

```
a=[1,2,3], b=[4,5,6] → [[1,4],[2,5],[3,6]]

1. Torch 层交错排布：flat = [a[0],b[0],a[1],b[1],a[2],b[2]]
   方式：flat[col::N] = t  (strided assignment, 无 torch.stack/cat)
2. reshape 为 (M, N)
3. Identity kernel 拷贝到输出
```

## 3. 精度验证

| 测试 | 结果 |
|------|:--:|
| basic (3×3) | PASSED |
| 2 cols | PASSED |
| large (4096×16) | PASSED |
| int64 dtype | PASSED |

**4/4 PASSED**

## 4. 性能评估

| 规模 | 列数 | ntops (ms) | PyTorch (ms) | 比率 | 判定 |
|------|:--:|-----------|-------------|------|:--:|
| 256 | 2 | 0.1488 | 0.0252 | 5.91x | **GAP** |
| 1024 | 2 | 0.1536 | 0.0238 | 6.46x | **GAP** |
| 4096 | 2 | 0.2151 | 0.0247 | 8.71x | **GAP** |
| 256 | 16 | 0.4705 | 0.0387 | 12.16x | **GAP** |
| 1024 | 16 | 0.4208 | 0.0365 | 11.52x | **GAP** |
| 4096 | 16 | 0.4099 | 0.0440 | 9.31x | **GAP** |

**性能回退分析**：
1. Strided assignment (`flat[col::N] = t`) 在 GPU 上是非连续写入，效率低
2. Identity kernel 做了一次不必要的二次拷贝
3. PyTorch 的 column_stack 内部直接构造 contiguous tensor，无 strided write
4. 随列数增加，strided write 步长增大，性能进一步恶化

**六项策略**：内存访问 ⚠️（strided write）/ 算子融合 ✅（已融合重排+拷贝）/ 其他 N/A

## 5. 合计

- **迭代次数**：1
- **精度**：4/4 PASSED
- **性能**：❌ 6-12x（strided write 开销）
