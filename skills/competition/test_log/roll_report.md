# roll 算子开发报告

> 两段 identity kernel。环形移位拆为连续 segment 拷贝，无需 gather/scatter。

## 1. 算子信息

| 项目 | 内容 |
|------|------|
| 算子名称 | `roll` |
| 分类 | 分段 identity kernel（每维度 2 次 launch） |
| 基线 | `torch.roll` |
| 生成文件 | `kernels/roll.py`, `torch/roll.py` |

## 2. 实现原理

```
roll([1,2,3,4,5], shift=2) → [4,5,1,2,3]

Kernel 1: input[-2:]  → output[:2]    ([4,5] wrap to front)
Kernel 2: input[:-2]  → output[2:]    ([1,2,3] shift right)
```

## 3. 精度验证

| 测试 | 结果 |
|------|:--:|
| roll 1D shift=2 | PASSED |
| roll 1D shift=-1 | PASSED |
| roll 2D dim=0 | PASSED |
| roll 2D dim=1 | PASSED |
| roll 3D dim=1 | PASSED |
| roll shift=0 | PASSED |

**6/6 PASSED**

## 4. 性能评估

| 规模 | ntops (ms) | PyTorch (ms) | 比率 | 判定 |
|------|-----------|-------------|------|:--:|
| 256×256 | 0.1405 | 0.0175 | 8.04x | launch overhead |
| 1024×1024 | 0.1364 | 0.0317 | 4.30x | overhead 摊薄中 |
| 4096×4096 | 0.4447 | 0.4544 | 0.98x | **OK** |

**六项策略**：内存访问 ✅（contiguous segment）/ 算子融合 N/A / 循环展开 N/A / 同步 N/A / 精度 N/A / 计算重组 ✅（两段式已最优）

**性能结论**：大规模下与 PyTorch 持平（0.98x）。两段 contiguous 拷贝 + identity kernel 是正确的性能策略。

## 5. 合计

- **迭代次数**：1
- **精度**：6/6 PASSED
- **性能**：0.98x @ 4096² ✅
