# meshgrid 算子开发报告

> 1D→2D broadcast 算子。自定义 arrangement 实现 tile + expand 广播模式。

## 1. 算子信息

| 项目 | 内容 |
|------|------|
| 算子名称 | `meshgrid` |
| 分类 | 自定义 arrangement（tile + expand 广播） |
| 实现方式 | 双 block_size tile，输入 tile→expand 广播到输出 |
| 基线 | `torch.meshgrid(indexing='xy')` |
| 生成文件 | `kernels/meshgrid.py`, `torch/meshgrid.py` |

## 2. 精度验证

| 测试 | 结果 |
|------|:--:|
| meshgrid([1,2,3], [4,5]) vs torch.meshgrid(xy) | PASSED |

## 3. 性能评估

| 规模 | ntops (ms) | PyTorch (ms) | 比率 | 判定 |
|------|-----------|-------------|------|:--:|
| 256×256 | 0.0947 | 0.0075 | 12.62x | **GAP** |
| 1024×1024 | 0.0931 | 0.0075 | 12.45x | **GAP** |
| 4096×4096 | 0.4393 | 0.0074 | 59.67x | **GAP** |

**性能分析**：view vs copy 的先天劣势。PyTorch 的 meshgrid 是零拷贝 view（expand + stride manipulation），NineToothed kernel 做了 O(N²) 的数据拷贝。此类广播算子 kernel 方案无法匹敌 PyTorch view。

**六项策略**：内存访问 ✅（coalesced）/ 其他 N/A

## 4. 边界情况

- ✅ 不等长输入（nx ≠ ny）
- ✅ 非整除 block_size
- ✅ 'xy' 索引（与 CPU 参考一致）

## 5. 合计

- **迭代次数**：0（已有实现）
- **精度**：PASSED
- **性能**：❌ 12-60x（view vs copy 先天劣势）
