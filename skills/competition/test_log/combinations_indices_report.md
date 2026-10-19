# combinations_indices 算子开发报告

> torch 层委托 `torch.combinations`。输出大小运行时决定 + 数据依赖循环 → 无法 kernel 化。

## 1. 算子信息

| 项目 | 内容 |
|------|------|
| 算子名称 | `combinations_indices` |
| 分类 | torch 层委托（无 kernel） |
| 实现方式 | `torch.combinations(torch.arange(n), r=k)` |
| 基线 | `torch.combinations` |
| 生成文件 | `torch/combinations.py` |

## 2. 为何没有 kernel

1. **输出大小运行时决定**：C(n,k) 无法在 premake 中表达
2. **顺序生成算法**：每个组合依赖前一组合，GPU 无法并行生成
3. **数据依赖 while 循环**：外层 while(1) 无法转为固定 range

## 3. 精度验证

| 测试 | 结果 |
|------|:--:|
| C(4,2) | PASSED |
| C(5,3) | PASSED |

## 4. 性能评估

| 规模 | ntops (ms) | PyTorch (ms) | 比率 | 判定 |
|------|-----------|-------------|------|:--:|
| C(30,5) | 6.1262 | 5.7477 | 1.07x | **OK** |

## 5. 合计

- **迭代次数**：1
- **精度**：2/2 PASSED
- **性能**：1.07x ✅
