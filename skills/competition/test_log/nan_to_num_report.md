# nan_to_num 算子开发报告

> element_wise + libdevice.isnan/isinf + ntl.where 条件替换。

## 1. 算子信息

| 项目 | 内容 |
|------|------|
| 算子名称 | `nan_to_num` |
| 分类 | element_wise + libdevice 条件检测 |
| 实现方式 | `libdevice.isnan` / `libdevice.isinf` + `ntl.where` 链式替换 |
| 基线 | `torch.nan_to_num` |
| 生成文件 | `kernels/nan_to_num.py`, `torch/nan_to_num.py` |

## 2. 精度验证

| 测试 | 结果 |
|------|:--:|
| nan_to_num([1,nan,inf,-inf,0]) vs torch | PASSED |

## 3. 性能评估

| 规模 | ntops (ms) | PyTorch (ms) | 比率 | 判定 |
|------|-----------|-------------|------|:--:|
| 256×256 | 0.1034 | 0.0169 | 6.12x | SLOW |
| 1024×1024 | 0.0697 | 0.0300 | 2.33x | SLOW |
| 4096×4096 | 0.4755 | 0.4485 | 1.06x | **OK** |

**六项策略**：内存访问 ✅ / 其他 N/A

## 4. 设计要点

- `libdevice.isnan` / `libdevice.isinf` 直接可用
- 正负无穷区分：`ntl.where(input > 0, 1, 0)` 生成符号 mask，与 isinf 做 `&` 组合
- 三个 0-dim runtime Tensor 传参（nan_val, posinf_val, neginf_val）

## 5. 合计

- **迭代次数**：2（默认值 + posinf/neginf 分离）
- **精度**：PASSED
- **性能**：1.06x @ 4096² ✅
