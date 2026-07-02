# comb 算子开发报告

> torch 层迭代实现。kernel 方案因 Triton 循环内 `//` 的类型推断问题放弃。

## 1. 算子信息

| 项目 | 内容 |
|------|------|
| 算子名称 | `comb` |
| 分类 | torch 层迭代（无 kernel） |
| 实现方式 | int64 GPU tensor 上的 Python 循环：`res = res * (n-k+i) // i` |
| 基线 | `math.comb`（CPU） |
| 生成文件 | `torch/comb.py`（kernel 文件已删除） |

## 2. 为何没有 kernel

经过 4 次编译尝试，Triton 循环内 `//` 整数除法始终触发类型不一致错误：

| 尝试 | 方案 | 错误 |
|:--:|------|------|
| 1 | `ntl.cast(res * ... // i, ntl.int64)` | int64→int32 退化 |
| 2 | `ntl.cast(..., ntl.int32)` + 算术 masking | int32→<['32'],int32> 不一致 |
| 3 | `ntl.where(i <= m, term, res)` (同 gcd 模式) | 同上 |
| 4 | 显式 `ntl.cast(n - m + i, ntl.int32)` | 同上 |

**根因**：Triton 的 `//` 运算符在循环内与 `ntl.where`/算术条件组合时，类型推断不稳定。此问题不发生在 `gcd` 的 `%` 操作上（`%` 保持了类型一致性）。

## 3. 精度验证

| 测试 | 结果 |
|------|:--:|
| basic ([5,10,20], [2,5,10]) | PASSED |
| large (C(30,15) to C(60,30)) | PASSED |

## 4. 性能评估

| 规模 | k max | ntops (ms) | 说明 |
|------|:--:|-----------|------|
| 4096 | ~25 | ~2ms | 受 max_k 循环次数影响 |

**性能结论**：torch 层 Python 循环（每个 k 值迭代一次），效率取决于 `max(k, n-k)` 的最大值。torch.special.comb 在当前 PyTorch 版本不可用，无法对比。

## 5. 合计

- **迭代次数**：5（4 次 kernel 编译失败 + 1 次 int32→int64）
- **精度**：2/2 PASSED
- **kernel**：无（torch 层迭代）
