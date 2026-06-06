# Nineteethed DSL 算子开发 — Agent 快速参考

> **工作流指引**：收到算子开发任务时，先通读 `skill/SKILL.md §0 工作流总览`，按「开发→测试→诊断→修复」四阶段执行。
> 关键模板在 `skill/templates/`，故障排查参照 `skill/references/failure_diagnosis.md`。

## 核心经验总结（来自 Add / ReLU / GELU 实现）

### AST 跟踪陷阱（最重要的坑）
application() 内的代码会被 AST 跟踪原样嵌入生成 Triton 代码，Triton 环境没有标准 Python 库。
- **禁止** `math.*`、`torch.*`、`numpy.*` → 用字面量数值
- **禁止**模块级变量引用（变量名被原样嵌入导致 NameError）
- **禁止** `**` 运算符（Triton tensor 无 `__pow__`） → `x * x * x`
- **禁止** `ntl.tanh`（不存在） → `(exp(t)-exp(-t))/(exp(t)+exp(-t))`
- **允许** `ntl.*` 函数、字面量数值、四则运算

### 非连续张量支持（关键修复）
- **不要** `flatten()` → 破坏 strides，转置张量写入错位
- **要** `tile(tuple(1 for _ in range(ndim-1)) + (block_size,))` → 保留 strides
- `Tensor(ndim)` 的 ndim 必须与实际张量维度一致

### Element-wise 通用 arrangement 模式
```python
def _element_wise_arrangement(*tensors, block_size):
    ndim = max(tensor.ndim for tensor in tensors)
    assert all(tensor.ndim == ndim or tensor.ndim == 0 for tensor in tensors)
    tile_shape = tuple(1 for _ in range(ndim - 1)) + (block_size,)
    return tuple(
        tensor.tile(tile_shape) if tensor.ndim != 0 else tensor
        for tensor in tensors
    )
```

### GELU 实现要点
- **近似版**: `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`
  - `sqrt(2/pi)` = `0.7978845608028654`（字面量）
  - `x^3` = `x * x * x`
  - `tanh` = 手动用 `ntl.exp`
  - 测试: `torch.nn.functional.gelu(x, approximate='tanh')`
- **精确版**: `x * 0.5 * (1 + erf(x / sqrt(2)))`
  - 使用 `ntl.erf`, `ntl.sqrt`
  - 测试: `torch.nn.functional.gelu(x)`

### 数据类型支持
- fp32: atol=1e-5, rtol=1e-5
- fp16: atol=1e-3, rtol=1e-3 (注意精度损失)
- bf16: 类似 fp16

### 广播操作
- 通过 `expand_as` 创建 stride=0 视图
- Triton 自动处理 HBM 广播
