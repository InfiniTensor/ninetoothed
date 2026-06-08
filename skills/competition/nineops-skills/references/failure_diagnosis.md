# 常见故障诊断指南

## 问题分类

### 1. Correctness 失败

**现象**: kernel 输出与 PyTorch baseline 不 match

**诊断步骤**:

1. **检查 dtype** — ntl 中是否做了类型提升？
   ```python
   # 好的做法：显式 cast 到 fp32 计算
   input_f32 = ntl.cast(input, ntl.float32)
   ```

2. **检查 tile 覆盖** — BLOCK_SIZE 能否整除总元素数？
   ```
   总元素数 % BLOCK_SIZE != 0 时 → 需要 mask 处理边界
   ```

3. **检查 shape 推导** — arrangement 中 shape 是否匹配？
   ```python
   # 打印 shape 调试
   print("input shape:", input.shape)
   print("output shape:", output.shape)
   ```

4. **检查 broadcast** — 广播维度是否正确 expand？
   ```python
   # 常见错误：直接 tile 而不是先 tile 再 expand
   ```

5. **检查 stride** — non-contiguous 的 ptr 偏移是否正确？

### 2. Crash / CUDA Error

**现象**: kernel 运行时报错（segfault, illegal memory access）

**诊断步骤**:

1. **检查 mask** — 边界 tile 是否有 mask？
2. **检查 other** — mask 命中区间的填充值是否合理？
   ```python
   other=0  vs  other=float("-inf")  # softmax 需要 -inf
   ```
3. **缩小规模** — 用 (1,) 或 (2, 2) 最小 shape 先验证不 crash
4. **检查 Tensor 声明** — `Tensor(1)` 是否正确反映维度数？

### 3. Performance 不达标

**现象**: kernel 比 PyTorch baseline 慢

**诊断步骤**:

1. **检查 BLOCK_SIZE** — 是否过小或过大？
   - 过小 → launch overhead 高
   - 过大 → 占用率高，warp 利用率低
2. **检查 load/store 模式** — 是否为合并访问？
3. **检查 autotune** — 是否使用 `block_size()` meta 符号？
4. **检查 generated source** — 是否有多余的循环或同步？

### 4. Compile 失败

**现象**: `make()` 或 `ninetoothed.make` 报错

**诊断步骤**:

1. **检查符号传递** — 所有 Symbol 是否都在 arrangement 和 application 中正确接收？
2. **检查 Tensor 数量** — tensors 元组长度是否匹配 arrangement/application 参数？
3. **检查 layout 展开** — 是否存在不支持的 tile 组合？
   ```python
   # 用 debug 模式查看展开过程
   import ninetoothed; ninetoothed.set_debug(True)
   ```

## 故障排查速查表

| 症状 | 最可能原因 | 检查点 |
|------|-----------|--------|
| 数值不匹配 | dtype/精度问题 | ntl.cast, atol/rtol |
| 部分元素错误 | 边界 tile 未处理 | mask, other |
| 全部为 0 | store 未执行 | output 赋值 |
| Crash | 越界访问 | mask, BLOCK_SIZE |
| 慢 | BLOCK_SIZE 不当 | autotune |
| Compile fail | 布局不可展开 | Tensor 声明 |

## 5. AST 跟踪错误（Nineteethed DSL 特有）

**现象**: kernel 编译成功但运行时 NameError / AttributeError

**诊断步骤**:

1. **检查 application 中的 Python 模块引用** — `math.*` 是否出现在 application() 函数内？
   ```python
   # 错误示例（在 application 内）
   def application(input, output):
       t = ntl.sqrt(2.0 / math.pi) * input  # ❌ NameError: name 'math' is not defined
   ```

2. **检查 application 中的模块级变量** — 变量名被 AST 跟踪原样嵌入？
   ```python
   _SQRT_2_OVER_PI = 0.7978845608028654
   def application(input, output):
       t = _SQRT_2_OVER_PI * input  # ❌ NameError: name '_SQRT_2_OVER_PI' is not defined
   ```

3. **检查 `**` 运算符** — Triton tensor 不支持 `__pow__`
   ```python
   # ❌ 错误
   t = input ** 3  # AttributeError: 'tensor' object has no attribute '__pow__'
   # ✅ 正确
   t = input * input * input
   ```

4. **检查 `ntl.tanh` 等不存在的 API** — 手动用 `ntl.exp` 组合
   ```python
   # ❌ 不存在
   output = ntl.tanh(t)
   # ✅ 手动实现
   exp_t = ntl.exp(t); exp_neg_t = ntl.exp(-t)
   output = (exp_t - exp_neg_t) / (exp_t + exp_neg_t)
   ```

**根因**: Nineteethed 的 AST 跟踪机制将 application() 内的 Python 代码转换为 Triton 代码，
但 Triton 编译环境中不存在标准 Python 库（math, torch, numpy）和 Python 模块级变量。

**解决方案**: `application()` 中只使用：`ntl.*` 函数、字面量数值、四则运算符。

## 6. 非连续张量 Correctness 失败

**现象**: kernel 对 contiguous 张量正确，但对 `.T` 或 `.t()` 转置张量数值错误

**根因**: arrangement 中使用了 `flatten()`，破坏了原始 strides，导致 Triton 按连续 strides
计算 ptr 偏移，在非连续张量上读写错位。

**诊断**: 检查 arrangement 是否调用了 `.flatten()`。

**修复**: 改用 preserve-ndim tile 模式：
```python
# ❌ 错误：flatten 破坏 strides
def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    return input.flatten().tile((BLOCK_SIZE,)), output.flatten().tile((BLOCK_SIZE,))

# ✅ 正确：保留 ndim，只 tile 最后一维
def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    tile_shape = (1,) * (input.ndim - 1) + (BLOCK_SIZE,)
    return input.tile(tile_shape), output.tile(tile_shape)
```

同时确保 `Tensor(ndim)` 的 ndim 与实际传入张量的维度一致。

## 日志收集命令

```bash
# 收集所有调试信息
python .skill/scripts/collect_task_log.py --output diagnose_log/
```
