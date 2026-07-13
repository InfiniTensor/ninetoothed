# 常见算子 DSL 实现模式

## 1. Element-wise 1D（广播 + 全 tile）

```
input.tile((BLOCK_SIZE,)) → output.tile((BLOCK_SIZE,))
```

- 适用于：add, mul, sub, div, silu, gelu, relu
- 所有输入输出沿最后一维均匀分块
- BLOCK_SIZE 通常为 1024 或 2048

### 非连续支持（重要）

**问题**：用 `flatten().tile((BLOCK_SIZE,))` 会破坏非连续张量的 strides，
导致 `torch.empty(N, M).t()` 等转置张量的输出数据错乱。

**方案**：对多维度张量，**不 flatten**，保留原始维度数，只 tile 最后一维：

```python
def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    tile_shape = (1,) * (input.ndim - 1) + (BLOCK_SIZE,)
    return input.tile(tile_shape), output.tile(tile_shape)
```

同时将 `Tensor(1)` 改为 `Tensor(ndim)` 以匹配实际维度。

### AST 跟踪约束

`application()` 中的 Python 代码会被 AST 跟踪并嵌入 Triton 代码，有以下限制：
- **禁止** `math.*`、`torch.*` 等 Python 模块调用
- **禁止**模块级变量引用（变量名被原样嵌入导致 NameError）
- **禁止** `**` 运算符（Triton tensor 无 `__pow__`）
- **改用字面量**：`0.7978845608028654` 而非 `math.sqrt(2.0 / math.pi)`
- `x ** 3` → `x * x * x`

### GELU 激活函数

```python
# 近似版（tanh 公式，ntl.exp 手动实现 tanh）
def application(input, output):
    t = 0.7978845608028654 * (input + 0.044715 * input * input * input)
    exp_t = ntl.exp(t); exp_neg_t = ntl.exp(-t)
    output = 0.5 * input * (1.0 + (exp_t - exp_neg_t) / (exp_t + exp_neg_t))  # noqa: F841

# 精确版（erf 公式）
def application(input, output):
    output = input * 0.5 * (1.0 + ntl.erf(input / ntl.sqrt(2.0)))  # noqa: F841
```

## 2. 行归约 2D

```
input.tile((1, BLOCK_SIZE)) → output.tile((1, BLOCK_SIZE))
```

- 适用于：softmax, rms_norm, layer_norm
- 保留第一维（batch/rows），沿第二维做归约
- BLOCK_SIZE 取 `input.shape[-1]` 以覆盖整行

## 3. Matmul 2D

```
input:   tile((BM, BK)) → tile((1, -1)) → expand(-1, N_blocks) → squeeze(0)
other:   tile((BK, BN)) → tile((-1, 1)) → expand(M_blocks, -1) → squeeze(1)
output:  tile((BM, BN))
```

- 三个 block_size 符号（BM, BN, BK）由 autotune 搜索
- application 中用 `ntl.dot` + 循环累加

## 4. BMM（批矩阵乘）

```
input:  tile((1, BM, BK)) → tile((1, 1, -1)) → expand(-1, -1, N_blocks) → squeeze(1)
other:  tile((1, BK, BN)) → tile((1, -1, 1)) → expand(-1, M_blocks, -1) → squeeze(2)
output: tile((1, BM, BN))
```

- 与 Matmul 2D 的区别：多一个 batch 维度
- 多分支前缀 `(1, ...)`

## 5. RoPE（stride-dilation 模式）

```
input:     tile((1, 1, BLOCK_SIZE)) → tile((1, -1, -1))
  → expand(-1, 2*cos.shape[0], -1) → squeeze(1) → squeeze(0)  # pass
  → tile((-1, -1)) → tile((1, DILATION))  # query

cos/sin:   tile((1, BLOCK_SIZE)) → tile((-1, -1)) → expand(...)
```

- 使用 `tile((1, DILATION))` 处理 stride 跳跃访问
- cos/sin 用 `expand` 广播到 query 的维度
- 多分支 kernel（多个 kernel 对象由同一个 make 产生）

## 6. MaxPool2D（滑动窗口）

```
input: tile((1, BLOCK_SIZE, 1, 1)) → tile((1, -1, BLOCK_SIZE_H, BLOCK_SIZE_W))
       → tile((1, -1, -1, -1)) → expand(-1, -1, -1, input.shape[3])
```

- 固定 batch=1，在 H 和 W 维度滑动
- 外层 tile 为 1（保留），内层按窗口大小平铺

## 7. Attention（Flash Attention online softmax）

```
query: tile((1, BLOCK_SIZE_M, BLOCK_SIZE_K))
key:   tile((1, BLOCK_SIZE_K, BLOCK_SIZE_N))
value: tile((1, BLOCK_SIZE_N, BLOCK_SIZE_K))
output:tile((1, BLOCK_SIZE_M, BLOCK_SIZE_K))
```

- application 中使用 ntl.dot 计算 score
- online softmax 维护 m_i, d_i 状态变量
- loop over key/value blocks

## 通用模式总结

```
单输入 1D:   Tensor(0 or 1) → tile((N,))
双输入 1D:   Tensor(0 or 1) × 2 + Tensor(1) → tile((N,))
行归约 2D:   Tensor(2) × 2 → tile((1, N))
Matmul:      Tensor(2) × 3 → MM 分块布局
BMM:         Tensor(3) × 3 → BMM 分块布局
RoPE:        Tensor(2 or 3) × multi → stride-dilation
Attention:   Tensor(3) × 4 → online softmax loop
```
