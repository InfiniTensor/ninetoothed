# Tensor 声明与元操作参考

## Tensor 构造

### 基本 Tensor

```python
from ninetoothed import Tensor

# N 维 tensor
Tensor(ndim, dtype=dtype)

# 带 padding fill 的 tensor（reduction/pooling 用）
Tensor(ndim, dtype=dtype, other=0)                # norm 类
Tensor(ndim, dtype=dtype, other=float("-inf"))     # softmax, max pooling
```

### 标量 Tensor

```python
# 编译时常量（推荐用于条件系数、枚举值、slope 等）
Tensor(0, constexpr=True, value=default_value)

# 运行时标量（不推荐用于浮点系数，可能类型不兼容）
Tensor(0, dtype=ninetoothed.float64)    # ⚠️ 会生成 fp64 指针，与 fp32 输入不兼容
Tensor(0, dtype=ninetoothed.int64)       # 整数标量
```

### 带约束的 Tensor

```python
# shape_options 约束特定维度
Tensor(4, dtype=dtype, shape_options=(
    None, None, None,
    {"constexpr": True, "upper_bound": 128}
))

# 带 constexpr shape 的 reduction 输入
Tensor(ndim, dtype=dtype, other=float("-inf"), shape_options={"constexpr": True})
```

## Tensor 元操作链

### tile — 分块

```python
tensor.tile((block_size_m, block_size_n))    # 2D 分块
tensor.tile((1, -1))                          # 1 保持，-1 自动推断
```

### expand — 广播

```python
tensor.expand((-1, output_shape))   # -1 保持原维度大小
tensor.expand((output_shape, -1))
```

### squeeze / unsqueeze — 维度操作

```python
tensor.dtype = tensor.dtype.squeeze(0)    # 去除退化维度 0
tensor.dtype = tensor.dtype.squeeze(1)    # 去除退化维度 1
```

### permute — 维度重排

```python
tensor.permute((1, 0, 2))    # 交换前两个维度
```

### flatten / ravel — 展平

```python
tensor.flatten(start_dim=1)    # 从维度 1 开始展平
tensor.ravel()                 # 全部展平
```

### pad — 填充

```python
tensor.pad((0, pad_size))    # 在最后一维右侧填充
```

### indexing — 索引

```python
tensor[0]     # 取第一维的切片
tensor[-1]    # 取最后一维的切片
```

## 典型操作链模式

### Element-wise（最简单）

```
input → tile(block_size) → [compute] → store
output → tile(block_size) → [compute] → store
```

无 tile 维度分裂，直接逐块处理。

### Reduction（一维规约）

```
input → tile((block_size,)) → tile((-1,)) → 遍历 shape[0] 累加
output → 直接对应
```

reduction arrangement 自动处理维度转置和 padding。

### Matmul（tile-expand-squeeze）

```
input(M,K) → tile(M_b,K_b) → tile(1,-1) → expand(-1,N_b) → squeeze(0)
other(K,N) → tile(K_b,N_b) → tile(-1,1) → expand(M_b,-1) → squeeze(1)
output(M,N) → tile(M_b,N_b)
```

在 K 维度做 tile，expand 到对应输出维度，squeeze 去退化维。

## 标量参数选择决策

```
标量参数需要与输入 tensor 做逐元素运算吗？
├── 是 → 标量值是否运行时可变？
│   ├── 是 → Tensor(ndim, dtype=dtype)，torch 层直接传值
│   └── 否 → 仍然用 Tensor(ndim, dtype=dtype)，更简单
└── 否 → 标量是否用于条件分支或编译时确定？
    ├── 是 → Tensor(0, constexpr=True, value=默认值)
    └── 否 → 如果是整数枚举 → Tensor(0, constexpr=True, value=enum_value)
              如果是浮点系数 → ⚠️ 用 constexpr，不要用 Tensor(0, dtype=ninetoothed.float64)
```

## premake 返回值规范

```python
def premake(参数..., dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, 配置参数=值, block_size=block_size)
    tensors = (
        Tensor(...),    # 输入
        Tensor(...),    # 输出
        # ... 其他 tensor
    )
    return arrangement_, application, tensors
```

- `arrangement_` 是 `functools.partial` 绑定的 arrangement 函数
- `application` 是计算函数的引用
- `tensors` 是 Tensor 模板元组，顺序 = kernel 调用时参数顺序
