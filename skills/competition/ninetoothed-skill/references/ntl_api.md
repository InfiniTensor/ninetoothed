# ntl（ninetoothed.language）API 参考

## 创建操作

| 操作 | 签名 | 说明 |
|------|------|------|
| `ntl.zeros` | `zeros(shape, dtype=ntl.float32)` | 创建零填充张量 |
| `ntl.full` | `full(shape, value, dtype=ntl.float32)` | 创建常量填充张量 |
| `ntl.arange` | `arange(start, end)` | 创建整数范围 |

## 数学操作

| 操作 | 签名 | 说明 |
|------|------|------|
| `ntl.exp` | `exp(x)` | 指数函数（float16 输入需先 cast 到 float32） |
| `ntl.exp2` | `exp2(x)` | 2^x（用于 exp2 trick 优化） |
| `ntl.log` | `log(x)` | 自然对数（float16 输入需先 cast 到 float32） |
| `ntl.log2` | `log2(x)` | 以 2 为底对数 |
| `ntl.sqrt` | `sqrt(x)` | 平方根 |
| `ntl.sin` | `sin(x)` | 正弦（需要 float32 输入） |
| `ntl.cos` | `cos(x)` | 余弦（需要 float32 输入） |
| `ntl.abs` | `abs(x)` | 绝对值 |
| `ntl.tanh` | `tanh(x)` | 双曲正切（需要 float32 输入） |

## 规约操作

| 操作 | 签名 | 说明 |
|------|------|------|
| `ntl.sum` | `sum(x, axis=None)` | 求和 |
| `ntl.max` | `max(x, axis=None)` | 最大值 |
| `ntl.min` | `min(x, axis=None)` | 最小值 |

## 比较与选择

| 操作 | 签名 | 说明 |
|------|------|------|
| `ntl.where` | `where(condition, x, y)` | 条件选择 |
| `ntl.maximum` | `maximum(x, y)` | 逐元素最大 |
| `ntl.minimum` | `minimum(x, y)` | 逐元素最小 |

## 类型转换

| 操作 | 签名 | 说明 |
|------|------|------|
| `ntl.cast` | `cast(x, dtype)` | 类型转换 |
| `x.to` | `x.to(dtype)` | 链式类型转换 |

## 线性代数

| 操作 | 签名 | 说明 |
|------|------|------|
| `ntl.dot` | `dot(a, b, input_precision=None)` | 矩阵乘法（块内） |
| `ntl.trans` | `trans(x)` | 转置 |

## 内存操作

| 操作 | 签名 | 说明 |
|------|------|------|
| `ntl.load` | `load(pointer, boundary=None)` | 从全局内存加载 |
| `ntl.store` | `store(pointer, value, boundary=None)` | 存储到全局内存 |
| `ntl.atomic_add` | `atomic_add(pointer, value)` | 原子加 |

## 程序信息

| 操作 | 签名 | 说明 |
|------|------|------|
| `ntl.program_id` | `program_id(axis)` | 获取当前 program ID |
| `ntl.offsets` | `offsets(dim)` | 获取指定维度的偏移量 |

## 数据类型

| 类型 | 说明 |
|------|------|
| `ntl.float16` | 半精度浮点 |
| `ntl.bfloat16` | BF16 |
| `ntl.float32` | 单精度浮点 |
| `ntl.float64` | 双精度浮点 |
| `ntl.int8` / `ntl.int16` / `ntl.int32` / `ntl.int64` | 有符号整数 |
| `ntl.uint8` / `ntl.uint16` / `ntl.uint32` / `ntl.uint64` | 无符号整数 |

## Symbol 参数（ninetoothed 顶层）

```python
from ninetoothed import Symbol, block_size
```

| 参数 | 说明 |
|------|------|
| `block_size()` | 自动调优的 meta symbol，默认范围 32-1024，必须是 2 的幂 |
| `constexpr` | 编译时常量（通过 `Tensor(0, constexpr=True, value=...)` 使用） |
| `lower_bound` | shape 维度的下界约束 |
| `upper_bound` | shape 维度的上界约束 |
| `power_of_two` | 约束值为 2 的幂 |

## libdevice 扩展

**优先检查 libdevice 再自己实现。** 通过 `from ninetoothed.language import libdevice` 访问 150+ CUDA libdevice 函数。

**关键函数**（完整列表可通过 `dir(libdevice)` 查看）：

| 类别 | 函数举例 |
|------|----------|
| 数学 | `pow`, `exp`, `log`, `sqrt`, `sin`, `cos`, `tanh`, `fma` |
| 位操作 | `float_as_int`, `int_as_float`, `clz`, `popc`, `brev` |
| 舍入 | `rint`, `floor`, `ceil`, `trunc`, `round`, `nearbyint` |
| 符号 | `copysign`, `signbit`, `fabs` |
| 特殊 | `nextafter`, `erf`, `tgamma`, `lgamma` |

**使用方式**：`from ninetoothed.language import libdevice`（不要用 `ntl.libdevice.xxx`，直接用 `libdevice.xxx`）。
**注意事项**：部分函数默认 double 精度，float32 输入需验证 subnormal 边界是否正确。

## 调试工具

| 工具 | 用途 |
|------|------|
| `ninetoothed.debugging.simulate_arrangement` | 验证 arrangement 映射是否正确（调试时使用） |

## 使用注意事项

1. **float16 安全**：`exp`、`log`、`sin`、`cos`、`tanh` 等数学函数在 float16 下易溢出，务必先 `ntl.cast(x, ntl.float32)`
2. **累加精度**：浮点累加器始终使用 `ntl.float32`
3. **输出赋值**：所有 output 赋值语句加 `# noqa: F841`
4. **禁止**：`torch.*`、`triton.*`、`cuda.*`、`numpy.*`、import、print、list/dict
