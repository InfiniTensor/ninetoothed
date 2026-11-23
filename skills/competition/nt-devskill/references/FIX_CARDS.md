# NineToothed 故障诊断卡片 (Fix Cards)

当算子编译或运行出错时，按以下卡片逐一排查。每张卡片对应一种典型错误信号，提供根因和修复方案。

---

## 禁止行为（Anti-patterns）

> 以下行为会导致评分扣分或引入隐蔽 bug。**绝对不要做。**

| #   | 禁止行为                            | 正确做法                               |
| --- | ----------------------------------- | -------------------------------------- |
| A-1 | 放宽容差（atol/rtol）来让测试通过   | 修复根因（通常是缺少 fp32 累加）       |
| A-2 | 删除或跳过失败的测试用例            | 诊断并修复 kernel                      |
| A-3 | 修复一个算子时重构无关代码          | 最小化改动范围                         |
| A-4 | CPU PyTorch vs GPU NineToothed 对比 | 始终在同设备上对比（GPU vs GPU）       |
| A-5 | 单个 shape/dtype 就宣称正确         | 至少 3 shapes × 2 dtypes + 非连续输入  |
| A-6 | 归约操作不做 fp32 累加              | 始终 `ntl.cast(x, ntl.float32)` 后累加 |
| A-7 | 正确性未通过就优化性能              | 先正确，再优化                         |

---

## 错误特征表（先量化错误，再找原因）

> 先看数据分布，再看代码。

| 症状                                       | 最可能的根因                                  | 参见          |
| ------------------------------------------ | --------------------------------------------- | ------------- |
| fp16 失败，fp32 通过                       | 累加未 upcast 到 fp32                         | FC-15         |
| 输出全零（max_abs ≈ 0）                    | output 变量未赋值                             | FC-03         |
| NaN / Inf 存在                             | exp 溢出（无数值稳定性）/ 除以零              | FC-07         |
| 均匀偏差，cosine_sim ≈ 1                   | 系统性精度损失（错误的 scale / 截断）         | FC-15         |
| 周期性/条纹状错误                          | tile 边界或 stride/offset 计算错误            | FC-16         |
| 只有尾部元素错误                           | 尾部 tile 对齐或 mask 处理问题                | FC-01         |
| 结果每次运行不同                           | 缺少 sync（加 `torch.cuda.synchronize()`）    | —             |
| 小 shape 通过，大 shape 失败               | tiling 边界覆盖错误                           | FC-01 / FC-16 |
| `TypeError: not iterable` from arrangement | arrangement 返回了裸 `Tensor` 而非 tuple      | FC-17         |
| `F841` lint 警告                           | `output = ...` 赋值是 kernel 副作用           | FC-18         |
| 2D op 值错误，差整行                       | 对按行操作使用了 `tile((BLOCK_SIZE,))`        | FC-19         |
| `RecursionError` building kernel           | output arrangement 镜像了 input 的 block 层级 | FC-20         |
| 非连续输入给出错误结果                     | kernel 假设了连续存储                         | FC-21         |
| gather 索引错位（值来自错误位置）          | 跨 tensor 索引时 size_1 分解不匹配            | FC-22         |
| "slice step must be positive" 等平台限制  | MetaX PyTorch 不支持某些 API 参数组合          | FC-23         |
| generated source 异常（编译通过但结果错）  | Triton IR 缺少 mask / load-store 比例异常     | FC-24         |
| AOT build 失败                            | 编译缓存陈旧或 constexpr 不匹配              | FC-25         |

**MERE/MARE 精度阈值**（通过条件：MERE < threshold 且 MARE < 10× threshold）：

| dtype      | threshold      |
| ---------- | -------------- |
| float32    | 1.22e-4 (2⁻¹³) |
| float16    | 9.77e-4 (2⁻¹⁰) |
| bfloat16   | 7.81e-3 (2⁻⁷)  |
| int / bool | 精确匹配       |

---

## FC-01: Illegal Memory Access

**信号：**
```
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
```

**根因：** Partial tile 越界读取。当输入维度不能被 BLOCK_SIZE 整除时，最后一个 tile 包含越界元素。

**修复方案（二选一）：**

方案 A — 填充值（推荐用于 reduce 类算子）：
```python
# Tensor 声明时设置填充值
tensors = (Tensor(2, other=float("-inf")), Tensor(2))
```

方案 B — 显式 mask（推荐用于 attention 等复杂算子）：
```python
# 在 application 中用 ntl.where 屏蔽越界
mask = tensor.offsets(-1) < tensor.source.shape[-1]
value = ntl.where(mask, value, 0.0)
```

**验证方法：** 使用不能整除 BLOCK_SIZE 的输入尺寸测试（如 `1823×781`）。

---

## FC-02: Cannot Squeeze Dim

**信号：**
```
ValueError: cannot squeeze dim X
```

**根因：** `dtype` 操作中的维度追踪错误。每次 `tile()` 会增加一层维度，`expand()` 会改变形状，squeeze 时必须指定正确维度。

**修复流程：**

1. 手动追踪维度变化：
```
原始 Tensor(2):                    2 dims
.tile((BM, BK)):                   +1 → 3 dims (outer grid + inner tile)
.tile((1, -1)):                    +1 → 4 dims
.expand(...):                      保持 4 dims
.dtype.squeeze(0):                 → 3 dims
```

2. 对于 3D+ tiling（如 bmm, attention），需要**双层** dtype 访问：
```python
arranged.dtype = arranged.dtype.squeeze((0, 1))        # 外层
arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))  # 内层
```

**验证方法：** 在 arrangement 中插入 `print(arranged.shape)` 确认维度。

---

## FC-03: Kernel Returns Zeros

**信号：** 输出全为零，无报错。

**根因：** `application` 函数中未正确赋值给输出参数。NineToothed 通过 **赋值语句**（`output = result`）触发写入，不是通过返回值。

**修复：**
```python
# 错误 ✗
def application(input, output):
    result = input * 2
    # 忘记赋值！

# 正确 ✓
def application(input, output):
    output = input * 2   # 必须赋值给参数名 output
```

**注意：** 赋值变量名必须与 arrangement 返回的参数名一致。

---

## FC-04: Autotuning Timeout / Hang

**信号：** 首次运行时卡住或编译极慢（>5 分钟）。

**根因：** `block_size()` 或 `Symbol(meta=True)` 触发的搜索空间过大。

**修复方案（按优先级）：**

1. 添加 `upper_bound` 缩小搜索范围：
```python
BLOCK_SIZE = Symbol("BLOCK_SIZE", meta=True, upper_bound=128)
```

2. 开发阶段用固定值替代 autotuning：
```python
BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)  # 编译期常量
# 并在调用时传 BLOCK_SIZE=1024
```

3. 对于 matmul 的三个 block_size，先固定再逐一调优：
```python
# 先用固定值
BM, BN, BK = 128, 128, 32
# 确认可运行后再换 block_size()
```

---

## FC-05: Broadcast Shape Mismatch

**信号：**
```
RuntimeError: The size of tensor a (X) must match the size of tensor b (Y)
```

**根因：** 在 reduction 后使用结果时，维度不匹配。`ntl.max(x, axis=1)` 会降维，后续运算需要重新扩展维度。

**修复：**
```python
# softmax 中的典型修复
row_max = ntl.max(x, 1)              # shape: [M]
shifted = x - row_max[:, None]       # shape: [M, BLOCK] ← 加 [:, None] 扩展
exp_x = ntl.exp(shifted)
row_sum = ntl.sum(exp_x, 1)          # shape: [M]
output = exp_x / row_sum[:, None]    # 同样需要扩展
```

---

## FC-06: Kernel Launch Fails Silently

**信号：** 没有报错也没有输出，或者输出 shape 为空。

**根因：** Grid 维度为 0（输入张量为空或 tile 参数错误）。

**修复：**
1. 检查输入张量是否在 CUDA 设备上：`assert x.is_cuda`
2. 检查 grid 维度不为零：`assert all(s > 0 for s in input.shape)`
3. 确认 `BLOCK_SIZE <= input.shape[-1]` 或已处理 partial tile
4. 添加诊断打印：
```python
# torch wrapper 中
print(f"Grid: {output.numel() // BLOCK_SIZE}, BLOCK: {BLOCK_SIZE}")
```

---

## FC-07: Numerical Instability

**信号：** 输出包含 NaN、Inf，或 `max_diff` 远大于容差。

**根因：** 浮点精度不足或数值不稳定。

**修复清单：**

| 场景          | 修复                                              |
| ------------- | ------------------------------------------------- |
| Softmax 溢出  | 先减 max：`x - max(x)` 再 exp                     |
| Sigmoid 精度  | `ntl.sigmoid(ntl.cast(x, ntl.float32))`           |
| rsqrt 精度    | `ntl.rsqrt(ntl.cast(variance, ntl.float32))`      |
| MatMul 累加   | 用 `ntl.zeros(..., dtype=ntl.float32)` 做累加器   |
| Attention exp | 用 `ntl.exp2` + 预乘 `log2(e)` 替代 `ntl.exp`     |
| 最终写回      | `output = acc.to(input.dtype.dtype)` 转回原始精度 |

---

## FC-08: Dtype Mismatch in ntl.dot

**信号：**
```
AssertionError or compile error about dtype in dot operation
```

**根因：** `ntl.dot` 要求输入为相同类型，且通常为 `float16` 或 `bfloat16`（不直接支持 float32 的 dot）。

**修复：**
```python
# 确保 dot 的输入是 half precision
a_half = ntl.cast(a, ntl.float16)
b_half = ntl.cast(b, ntl.float16)
result = ntl.dot(a_half, b_half)  # 返回 float32
```

---

## FC-09: Composition Import Error

**信号：**
```
ModuleNotFoundError: No module named 'examples.matmul.kernel'
```

**根因：** 组合算子依赖其他 examples 的子模块，但 `sys.path` 或 `__init__.py` 未正确配置。

**修复：**
1. 确保项目根目录在 `sys.path` 中（`scripts/validate.py` 和 `tests/conftest.py` 会自动处理）
2. 每个 `examples/<op>/` 目录必须有 `__init__.py`
3. 导入路径规范：
```python
# 从 kernel.py 导入
from examples.matmul.kernel import application, arrangement, kernel
# 从 __init__.py 导入 torch wrapper
from examples.matmul import mm
```

---

## FC-10: Tile Shape Not Power of 2

**信号：** 编译报错或性能极差。

**根因：** GPU warp size 要求 tile 维度为 2 的幂次（16/32/64/128），否则硬件效率低下。

**修复：**
```python
# 使用 next_power_of_2 向上取整
import math
BLOCK_SIZE = 1 << math.ceil(math.log2(N))  # 如 N=781 → 1024
```

或在 torch wrapper 中：
```python
BLOCK_SIZE = triton.next_power_of_2(x.shape[-1])
kernel(x, output, BLOCK_SIZE=BLOCK_SIZE)
```

---

## FC-11: MetaX GPU 兼容性

**信号：** 在 NVIDIA GPU 上可运行，但在 MetaX GPU 上报错。

**根因：** MetaX GPU 使用 MACA 兼容层模拟 CUDA API，某些 Triton 特性可能不完全支持。

**修复方案：**
1. 确认环境变量已设置（`MACA_PATH`, `LD_LIBRARY_PATH`）
2. 避免使用 `ntl.atomic_add`（MetaX 可能不支持）
3. 避免 float32 的 `ntl.dot`（用 float16 输入 + float32 累加器）
4. 降低 tile 大小（MetaX SM 数量较少，`BLOCK_SIZE=64` 可能更合适）
5. 用 `scripts/doctor.py` 检查环境

---

## FC-12: Compilation Cache Stale

**信号：** 修改了 kernel 代码但运行时行为不变。

**根因：** NineToothed 使用基于 SHA256 的缓存（`~/.ninetoothed/`），修改代码后缓存可能未失效。

**修复：**
```bash
# 清除缓存
rm -rf ~/.ninetoothed/

# 或在 Python 中
import shutil, os
cache_dir = os.path.expanduser("~/.ninetoothed")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
```

---

## FC-13: libdevice 编译失败（MetaX）

**信号：**
```
CompilationError: AttributeError("module 'triton.language' has no attribute 'libdevice'")
```

**根因：** 在 `application` 函数中使用 `ntl.libdevice.lgamma(...)` 或类似的 `ntl.libdevice.X` 调用。ninetoothed 编译器将其映射为 `triton.language.libdevice.X`，但 MetaX 的 Triton fork 中 `libdevice` 不在 `triton.language` 下，而在 `triton.language.extra.libdevice`。

**修复：**
```python
# ❌ 错误写法
import ninetoothed.language as ntl
def application(input, output):
    output = ntl.libdevice.lgamma(input)

# ✅ 正确写法：模块级全局导入 libdevice
from triton.language.extra import libdevice
import ninetoothed.language as ntl
def application(input, output):
    output = libdevice.lgamma(ntl.cast(input, ntl.float32))
```

**原理：** ninetoothed 的 `_Inliner` 会扫描 `application` 函数的全局变量，检测到 `libdevice` 模块引用后，自动在生成的 Triton kernel 中插入 `from triton.language.extra import libdevice as <alias>`。使用 `ntl.libdevice` 不会触发此检测。

---

## FC-14: __init__.py 远程同步失败

**信号：**
```
ImportError: cannot import name 'xxx' from partially initialized module
```
或
```
ModuleNotFoundError: No module named 'xxx'
```
（但本地文件明明存在）

**根因：** 上传了本地的 `__init__.py`，但其中包含了远程服务器上尚不存在的模块引用。通常发生在本地有未提交的修改（如之前开发的其他算子），而远程服务器还是干净状态。

**修复：**
1. **上传前检查远程状态：**
   ```bash
   # 通过 MCP 检查远程有哪些文件
   remote_ls("path/to/kernels")
   ```
2. **只上传新算子的文件，不要直接覆盖 __init__.py**
3. **如果需要更新 __init__.py，在远程编辑而非上传本地版本：**
   ```python
   # 通过 run_python 在远程追加一行 import
   run_python("""
   import re
   with open('path/__init__.py', 'r') as f:
       content = f.read()
   content = content.replace('    le,', '    le,\\n    lgamma,')
   with open('path/__init__.py', 'w') as f:
       f.write(content)
   """)
   ```
4. **或者：先用 git 确认远程基线，再在基线上增量修改**

---

## FC-15: fp16/bf16 精度不足（累加未 upcast）

**信号：**
```
fp16 测试 FAIL（atol=1e-3），fp32 测试 PASS
或 MERE/MARE 超过阈值
```

**根因：** 归约操作（sum/softmax/norm）在 fp16 精度下累加，导致溢出或精度损失。

**修复：**
```python
# ❌ 错误：fp16 累加
output = ntl.sum(input, axis=1)

# ✅ 正确：先 cast 到 fp32 再累加
input_f32 = ntl.cast(input, ntl.float32)
output = ntl.sum(input_f32, axis=1).to(output.dtype)
```

> **禁止放宽容差来通过测试。** 精度不足说明 kernel 实现有缺陷。

---

## FC-16: Tile 边界/stride 计算错误

**信号：**
```
周期性/条纹状错误模式
或小 shape 通过但大 shape 失败
```

**根因：** tile 的 stride 或 offset 计算有误，导致部分元素被跳过或重复读取。

**修复：**
1. 使用 `debug_arrangement.py` 验证 tile 映射：
   ```bash
   python scripts/debug_arrangement.py module.path:arrangement
   ```
2. 检查 `oob_count` 和 `unique_sources` 是否正确
3. 确认 `tile(strides=)` 参数与预期的窗口步幅一致

---

## FC-17: arrangement 返回 TypeError: not iterable

**信号：**
```
TypeError: cannot unpack non-iterable Tensor object
```

**根因：** arrangement 函数返回了裸 `Tensor` 而非 tuple。

**修复：**
```python
# ❌ 错误：返回裸 Tensor
def arrangement(input, output, BLOCK_SIZE=1024):
    return input.tile((BLOCK_SIZE,))  # 只有一个返回值

# ✅ 正确：返回 tuple（注意末尾逗号）
def arrangement(input, output, BLOCK_SIZE=1024):
    input_arr = input.tile((BLOCK_SIZE,))
    output_arr = output.tile((BLOCK_SIZE,))
    return input_arr, output_arr  # 返回 tuple
```

---

## FC-18: F841 lint 警告（output 赋值未使用）

**信号：**
```
F841 local variable 'output' is assigned to but never used
```

**根因：** `output = ...` 赋值是 kernel 的副作用（写入 GPU 内存），但 lint 工具无法识别。

**修复：** 在赋值行末尾添加 `# noqa: F841`：
```python
def application(input, output):
    output = input + 1  # noqa: F841
```

---

## FC-19: 2D 算子值错误（差整行）

**信号：**
```
2D op 输出值错误，每行偏移固定值
```

**根因：** 对按行操作（如 softmax）使用了 1D tile `tile((BLOCK_SIZE,))`，而不是 2D row tile `tile((1, BLOCK_SIZE))`。

**修复：**
```python
# ❌ 错误：1D tile（tile 的是 dim 0，即行维度）
input.tile((BLOCK_SIZE,))

# ✅ 正确：2D row tile（每次取一行）
input.tile((1, BLOCK_SIZE))
```

---

## FC-20: RecursionError building kernel

**信号：**
```
RecursionError: maximum recursion depth exceeded
```

**根因：** output arrangement 强制镜像了 input 的 block 层级结构，导致无限递归。

**修复：** 独立安排 output arrangement，不要与 input 共享 block 层级：
```python
# ❌ 错误：output 镜像 input
output_arranged = output.tile((1, 1, WINDOW_H, WINDOW_W))  # 与 input 相同层级

# ✅ 正确：output 独立安排
output_arranged = output.tile((1, 1, 1, 1)).ravel()
output_arranged = output_arranged.flatten(end_dim=4).flatten(start_dim=1)
output_arranged = output_arranged.tile((BLOCK_SIZE, -1))
output_arranged.dtype = output_arranged.dtype.squeeze(1)
```

---

## FC-21: 非连续输入给出错误结果

**信号：**
```
连续输入测试 PASS，但 transposed/strided 输入 FAIL
```

**根因：** kernel 假设了连续存储（contiguous），但输入可能是转置或步幅切片。

**修复（二选一）：**

方案 A — wrapper 连续化快速路径（最简单）：
```python
def my_op(input, *, out=None):
    input = input.contiguous()  # 确保连续
    if out is None:
        out = torch.empty_like(input)
    kernel(input, out)
    return out
```

方案 B — 在 arrangement 中处理 stride（零额外拷贝）：
```python
# 用 tile(strides=) 或 permute 表达 layout
# 详见 references/LAYOUT.md
```

> **必须测试非连续输入。** 参见 `references/LAYOUT.md` §非连续输入正确性测试。

---

## FC-22: 跨 tensor gather 索引错位（静默数据错误）

**信号：**
```
输出数值错误但无异常，部分位置的值来自错误的源位置
差值呈现固定偏移模式（不是随机错误）
```

**根因：** 九齿的 `tensor[idx]` 使用 arrangement 中**第一个 tensor 的 `size_1`** 做索引分解（`row = idx // size_1, col = idx % size_1`）。当用 tensor A 的 offsets 去 gather tensor B 时，如果 A 和 B 的 `size_1` 不同，分解出的 row/col 映射到 B 的错误物理位置。

**修复方案（按优先级）：**

方案 A — 统一 size_1（最简单）：
```python
# 所有 tensor flatten 成 1D，block_size 统一
def arrangement(source, output, BLOCK_SIZE=BLOCK_SIZE):
    return source.flatten().tile((BLOCK_SIZE,)), output.flatten().tile((BLOCK_SIZE,))
# 此时 source.size_1 == output.size_1 == BLOCK_SIZE → gather 正确
```

方案 B — 避免跨 tensor gather（推荐）：
```python
# kernel 只做 1D copy，scatter 索引在 wrapper 中处理
# 详见 CODE_TEMPLATES.md Pattern 13
def application(source, output_slice):
    output_slice = source  # 简单拷贝，无跨 tensor 索引
```

**验证方法：** 用已知输入（如 `torch.arange`）测试，检查输出中每个值是否来自预期的源位置。

> **参考：** `references/API_REFERENCE.md` §Cross-Tensor Gather Indexing

---

## FC-23: 平台 API 限制（负 step / 不支持的操作）

**信号：**
```
RuntimeError: slice step must be positive
或 torch.xxx 不支持某些参数组合
```

**根因：** MetaX 上的 PyTorch（如 2.6.0+metax）可能不支持某些标准 PyTorch API 的全部参数组合。常见限制：
- `torch.slice_scatter` 不支持负 step
- 某些 `torch.xxx` 操作在 MetaX 上行为与 NVIDIA GPU 不同

**修复方案：**

1. **检测平台限制**：
```python
import torch
try:
    torch.slice_scatter(torch.zeros(5, device='cuda'), torch.zeros(2, device='cuda'), step=-1)
    HAS_NEGATIVE_STEP = True
except RuntimeError:
    HAS_NEGATIVE_STEP = False
```

2. **wrapper 中实现缺失功能**：
```python
if step < 0 and not HAS_NEGATIVE_STEP:
    # 翻转 src + 转换为等价正 step 操作
    flip_idx = torch.arange(src.size(dim) - 1, -1, -1, device=src.device)
    src = src.index_select(dim, flip_idx[:actual_len])
    new_start = start + (actual_len - 1) * step
    return my_op(input, src, dim=dim, start=new_start, step=-step)
```

3. **自定义参考实现**（当 torch 无法提供参考时，**必须使用此方案**）：
```python
# 用手动计算或 NumPy/CPU PyTorch 替代 torch 参考
def manual_slice_scatter_ref(input, src, dim, start, end, step):
    """Pure-Python reference that works regardless of platform limitations."""
    out = input.clone()
    slices = [slice(None)] * input.ndim
    slices[dim] = slice(start, end, step)
    out[slices] = src
    return out

# 在测试中使用自定义参考
ref = manual_slice_scatter_ref(input, src, dim, start, end, step)
assert torch.allclose(nt_out, ref, atol=atol, rtol=rtol)
```

> **⚠️ 禁止跳过测试。** 不要使用 `pytest.skip()` 跳过平台不支持的用例。正确做法是编写自定义参考实现来替代 `torch.xxx`，确保所有测试用例都有验证。参见 Anti-pattern A-2（删除或跳过失败的测试用例）。

---

## FC-24: Generated Source 异常（Triton IR 问题）

**信号：**
```
kernel 能编译但输出错误，且 application 代码看起来正确
或 inspect_generated.py 报告异常 load/store 比例或缺少 mask
```

**根因：** 九齿编译器生成的 Triton IR 可能与预期不符。常见问题：
- **缺少 boundary mask**：partial tile 无 mask → 越界读写
- **load/store 比例异常**：`Load/Store ratio > 3` 暗示冗余加载（tile 重叠或 expand 错误）
- **gather 索引错位**：`tensor[idx]` 使用了错误的 `size_1` 分解（详见 FC-22）
- **dtype 提升**：编译器自动提升了 dtype 导致精度问题

**诊断流程：**
```bash
# 1. 运行 generated source 检查
python scripts/inspect_generated.py --op <name>

# 2. 检查输出中的关键指标：
#    - Load/Store 数量（正常比例 ≤ 3:1）
#    - mask 存在性（非对齐 shape 必须有 mask）
#    - fp32 cast 存在性（归约操作应有 fp32 累加）
#    - num_warps（tile 大小是否合理）

# 3. 如果发现问题，直接查看 Triton IR：
#    cat ~/.ninetoothed/<hash>/kernel.py
```

**修复方案：**
1. 缺少 mask → 在 application 中显式添加 `ntl.where(offsets < shape, value, fill)`
2. load/store 比例高 → 检查 arrangement 是否有不必要的 expand 或重复 tile
3. dtype 提升 → 在关键路径显式 `ntl.cast(x, ntl.float32)` 控制精度

---

## FC-25: AOT Build 失败

**信号：**
```
bash scripts/aot_build_smoke.sh <op> 输出 FAIL
或 AOT 编译后的 kernel 行为与 JIT 不同
```

**根因：** AOT (Ahead-of-Time) 编译可能因以下原因失败：
- **编译缓存陈旧**：`~/.ninetoothed/` 中旧缓存干扰 AOT 编译
- **符号依赖缺失**：kernel 依赖的外部模块（如 libdevice）在 AOT 模式下未正确链接
- **constexpr 值不匹配**：AOT 编译时的 Symbol 值与运行时不一致

**诊断流程：**
```bash
# 1. 运行 AOT 烟雾测试
bash scripts/aot_build_smoke.sh <op>

# 2. 检查输出中的关键信息：
#    - "AOT build: OK" → 编译成功
#    - "AOT build: FAIL" → 检查后续错误消息
#    - "Generated source: OK" → Triton IR 生成正常

# 3. 如果 AOT 失败但 JIT 正常：
rm -rf ~/.ninetoothed/   # 清除缓存
bash scripts/aot_build_smoke.sh <op>  # 重试
```

**修复方案：**
1. 缓存陈旧 → `rm -rf ~/.ninetoothed/` 后重试
2. libdevice 依赖 → 确保 `from triton.language.extra import libdevice` 在模块级（FC-13）
3. constexpr 不匹配 → AOT 模式下用固定 Symbol 值，不用 `block_size()` autotuning
4. 如果 AOT 持续失败 → 标记为 "AOT unsupported"，使用 JIT 模式

---

## 快速诊断流程

```
出错 → 检查错误消息 → 匹配卡片编号
  ├── "illegal memory access"     → FC-01
  ├── "cannot squeeze"            → FC-02
  ├── 输出全零                    → FC-03
  ├── 卡住/极慢                   → FC-04
  ├── "size must match"           → FC-05
  ├── 静默失败                    → FC-06
  ├── NaN/Inf/大误差              → FC-07
  ├── dot dtype error             → FC-08
  ├── ModuleNotFoundError         → FC-09
  ├── 性能极差                    → FC-10
  ├── 仅 MetaX 报错               → FC-11
  ├── 修改无效                    → FC-12
  ├── "no attribute 'libdevice'"  → FC-13
  ├── "cannot import name" (远程) → FC-14
  ├── fp16 失败 fp32 通过         → FC-15
  ├── 周期性/条纹状错误           → FC-16
  ├── "TypeError: not iterable"   → FC-17
  ├── F841 lint 警告              → FC-18
  ├── 2D op 值差整行              → FC-19
  ├── RecursionError              → FC-20
  ├── 非连续输入结果错误          → FC-21
  ├── gather 索引错位（静默）     → FC-22
  ├── 平台 API 限制（负 step 等） → FC-23
  ├── generated source 异常       → FC-24
  └── AOT build 失败              → FC-25
```
