# Layout-Sensitive 算子参考（非连续 / stride / 窗口 / padding）

> 适用于 flip, narrow, permute, space-to-depth,
> pooling, 以及任何需要处理非连续/步幅/偏移输入的算子。

---

## 决策：在 arrangement 中处理 layout，还是在 wrapper 中处理？

两种合法策略——选择一种并说明你用了哪种：

**策略 1：Wrapper 连续化快速路径**（最简单，总是正确）

```python
def my_op(input, *, out=None):
    input = input.contiguous()  # 确保连续内存
    if out is None:
        out = torch.empty_like(input)
    kernel(input, out)
    return out
```

适用场景：layout 转换开销相对于计算可以忽略时；作为正确性 baseline。

**策略 2：In-arrangement layout**（零额外拷贝）

```python
# 用 permute / tile(strides=, dilation=) / ravel / flatten
# 让 kernel 直接读取原始存储
def arrangement(input, output):
    input_arranged = input.tile((1, 1, 3, 3), strides=(1, 1, 1, 1))  # 重叠窗口
    # ... ravel / flatten / permute ...
```

适用场景：避免拷贝对性能很重要时。

**推荐流程：** 先实现策略 1 作为正确性 oracle，如果性能需要再实现策略 2，并 benchmark 对比。

---

## 步幅/窗口 tiling — `tile(strides=, dilation=)`

`tile` 接受 `strides`（每个 tile 的起始间隔）和 `dilation`（tile 内元素间距）。

```python
# 非重叠 2x2 窗口（pooling 风格）
input.tile((1, 1, 2, 2))

# 重叠窗口：3x3 tile，步幅 1
input.tile((1, 1, 3, 3), strides=(1, 1, 1, 1))

# 膨胀窗口（如空洞卷积）
input.tile((1, 1, 3, 3), dilation=(1, 1, 2, 2))
```

---

## Space-to-depth / 窗口折叠 — ravel + flatten（已验证模式）

这是真正的 `max_pool2d` arrangement 模式：tile 出窗口维度 → `ravel()` 展平整个层级 → `flatten` 合并 batch 维度 → 重新 `tile` 成 block。

```python
def arrangement(input, output):
    # 1. tile 出窗口维度
    input_arranged = input.tile((1, 1, WINDOW_H, WINDOW_W))

    # 2. ravel 展平整个 tile 层级
    input_arranged = input_arranged.ravel()

    # 3. flatten 合并 batch 维度
    input_arranged = input_arranged.flatten(end_dim=4).flatten(start_dim=1)

    # 4. 重新 tile 成 block
    input_arranged = input_arranged.tile((BLOCK_SIZE, -1))

    # output 类似处理
    output_arranged = output.tile((1, 1, 1, 1)).ravel()
    output_arranged = output_arranged.flatten(end_dim=4).flatten(start_dim=1)
    output_arranged = output_arranged.tile((BLOCK_SIZE, -1))
    output_arranged.dtype = output_arranged.dtype.squeeze(1)

    return input_arranged, output_arranged
```

**pixel_unshuffle**（space-to-depth, factor r）：同样的形状代数——tile H,W 维度 by r, ravel/permute 使 r·r 窗口落在 channel 轴上，写入 `(B, C·r·r, H/r, W/r)` 输出。

---

## Transpose 风格 — `permute(dims)`

`permute` 在 arrangement 层面重排维度，不产生数据拷贝。

```python
# 转置访问模式
input_arranged = input.permute((1, 0))  # 交换 dim 0 和 dim 1
input_arranged = input_arranged.tile((BLOCK_M, BLOCK_N))
```

> **验证建议：** 使用 `simulate_arrangement` 确认 permute 后的元素映射正确。

---

## Padding — `pad(pad)`

`pad(pad)` 对每个维度添加 `(left, right)` padding。配合 `Tensor(other=...)` 使用，确保 padding 区域有定义的填充值。

```python
# 在 arrangement 中 pad
input_arranged = input.pad((0, 1, 0, 1))  # 右边各 pad 1
input_arranged = input_arranged.tile((BLOCK_H, BLOCK_W))
```

---

## 非连续输入正确性测试（此类别必须包含）

测试矩阵**必须**包含非连续输入：

```python
# 必须测试的非连续场景
x = torch.randn(64, 128, device="cuda")

# 1. 转置（最常见）
x_t = x.t()
assert not x_t.is_contiguous()
result = my_op(x_t)
expected = torch_reference(x_t)
assert torch.allclose(result, expected, atol=1e-5)

# 2. 步幅切片
x_s = x[::2, ::3]
assert not x_s.is_contiguous()
result = my_op(x_s)
expected = torch_reference(x_s)
assert torch.allclose(result, expected, atol=1e-5)

# 3. 部分切片
x_p = x[:, :100]
result = my_op(x_p)
expected = torch_reference(x_p)
assert torch.allclose(result, expected, atol=1e-5)
```

---

## 九齿 flatten()/tile() 与 PyTorch 的关键差异（CRITICAL）

> **九齿的 `flatten()` 和 `tile()` 在非连续 tensor 上尊重原始 strides，写入会传播回原始 buffer。PyTorch 的 `flatten()` 对非连续 tensor 创建副本。**

这是 slice_scatter 等 scatter 类算子能否正确实现的关键差异：

| 操作                               | PyTorch 行为             | 九齿行为                                                |
| ---------------------------------- | ------------------------ | ------------------------------------------------------- |
| `tensor.flatten()` 对连续 tensor   | 返回 view，写入传播      | 返回 view，写入传播                                     |
| `tensor.flatten()` 对非连续 tensor | **创建副本**，写入不传播 | **尊重 strides**，写入传播回原始 buffer                 |
| `tensor.permute(...).flatten()`    | **创建副本**（因非连续） | **尊重 strides**（写入通过 stride 映射到正确位置）      |
| `tensor.tile(shape)` 对非连续      | N/A                      | **尊重 strides**（生成的 Triton store 使用原始 stride） |

**验证实验（已在 MetaX C500 上确认）：**

```python
# 九齿 tile() 对转置 tensor：写入正确传播
x = torch.arange(16, device='cuda', dtype=torch.float32).reshape(4, 4)
out = torch.empty(4, 4, device='cuda')
out_t = out.t()  # non-contiguous

kernel = ninetoothed.make(arrangement, application, tensors)
kernel(x, out_t)
# 结果：out == x.t() ✓ — 写入通过 stride 正确映射

# 九齿 flatten() 对转置 tensor：写入正确传播
out2 = torch.empty(4, 3, device='cuda')
out2_t = out2.t()  # (3,4) non-contiguous
kernel(x, out2_t)
# 结果：out2 == x.t() ✓ — flatten 尊重 stride，写入传播
```

**实践影响：**
- **scatter 算子**：可以在 wrapper 中 permute 使 scatter dim 到最后，flatten 后传给 kernel，kernel 的写入会正确传播回原始 output buffer
- **安全模式**：`output.permute(perm).flatten()` → kernel 写入 → 原始 output 被修改 ✓
- **PyTorch 对比**：`output.permute(perm).flatten()` 在 PyTorch 中创建副本 → 写入不传播 ✗

---

## Storage Offset 与双非连续 Copy

### Storage Offset（存储偏移）

PyTorch tensor 可能有一个非零的 `storage_offset`（例如从大 buffer 中切片得到的 tensor）。九齿 kernel 接收 tensor 指针时，指针已经包含了 storage offset，因此 **kernel 内部不需要额外处理 storage_offset**。

```python
# 验证：storage_offset 对九齿 kernel 透明
buf = torch.randn(100, device='cuda')
slice_a = buf[10:50]   # storage_offset = 10
slice_b = buf[60:100]  # storage_offset = 60

# 九齿 kernel 直接使用这两个 tensor，无需手动处理 offset
kernel(slice_a, slice_b, output)  # 指针已包含 offset
```

**但要注意：** `tensor.data_ptr()` 会包含 offset，而 `.offsets()` 返回的是逻辑索引（从 0 开始），不包含 storage offset。这意味着：

| 概念                    | 是否包含 storage_offset | 说明                               |
| ----------------------- | :---------------------: | ---------------------------------- |
| `tensor.data_ptr()`     |            ✅            | 物理内存起始地址                   |
| `.offsets(dim)`         |            ❌            | 逻辑元素索引（从 0 开始）          |
| `tensor[idx]`（gather） |            ❌            | 逻辑索引，kernel 内部自动加 offset |
| `tile(strides=)`        |            ❌            | 逻辑步幅，kernel 内部自动处理      |

### 双非连续 Copy（Strided Source → Strided Destination）

当 source 和 destination 都是非连续 tensor 时（如两个不同 stride 的切片），Pattern 13 的 1D copy kernel 仍然可用，因为九齿的 `flatten()` 和 `tile()` 尊重原始 strides：

```python
# source 非连续（转置），destination 非连续（步幅切片）
source = torch.randn(4, 8, device='cuda').t()    # (8, 4) non-contiguous
dst_buf = torch.randn(20, 10, device='cuda')
dest = dst_buf[2:10, 1:5]                         # (8, 4) non-contiguous

# 九齿 flatten() 尊重 strides → 1D copy kernel 正确工作
kernel(source.flatten(), dest.flatten())
# source.flatten() → 通过 stride 正确读取
# dest.flatten()   → 通过 stride 正确写入
```

**已在 MetaX C500 验证：** 九齿的 flatten()/tile() 在非连续 tensor 上尊重原始 strides，写入会传播回原始 buffer（详见 §九齿 flatten()/tile() 与 PyTorch 的关键差异）。

---

## 常见陷阱

| 陷阱                                       | 说明                              | 修复                                          |
| ------------------------------------------ | --------------------------------- | --------------------------------------------- |
| `offsets()` 不接受 dim 参数                | 九齿 0.25.0 中 `offsets()` 无参数 | 直接使用 `tensor.offsets()`                   |
| output arrangement 镜像 input 的窗口 block | 可能导致 `RecursionError`         | 独立安排 output（如 `tile((1,))`）            |
| `unsqueeze` 在 arrangement 中 eval 失败    | 某些版本不支持                    | 在 wrapper 中做 reshape（`view`/`unsqueeze`） |
| 假设连续存储                               | 非连续输入会给出错误结果          | 用 `.contiguous()` 或 `tile(strides=)`        |
| 转置后 stride 计算错误                     | tile 形状与 stride 不匹配         | 用 `simulate_arrangement` 验证映射            |

---

## 调试工具

使用 `scripts/debug_arrangement.py` 在编译 kernel 之前验证 arrangement：

```bash
python scripts/debug_arrangement.py examples.my_op.kernel:arrangement
```

输出包含：
- 每个 tensor 的 source/target shape
- program 数量和 tile shape
- OOB 计数（-1 sentinel 值）
- 元素覆盖率（所有 source 元素是否至少被读取一次）
