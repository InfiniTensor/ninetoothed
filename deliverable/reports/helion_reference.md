# Helion Code Generation 参考分析

**日期**: 2026-07-12  
**用途**: 对比参考，为 NineToothed T1-2-1 特化优化提供方向指引  
**来源**: https://github.com/pytorch/helion (commit as of 2026-07-12)

---

## 1. Helion 概述

Helion 是 PyTorch 团队开发的、基于 Triton 的高层 DSL，与 NineToothed 定位高度相似。2026 年 6 月在 PLDI 2026 发表 Tutorial。核心思路是通过自动调优、多层 IR 和代码生成，将用户的高级 kernel 描述编译为高效的 Triton 代码。

### 与 NineToothed 的架构对比

| 维度 | Helion | NineToothed |
|------|--------|-------------|
| IR 层次 | FX Graph → Device IR → AST | 符号 Tensor 对象 → 直接 AST 生成 |
| 代码生成入口 | `generate_ast.py` (`GenerateAST`) | `generation.py` (`CodeGenerator`) |
| mask 生成 | `SubscriptIndexing.per_dim_indexing()` | `_generate_offsets_and_mask()` |
| 指针算术 | 仅非 size-1 维度生成 `idx * stride` | 所有维度生成 `offset[dim] * stride[dim]` |
| 后端 | Triton / Pallas / CuTe / Metal | Triton 单后端 |
| Mask 优化 Pass | `remove_unnecessary_masking` | 无 |
| 循环结构 | `tile_strategy.py` 创建 DeviceLoopState | `_generate_pid_indices` + `_generate_innermost_indices` |

---

## 2. 关键代码路径

### 2.1 文件结构

| 文件 | 功能 |
|------|------|
| `helion/_compiler/kernel_compiler.py` | 编译管线编排 (parse → unroll → customize → type_propagate → finalize_config → lower) |
| `helion/_compiler/generate_ast.py` | 核心代码生成：`GenerateAST` 遍历 FX Graph 生成 Triton/Pallas/CuTe AST |
| `helion/_compiler/indexing_strategy.py` | 三种索引策略：`PointerIndexingStrategy`、`BlockPtrIndexingStrategy`、`TensorDescriptorIndexingStrategy` |
| `helion/_compiler/node_masking.py` | Mask 优化 Pass：`remove_unnecessary_masking`、`defer_pallas_load_masks` |
| `helion/_compiler/tile_strategy.py` | Tile 策略分发和 DeviceLoopState 管理 |
| `helion/_compiler/device_ir.py` | Device IR：`DeviceIR` + `GraphInfo` 变体 (Root/ForLoop/ReductionLoop/If/WhileLoop) |
| `helion/_compiler/aten_lowering.py` | `torch.ops.aten.*` 降低到 Helion AST |
| `helion/language/memory_ops.py` | `hl.load` / `hl.store` API |
| `helion/_compiler/cute/layout_rules.py` | CuTe 后端 layout 规则 |
| `helion/_compiler/cute/cute_mma.py` | CuTe 矩阵乘法 codegen，含 TMA fast path |

### 2.2 Mask 生成流程

`PointerIndexingStrategy.codegen_load()` 的核心流程：

```
state = DeviceLoopState(offset_var, index_var, mask_var)
  → SubscriptIndexing.create(state, fake_tensor, subscript, extra_mask)
    → compute_per_dim_indexing()
      → 对每个下标元素计算 index_expr 和 mask_expr
      → 当 mask_var(block_id) 返回 None 时，不生成 mask
    → 返回 PerDimIndexing(index_values, mask_expr, broadcast_dims)
  → 生成 tl.load(ptr + offset, mask, other=0) 或 tl.load(ptr + offset) 无 mask
```

### 2.3 Mask 优化 Pass

**`remove_unnecessary_masking`** (`node_masking.py:99`):
- 移除 `_mask_to` 节点，当输入已有相同 mask 值时
- 保留馈入 reduction 的 mask（保持正确的零填充）

**`defer_pallas_load_masks`** (`node_masking.py:220`):
- 将 Pallas load 的越界 mask 推迟到下游 `_mask_to`
- 仅在 mask 在 major dim 且通过 permute 传递到 consumer 的 last-two dims 时生效
- 限制为纯轴置换 (`permute.default`)

---

## 3. 与 NineToothed T1-2-1 直接相关的技术

### 3.1 可整除 tile 的 mask 消除

**Helion 的实现** (`tile_strategy.py`):

```python
# mask_var(block_id) 在 tile 对齐时返回 None
def mask_var(self, block_idx):
    if loops := self.active_device_loops[block_idx]:
        return loops[-1].strategy.mask_var(block_idx)
    return None
```

`is_end_matching` 检查 tile 的 end 是否精确匹配 tensor 维度 extent。匹配时 → 无 mask。

**对 NineToothed 的启示**:

当前我们的 `_generate_offsets_and_mask` 在源层保留 `pid*BLOCK + arange < N`。当 `N % BLOCK == 0` 时，这个条件永远为真，可以完全省略。Helion 通过 `is_end_matching` 做这个判断。

**实现思路**: 在 `_generate_overall_offsets_and_mask` 中，对每个源层维度，检查 `size % tile_stride == 0`。如果整除，跳过源层 upper bound 的生成。

### 3.2 Size-1 维度的 stride 跳过

**Helion 的实现** (`indexing_strategy.py:1763`):

```python
for i, idx in enumerate(per_dim.dim_index_exprs):
    if not _is_size_one(fake_value.size(i)):
        stride = state.device_function.tensor_stride(fake_value, i).name
        index_expr.append(f"{idx} * {stride}")
    # size-1 维度：不生成 index * stride，贡献 0 偏移
```

**对 NineToothed 的启示**:

当前 `_generate_overall_offsets_and_mask` 对所有维度生成 `offset[dim] * stride[dim]`：

```python
overall_offsets = sum(
    offsets[source_dim] * Symbol(tensor.source.stride_string(source_dim))
    for source_dim in range(tensor.source.ndim)
)
```

对于广播维度（stride=0），`_BinOpSimplifier` 不能做 `x * 0 → 0`（因为会破坏 Triton 的 block 类型）。但如果在生成时就跳过 stride=0 的维度，指针表达式会更简洁，且不会丢失 block 类型信息。

### 3.3 Contiguous 访问的 `tl.max_contiguous` hint

**Helion 的实现** (`indexing_strategy.py:245`):

```python
# 评估 gather index 的连续运行长度
offset = self._eval(index_node)
k = self._max_run(offset.to(torch.int64))
# 当 2 <= k < run_extent 且 k * elem_size in {4, 8, 16} 时，发出 hint
contiguity[-1] = k

# 包装为 tl.max_contiguous
offset_ast = expr_from_string(
    f"tl.max_contiguous({{off}}, {contiguity!r})", off=inner_ast
)
```

**对 NineToothed 的启示**:

`tl.max_contiguous` 是 Triton 内置的编译器 hint，告诉后端 pointer 中有连续 k 个元素可用。这能帮助 Triton 生成向量化更好的内存访问代码。实现成本极低，只需在 pointer 表达式外包一层 `tl.max_contiguous(ptr + offset, BLOCK_SIZE)`。

### 3.4 TMA Fast Path (CuTe 后端)

**Helion 的实现** (`cute_mma.py:2263`):

`tcgen05_static_full_tma_fast_path`：当 matmul 的 operand 完全通过 TMA 加载时，跳过每轮迭代的边界检查。

**对 NineToothed 的启示**: AOT 场景下，当已知所有维度都对齐时，可以生成完全无 mask 的 kernel variant。这与我们的 AOT 变体特化方向一致。

---

## 4. 关键代码片段摘录

### 4.1 Load 生成 (`indexing_strategy.py:575`)

```python
def codegen_load(self, state, name, fake_tensor, subscript, extra_mask=None):
    indexing = SubscriptIndexing.create(state, fake_tensor, subscript, extra_mask)
    extra = ", other=0" if indexing.has_mask() else ""
    # tl.load(ptr + offset, mask, other=0) 或 tl.load(ptr + offset)
    return f"tl.load({name} + {{offset}}, {{mask}}{extra})"
```

### 4.2 Per-dim 索引构造 (`indexing_strategy.py:1432`)

```python
@staticmethod
def compute_per_dim_indexing(state, dyn_dims, block_shape, fake_tensor,
                              subscript, extra_mask, mask_override):
    per_dim_masks = []
    per_dim_index_exprs = []
    per_dim_broadcast_dims = []

    for i, dim in enumerate(dyn_dims):
        mask_var = state.codegen.mask_var(block_id)
        if mask_var is not None:
            per_dim_masks.append(mask_var)
        # ... 计算 index_expr

    mask_expr = " & ".join(per_dim_masks) if per_dim_masks else None
    return PerDimIndexing(per_dim_index_exprs, mask_expr, ...)
```

### 4.3 Mask 去重 (`node_masking.py:99`)

```python
def remove_unnecessary_masking(graph):
    for node in graph.nodes:
        if node.target == _mask_to:
            input_mask = _get_mask(node.args[0])
            self_mask = node.args[2]
            if input_mask == self_mask:
                # 替换所有使用为 input，移除冗余 _mask_to
                node.replace_all_uses_with(node.args[0])
```

---

## 5. 对 NineToothed 的优化建议

基于 Helion 的参考，按 ROI 排序：

| 优先级 | 优化 | 预期收益 | 参考 Helion 位置 | 实现位置 |
|--------|------|---------|-----------------|---------|
| **P0** | 可整除 tile 时消除源层 mask | 最高 mask reduction 收益 | `tile_strategy.py` `mask_var` | `generation.py` `_generate_offsets_and_mask` |
| **P1** | Size-1 维度跳过 stride 乘法 | 减少 stride 表达式 | `indexing_strategy.py:1763` | `generation.py` `_generate_overall_offsets_and_mask` |
| **P2** | `tl.max_contiguous` hint | 可能提升 runtime | `indexing_strategy.py:245` | `generation.py` `_generate_pointers_and_mask` |
| **P3** | AOT TMA-like 无 mask variant | AOT 变体命中 | `cute_mma.py:2263` | `aot.py` `_build_variant` |

---

## 6. 补充分析 — Helion 深层机制

### 6.1 编译管线 (kernel_compiler.py)

`KernelCompiler.compile()` 的六阶段管线：

```
parse → unroll → customize_ast → propagate_types → finalize_config → lower
```

- **parse**: `inspect.getsource()` → `ast.parse()` → `ExtendedAST` → `KernelDefinition`
- **unroll**: `unroll_static_loops()` 展开 Python 级 static `for` 循环
- **customize_ast**: 后端特定的 AST 改写（CuTe DSL 兼容性）
- **propagate_types**: 使用 `TypeInfo` 层次结构进行类型推断
- **finalize_config**: `CompileEnvironment.finalize_config_spec()` 锁定 autotuning 配置
- **lower**: `lower_to_device_ir()` 将 AST 追踪为 FX Graph IR，然后执行 mask 消除、reduction rolling、config fact 收集，最终 codegen

### 6.2 三种索引策略 (indexing_strategy.py)

`IndexingStrategy.select()` 根据 config 分发到三者之一：

| 策略 | 输出 | 约束 |
|------|------|------|
| `PointerIndexingStrategy` | `tl.load(base + offset, mask, other=0)` | 通用，无约束 |
| `BlockPtrIndexingStrategy` | `tl.load(tl.make_block_ptr(...), boundary_check=...)` | ndim ≥ 2, 间接 load 除外 |
| `TensorDescriptorIndexingStrategy` | `descriptor.load(offsets)` (TMA) | 2 ≤ ndim ≤ 5; 恰好一个 stride-1 维度; 所有非连续维度对齐; block_size ≤ dim_extent |

**回退链**: block_ptr 不支持 → pointer; tensor_descriptor 不支持 → pointer。

### 6.3 编译环境中的 known_multiple (compile_environment.py)

`CompileEnvironment.known_multiple(a, b)`:
```python
if isinstance(a, (int, sympy.Integer)) and isinstance(b, int):
    return (int(a) % b) == 0
return False
```

仅在两个参数都是具体整数时才返回 `True`。对于符号化大小，保守地返回 `False`。这一保守策略 NineToothed 已沿用。

### 6.4 Mask 消除的完整流程

1. **Tile 配准阶段** (`_setup_mask`): `known_multiple(block_size)` → 如可整除，`mask_var = None`
2. **逐维索引阶段** (`compute_per_dim_indexing`): `mask_var(block_id)` 返回 `None` → 该维度不生成 mask
3. **FX Graph 后处理阶段** (`remove_unnecessary_masking`): 移除冗余的 `_mask_to` 节点（保留馈入 reduction 的 mask）
4. **Pallas 特化** (`defer_pallas_load_masks`): 将 major-dim 的 eager load mask 推迟到 consumer 的 last-two dims

### 6.5 CuTe MMA 快速路径 (cute_mma.py)

`tcgen05_static_full_tma_fast_path` 标志，当以下全部成立时启用：

```python
tcgen05_static_full_tiles        # K dim 是 BK 的倍数
tcgen05_use_tma_pipeline         # TMA pipeline 激活
not tcgen05_is_two_cta           # 非 2-CTA 拆分
not tcgen05_use_role_local_mma_exec  # 非 per-role local exec
```

启用时 K 循环 TMA full-tile predicate 被完全跳过 —— 每一轮 K tile 都保证是满的。这类似于 NineToothed AOT 的 divisibility variant 概念。

### 6.6 Reduction Rolling (device_ir.py)

Reduction 循环可以 "rolled"（以小步长迭代）而不是完全展开。由 autotuner 选择，在 `build_codegen_graphs` 时应用到 graph copies。

### 6.7 Lane Loop 优化 (tile_strategy.py)

- **`split_lane_loop_reductions`**: 将 per-thread lane 循环重写为两遍结构（accumulate → finalize → consume），避免 per-lane 开销
- **`hoist_lane_invariant_chunk_recurrence`**: 检测 GDN/recurrence kernel 中的 `dot_acc` 模式，将 lane-invariant rescale 提升到 chunk 级别
- **`interchange_lane_outside_serial_reductions`**: 为 layer_norm backward 拆分 `for LANE: for MB:` 嵌套

---

## 7. v0.0.2 实现记录 — Helion 参考优化

### 7.1 P0: 可整除 tile 的 source-level mask 消除

**Helion 参考**: `NDTileStrategy._setup_mask()` (`tile_strategy.py:3531`) + `known_multiple()`

**NineToothed 实现** (`generation.py:_generate_offsets_and_mask`, lines 774-795):

```python
source_skip_upper = set()
if not has_unary_level:
    for source_dim in range(tensor.source.ndim):
        source_size_val = CodeGenerator._try_get_constant_int(
            tensor.source.shape[source_dim]
        )
        if source_size_val is None:
            continue
        for target_dim, tile_size in zip(innermost.target_dims, innermost.shape):
            if target_dim == source_dim:
                tile_size_val = CodeGenerator._try_get_constant_int(tile_size)
                if tile_size_val is not None and source_size_val % tile_size_val == 0:
                    source_skip_upper.add(source_dim)
                break
```

**启用条件**: 
- 无 unary level（无 unsqueeze/pad/slice 等）
- source 维度 size 是具体整数
- innermost tile 维度 size 是具体整数
- `source_size % tile_size == 0`

**验证结果**: `Tensor(ndim=1, shape=(1024,)).tile((128,))` → mask 仅含 `pid < 8`，source upper bound 消除正确。`Tensor(ndim=1, shape=(1000,)).tile((128,))` → mask 保留 source upper bound。

### 7.2 P1: Size-1 维度的 stride 跳过

**Helion 参考**: `SubscriptIndexing.create()` (`indexing_strategy.py:1763-1767`) + `_is_size_one()`

**NineToothed 实现**:

1. `_generate_overall_offsets_and_mask` (`generation.py:743-746`): 对 source 维度，当 `Symbol(shape[dim]) == 1` 或 offset 以 `0 * ...` 开头时，跳过 stride 项:
   ```python
   if CodeGenerator._is_effectively_zero_stride(tensor, source_dim, offsets):
       continue
   ```

2. `offsets()` (`tensor.py:565-586`): 支持 per-dim skip 标志; 当 `Symbol(size) == 1` 时完全跳过该维度的 mask 生成。

3. 新增 `_is_effectively_zero_stride()` 静态方法 (`generation.py:754-767`):
   ```python
   if Symbol(tensor.source.shape[dim]) == 1:       # 编译期已知的 size-1
       return True
   if offset node is (0 * expr):                    # broadcast/expand 产生的零偏移
       return True
   ```

### 7.3 P1: tl.max_contiguous hint

**Helion 参考**: `_PointerLoadContiguity.derive()` (`indexing_strategy.py:308`) + 包装模式 (`indexing_strategy.py:595-612`)

**NineToothed 实现** (`generation.py:_generate_pointers_and_mask`, lines 681-695):

```python
innermost = tensor.innermost()
if len(innermost.shape) == 1:
    tile_size_val = CodeGenerator._try_get_constant_int(innermost.shape[0])
    if tile_size_val is not None and tile_size_val > 1:
        pointers = call("max_contiguous", pointers, Symbol(f"[{tile_size_val}]"))
```

**启用条件**: innermost 层恰好 1 个维度，且 tile size 是 > 1 的具体整数。

**注意**: Helion 的 `_PointerLoadContiguity` 远比 NineToothed 的实现复杂，包括 allowlist/blocklist、`_max_run()` 求最大连续运行长度、以及 SWIZZLE 布局检测。NineToothed 当前仅覆盖简单连续 load 场景。

### 7.4 未采纳的优化: _BinOpSimplifier `0*x → 0`

未添加 `0 * x → 0` 到 `_BinOpSimplifier`，原因与原始报告一致：简化 `0 * index → 0` 会丢失 Triton block pointer 的类型信息。尺寸为 1 的维度在 codegen 阶段（而非清理阶段）通过跳过 stride 项来优化。

---

## 8. 引用

- Helion 仓库: https://github.com/pytorch/helion
- PLDI 2026 Tutorial: https://pldi26.sigplan.org/details/pldi-2026-tutorials/1/Writing-Performance-Portable-Kernels-Simplified-with-Helion
- Helion 文档: https://helionlang.com
- Triton: https://github.com/triton-lang/triton