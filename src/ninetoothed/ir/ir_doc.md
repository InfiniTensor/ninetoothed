# NineToothed IR 系统文档

## 概述

NineToothed 使用多层 IR (Intermediate Representation) 管线将高层张量操作逐步转换为 Triton GPU kernel 代码。管线共分 5 层：

```
L0 (Python AST) → L1 (Tensor+Tiling IR) → L2 (Memory IR) → L3 (Triton IR) → L4 (Code)
```

每层 IR 节点都继承自 `IRNode` 基类，提供 S-expression 风格的 dump 输出和内联渲染 (`_to_inline()`)。

## 目录

- [L0: Python AST](#l0-python-ast)
- [L1: Tensor + Tiling IR](#l1-tensor--tiling-ir)
- [L2: Memory IR](#l2-memory-ir)
- [L3: Triton IR](#l3-triton-ir)
- [L4: Generated Code](#l4-generated-code)
- [Pipeline 编排](#pipeline-编排)
- [IRNode 基类](#irnode-基类)

---

## L0: Python AST

**对应文件**: `src/ninetoothed/ir/pipeline.py` 中的 `_get_tree()` 方法

L0 是管线的入口，不是自定义 IR，而是 Python 标准库的 `ast.Module`。`IRPipeline._get_tree()` 负责：

1. 通过 `inspect.getsource()` 获取用户函数的源码
2. 使用 `ast.parse()` 解析为 AST
3. 通过 `_Inliner` 内联所有辅助函数调用
4. 如使用了 `libdevice`，自动添加 `from triton.language.extra import libdevice` 导入

```python
# 用户输入
def application(input: Tensor(1), other: Tensor(1), output: Tensor(1)):
    output = input + other
```

被解析为包含 `FunctionDef` 的 `ast.Module`，其中参数的类型注解携带 `Tensor` 对象及其 tiling 信息。

---

## L1: Tensor + Tiling IR

**对应文件**: `src/ninetoothed/ir/tensor_ir.py`, `src/ninetoothed/ir/passes/ast_to_l1.py`

**转换 Pass**: `ASTToL1Pass`

L1 是第一层自定义 IR，保留张量语义和 tiling 信息。核心思想是**在 AST 层面捕获张量类型信息**，为后续的内存寻址计算做准备。

### 节点类型

#### 元数据

| 节点 | 字段 | 说明 |
|------|------|------|
| `TileOp` | `kind`, `args`, `kwargs` | 记录张量的 tiling 操作历史，如 `tile((BLOCK_SIZE,))` |

#### 表达式 (L1Expr)

| 节点 | 字段 | 说明 |
|------|------|------|
| `L1TensorAccess` | `param_name`, `tensor`, `indices` | 张量读写访问。`tensor` 保留原始 `Tensor` 对象，用于后续提取 shape/stride 信息 |
| `L1BinOp` | `op`, `lhs`, `rhs` | 二元运算: `+`, `-`, `*`, `//`, `%` 等 |
| `L1UnaryOp` | `op`, `operand` | 一元运算: `-`, `~`, `not` |
| `L1Compare` | `op`, `left`, `right` | 比较: `<`, `>=`, `==` 等 |
| `L1BoolOp` | `op`, `values` | 布尔运算: `and`, `or` |
| `L1Call` | `func`, `args`, `kwargs` | 函数调用 |
| `L1Name` | `name` | 变量引用 |
| `L1Constant` | `value` | 字面量 |
| `L1Attribute` | `obj`, `attr` | 属性访问: `obj.attr` |
| `L1Subscript` | `value`, `slice` | 下标访问: `value[slice]` |
| `L1Tuple` | `elts` | 元组 |
| `L1IfExp` | `test`, `body`, `orelse` | 三元表达式 |

#### 张量专用节点

| 节点 | 字段 | 说明 |
|------|------|------|
| `L1DataPtr` | `tensor` | `tensor.data_ptr()` 调用，获取数据指针 |
| `L1Offsets` | `tensor`, `dim` | `tensor.offsets(dim)` 调用，获取偏移量 |
| `L1Stride` | `tensor`, `dim` | `tensor.stride(dim)` 调用，获取步长 |
| `L1DtypeAttr` | `tensor` | `tensor.dtype` 属性访问 |

#### 语句 (L1Statement)

| 节点 | 字段 | 说明 |
|------|------|------|
| `L1Assign` | `target`, `value` | 赋值: `output = input + other` |
| `L1ExprStmt` | `expr` | 表达式语句 |
| `L1Return` | `value` | 返回语句 |
| `L1For` | `target`, `iter`, `body` | for 循环 |
| `L1If` | `test`, `body`, `orelse` | if 语句 |

#### 函数与参数

| 节点 | 字段 | 说明 |
|------|------|------|
| `L1TensorParam` | `name`, `tensor`, `tile_history`, `ndim`, `dtype`, `other`, `jagged_dim` | 携带张量元数据的函数参数 |
| `L1Function` | `name`, `params`, `body`, `invariants` | 顶层 kernel 函数 |

### ASTToL1Pass 转换逻辑

1. 遍历 `ast.FunctionDef`，从类型注解中提取 `Tensor` 对象
2. 为每个参数构建 `L1TensorParam`，从 `tensor._history` 提取 `TileOp` 列表
3. 遍历函数体中的语句，将 AST 节点映射为 L1 IR 节点
4. 识别张量访问（`ast.Name` 或 `ast.Subscript` 匹配参数名），生成 `L1TensorAccess`
5. 识别张量方法调用（`.data_ptr()`, `.offsets()`, `.stride()`），生成专用节点

### Dump 示例

```
(L1Function
  name='application'
  params=
    [0]=(L1TensorParam
      name='input'
      ndim=1
      tile_history=
        [0]=(TileOp kind='tile' args=((BLOCK_SIZE,),))
      shape=(...)
      innermost_shape=(BLOCK_SIZE)
    )
    ...
  body=
    [0]=(L1Assign
      target=(L1TensorAccess param_name='output' ...)
      value=(L1BinOp op='+'
        lhs=(L1TensorAccess param_name='input' ...)
        rhs=(L1TensorAccess param_name='other' ...)
      )
    )
)
```

---

## L2: Memory IR

**对应文件**: `src/ninetoothed/ir/memory_ir.py`, `src/ninetoothed/ir/passes/l1_to_l2.py`

**转换 Pass**: `L1ToL2Pass`

L2 是管线的核心层，将高层张量操作**降级为显式的内存操作**：指针算术、load/store、边界 mask。这是最复杂的转换。

### 关键变化 (L1 → L2)

| L1 概念 | L2 概念 |
|---------|---------|
| `L1TensorAccess` (读) | `L2Load(pointer, mask, other)` |
| `L1TensorAccess` (写) | `L2Store(pointer, value, mask)` |
| 张量参数 | 展开为 `size_0`, `stride_0`, `pointer` 等 `L2Param` |
| 无 | `L2Invariant` (预计算的指针、索引) |
| 无 | 边界 mask 表达式 |

### 节点类型

#### 表达式 (L2Expr)

| 节点 | 字段 | 说明 |
|------|------|------|
| `L2PointerExpr` | `base`, `offsets` | 内存指针表达式: `base + offset_0 + offset_1 + ...` |
| `L2OffsetTerm` | `stride`, `index` | 偏移项: `stride * index` 或仅 `index` |
| `L2MaskExpr` | `conditions` | mask 表达式: `cond_0 & cond_1 & ...` |
| `L2BinOp` | `op`, `lhs`, `rhs` | 二元运算，支持 `&` 链展平 |
| `L2Compare` | `op`, `left`, `right` | 比较 |
| `L2BoolOp` | `op`, `values` | 布尔运算 |
| `L2Call` | `func`, `args`, `kwargs` | 函数调用 (如 `ninetoothed.language.load(...)`) |
| `L2Name` | `name` | 变量引用 |
| `L2Constant` | `value` | 字面量 |
| `L2Subscript` | `value`, `slice` | 下标 |
| `L2Tuple` | `elts` | 元组 |

#### 语句 (L2Statement)

| 节点 | 字段 | 说明 |
|------|------|------|
| `L2Load` | `pointer`, `mask`, `other` | 从显式指针地址加载数据，`mask` 控制越界 |
| `L2Store` | `pointer`, `value`, `mask` | 向显式指针地址存储数据 |
| `L2Assign` | `target`, `value` | 赋值 |
| `L2ExprStmt` | `expr` | 表达式语句 |
| `L2For` | `target`, `iter`, `body` | for 循环 |
| `L2If` | `test`, `body`, `orelse` | if 语句 |

#### 函数与参数

| 节点 | 字段 | 说明 |
|------|------|------|
| `L2Param` | `name`, `is_constexpr`, `annotation` | 函数参数（展开后的标量参数） |
| `L2Invariant` | `target`, `value` | 预计算的常量表达式（如 pid、指针、load 结果） |
| `L2Function` | `name`, `params`, `body`, `invariants`, `grid_expr` | 顶层函数 |

### L1ToL2Pass 转换逻辑

L1ToL2Pass 是最复杂的 pass，核心流程如下：

1. **参数展开**: 将每个 `L1TensorParam` 的 `Tensor` 对象展开为 `L2Param` 列表（`size_0`, `stride_0`, `pointer` 等）
2. **张量访问转换**:
   - `L1TensorAccess` (非标量) → 生成 `load` 调用存入 invariant，返回 `L2Name` 引用
   - `L1TensorAccess` (标量, `ndim=0`) → 直接返回 `L2Name`
3. **赋值转换**: `L1Assign(target=L1TensorAccess, value=...)` → `L2Store(pointer, value, mask)`
4. **指针计算**: 委托 `CodeGenerator` 静态方法：
   - `_generate_pid_indices()` → 通过 `Tensor._unravel_index()` 将 `program_id(0)` 展开为多维索引
   - `_generate_innermost_indices()` → 生成 `arange(0, BLOCK_SIZE)` 等内层索引
   - `_generate_overall_offsets_and_mask()` → 计算完整的偏移量和边界 mask
5. **Symbol → L2 转换**: 将 `Symbol` 对象内部的 AST 节点转换为 `L2BinOp`, `L2Call` 等 L2 IR 节点

### Dump 示例

```
(L2Function
  name='application'
  params=[tensor_0_size_0, BLOCK_SIZE, tensor_0_stride_0, tensor_0_pointer, ...]
  invariants=
    pid = ninetoothed.language.program_id(0)
    tensor_0_index_0 = pid
    tensor_0_pointers = tensor_0_pointer
    _load_91744 = ninetoothed.language.load((pointers + offsets), mask=True & index < bound & ...)
  body=
    [0]=store((tensor_2_pointers + offsets), (_load_91744 + _load_91936), mask=...)
)
```

---

## L3: Triton IR

**对应文件**: `src/ninetoothed/ir/triton_ir.py`, `src/ninetoothed/ir/passes/l2_to_l3.py`

**转换 Pass**: `L2ToL3Pass`

L3 在 L2 基础上做**命名空间替换**和**专用节点转换**，将 ninetoothed 内部表示映射为 Triton API。

### 关键变化 (L2 → L3)

| L2 概念 | L3 概念 |
|---------|---------|
| `ninetoothed.language.program_id(0)` | `L3ProgramId(axis=0)` |
| `ninetoothed.language.arange(0, N)` | `L3Arange(start=0, end=N)` |
| 名称中的 `ninetoothed` 前缀 | 替换为 `triton` 前缀 |
| `L2PointerExpr` | 展平为 `L3BinOp` 加法链 |
| `L2MaskExpr` | 展平为 `L3BinOp(op='&')` 链 |

### 节点类型

#### 表达式 (L3Expr)

| 节点 | 字段 | 说明 |
|------|------|------|
| `L3ProgramId` | `axis` | `program_id(axis)` 调用的专用表示 |
| `L3Arange` | `start`, `end` | `arange(start, end)` 调用的专用表示 |
| `L3BinOp` | `op`, `lhs`, `rhs` | 二元运算，支持 `&` 链展平 |
| `L3Compare` | `op`, `left`, `right` | 比较 |
| `L3BoolOp` | `op`, `values` | 布尔运算 |
| `L3Call` | `func`, `args`, `kwargs` | 函数调用 |
| `L3Name` | `name` | 变量引用（名称已替换为 triton 前缀） |
| `L3Constant` | `value` | 字面量 |
| `L3Subscript` | `value`, `slice` | 下标 |
| `L3Tuple` | `elts` | 元组 |

#### 语句 (L3Statement)

与 L2 语句结构相同，仅类型不同：
`L3Load`, `L3Store`, `L3Assign`, `L3ExprStmt`, `L3Return`, `L3For`, `L3If`

#### 函数与参数

| 节点 | 字段 | 说明 |
|------|------|------|
| `L3Param` | `name`, `is_constexpr`, `annotation` | 函数参数 |
| `L3Invariant` | `target`, `value` | 预计算表达式 |
| `L3Grid` | `expr` | Grid 配置表达式 |
| `L3Autotune` | `configs`, `key` | Autotune 配置 |
| `L3Function` | `name`, `params`, `body`, `invariants`, `grid`, `autotune` | 顶层 Triton kernel 函数 |

### L2ToL3Pass 转换逻辑

1. **命名空间替换**: 所有名称中的 `ninetoothed` → `triton`
2. **专用节点转换**:
   - `L2Call(func='program_id', args=[0])` → `L3ProgramId(axis=0)`
   - `L2Call(func='arange', args=[0, N])` → `L3Arange(start=0, end=N)`
3. **指针表达式展平**: `L2PointerExpr(base, offsets)` → `base + offset_0 * stride_0 + ...`
4. **mask 表达式展平**: `L2MaskExpr(conditions)` → `cond_0 & cond_1 & ...`
5. 参数、invariant、语句逐一转换

### Dump 示例

```
(L3Function
  name='application'
  params=[tensor_0_pointer, BLOCK_SIZE, tensor_0_stride_0, ...]
  invariants=
    pid = triton.language.program_id(0)
    _load_91744 = triton.language.load((pointers + offsets), mask=True & index < bound & ...)
  body=
    [0]=store((tensor_2_pointers + offsets), (_load_91744 + _load_91936), mask=...)
)
```

---

## L4: Generated Code

**对应文件**: `src/ninetoothed/ir/passes/l3_to_code.py`

**转换 Pass**: `L3ToCodePass`

L4 是管线的最终输出——可执行的 Triton Python 源代码。

### L3ToCodePass 转换逻辑

1. **生成导入**: `import triton`, `import triton.language`
2. **生成装饰器**: `@triton.jit` 或 `@triton.autotune(configs=[...], key=[...])`
3. **生成函数签名**: `def kernel_name(param_0, param_1, ...):`
4. **生成 invariant**: 函数体开头的预计算赋值语句
5. **生成 body**: 将 L3 语句格式化为 Python 代码
   - `L3Assign` → `target = value`
   - `L3Store` → `triton.language.store(pointer, value, mask=mask)`
   - `L3For` → `for target in iter:`
   - `L3If` → `if test:`
6. **可选美化**: 使用 `ruff format` 格式化输出

### 输出示例

```python
import triton
import triton.language

@triton.jit

def application(tensor_0_size_0, BLOCK_SIZE, tensor_0_stride_0, tensor_0_pointer, ...):
    pid = triton.language.program_id(0)
    _load_91744 = triton.language.load((...), mask=(...))
    triton.language.store((...), (_load_91744 + _load_91936), mask=(...))
```

---

## Pipeline 编排

**对应文件**: `src/ninetoothed/ir/pipeline.py`

`IRPipeline` 类负责协调整个转换管线。

### 初始化参数

| 参数 | 说明 |
|------|------|
| `context` | 函数参数名到 `Tensor` 类型的映射 |
| `args` | 函数参数类型列表 |
| `caller` | 调用者 (`"torch"` 等) |
| `kernel_name` | 生成的 kernel 名称 |
| `num_warps` | warp 数量配置 |
| `num_stages` | 流水线阶段数 |
| `max_num_configs` | 最大 autotune 配置数 |
| `prettify` | 是否美化生成的代码 |
| `dump_ir` | 是否在每层转换后打印 IR dump |

### 执行流程

```python
pipeline = IRPipeline(context, args, caller, kernel_name, ...)
pipeline.run(func)
```

`run()` 方法依次执行：

1. `_get_tree(func)` — L0: 解析源码为 AST，内联辅助函数
2. `ASTToL1Pass(context).transform(tree)` — L0 → L1
3. `L1ToL2Pass().transform(l1_func)` — L1 → L2
4. `L2ToL3Pass().transform(l2_func)` — L2 → L3
5. `L3ToCodePass(prettify).transform(l3_func)` — L3 → L4
6. `cache_source(source)` — 缓存生成的源码文件

支持 `stop_after` 参数在任意层提前终止（`"l1"`, `"l2"`, `"l3"`），便于调试和测试。

### Dump 机制

当 `dump_ir=True` 时，每层转换后自动打印 IR dump：

- L1/L2/L3 使用 S-expression 格式的 `dump()` 方法
- 表达式节点通过 `_to_inline()` 生成紧凑的单行表示
- mask 的 `&` 链自动展平为 `cond1 & cond2 & ...` 格式
- 冗长的内部名称通过 `_shorten_name()` 缩短（如 `ninetoothed_ninetoothed_tensor_0_size_0` → `tensor_0_size_0`）

---

## IRNode 基类

**对应文件**: `src/ninetoothed/ir/base.py`

所有 IR 节点继承自 `IRNode`，提供以下功能：

### 方法

| 方法 | 说明 |
|------|------|
| `dump(indent=0)` | 生成 S-expression 风格的多行文本输出 |
| `_to_inline()` | 生成紧凑的单行字符串表示（子类应覆盖） |
| `_shorten_name(name)` | 静态方法，缩短冗长的内部名称 |
| `_dump_body(indent)` | 覆盖以自定义 dump 输出 |
| `_iter_fields()` | 遍历所有非私有属性 |
| `__eq__()` | 基于 `_iter_fields()` 的相等比较 |
| `__repr__()` | 返回 `dump()` 结果 |
| `__hash__()` | 返回对象 ID |

### 名称缩短规则

`_shorten_name()` 对以下前缀进行缩短：

| 原始前缀 | 缩短后 |
|----------|--------|
| `ninetoothed_ninetoothed_` | (移除) |
| `ninetoothed_constexpr_prefix_` | (移除) |
| `triton_constexpr_prefix_` | (移除) |
| `triton_triton_` | (移除) |
| `ninetoothed_tensor_` | (移除) |
| `triton_tensor_` | (移除) |
| `ninetoothed_pid` | `pid` |
| `triton_pid` | `pid` |

---

## 各层对比总结

| 层级 | 核心关注点 | 输入 | 输出 |
|------|-----------|------|------|
| L0 | 源码解析 | Python 函数 | `ast.Module` |
| L1 | 张量语义 + Tiling | `ast.Module` | `L1Function` (含 Tensor 对象) |
| L2 | 内存操作 | `L1Function` | `L2Function` (含指针/mask) |
| L3 | Triton API 映射 | `L2Function` | `L3Function` (ninetoothed→triton) |
| L4 | 可执行代码 | `L3Function` | Python 源码字符串 |
