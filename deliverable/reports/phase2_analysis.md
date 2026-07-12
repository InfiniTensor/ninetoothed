# 阶段 2：AST 变换理解与冗余分析

### 1. AST 变换机制理解

NineToothed 的核心是 Python AST 变换。`CodeGenerator` 继承 `ast.NodeTransformer`，调用 `self.visit(tree)` 自动递归遍历整个 AST 树，每遇到一个节点就按类型名 dispatch 到对应的 `visit_*` 方法。

#### dispatch 机制

```
self.visit(tree)
  -> visit_Module(node)         # 匹配 Module
       -> generic_visit         # 递归遍历 body
         -> visit_FunctionDef   # 匹配 FunctionDef
              -> generic_visit  # 递归遍历 body
                -> visit_Assign # 匹配 Assign
                     -> generic_visit  # 递归遍历 value
                       -> (BinOp: 没有 visit_BinOp -> generic_visit)
                          -> visit_Name('lhs') -> _generate_load -> tl.load
                          -> visit_Name('rhs') -> _generate_load -> tl.load
                     -> return Expr(tl.store(...))
```

没有定义 `visit_*` 的节点类型走 `generic_visit` 继续递归。`CodeGenerator` 只定义它需要替换的方法：

| 方法 | 作用 |
|------|------|
| `visit_Module` | 末尾添加 autotune 包装 + launch 函数 |
| `visit_FunctionDef` | 重写参数列表、加 `@triton.jit` 装饰器 |
| `visit_Call` | 处理 tensor 方法调用（data_ptr/offsets/stride） |
| `visit_Subscript` | `lhs[i]` → `tl.load(...)` |
| `visit_Attribute` | `tensor.data_ptr` 等属性访问 |
| `visit_Name` | 变量名在右值 → `tl.load` |
| `visit_Assign` | `output = ...` → `tl.store(...)` |

#### _in_context 机制

```python
# generation.py:97
self._context = inspect.get_annotations(func)
# -> {'lhs': <Tensor>, 'rhs': <Tensor>, 'output': <Tensor>}

# generation.py:369
def _in_context(self, node):
    return isinstance(node, ast.Name) and node.id in self._context
```

`_in_context` 检查 AST 中的变量名是否在 `self._context` 字典（参数注解）中。是 → 九齿 tensor → 生成 `tl.load/store`。否 → 普通变量（如 `accumulator`、`k`）→ 原样保留。

### 2. 整体架构重新理解

```
用户 DSL（参数注解 = arrangement，函数体 = application）
    |
    v
@ninetoothed.jit (jit.py)
    | 薄壳：设定 num_warps/num_stages，调 CodeGenerator
    v
CodeGenerator.__call__ (generation.py:46)
    | 1. inspect.get_annotations(func) -> self._context (arrangement 结果)
    | 2. ast.parse(inspect.getsource(func)) -> AST tree
    | 3. self.visit(tree) -> AST 变换（关键步骤）
    | 4. Tritonizer().visit(tree) -> 加 import triton，替换 ninetoothed -> triton
    | 5. _BinOpSimplifier().visit(tree) -> x+0->x 等化简
    | 6. ast.unparse(tree) -> Triton 源码字符串
    | 7. cache_source(source) -> ~/.ninetoothed/<sha256>.py
    v
Triton 源码 -> importlib 加载 -> Triton JIT 编译 -> PTX/SASS
```

#### 辅助类职责

| 类 | 行 | 作用 |
|----|----|------|
| `Tritonizer` | 840 | 插入 `import triton`，替换 `ninetoothed.xxx` → `triton.xxx` |
| `_Inliner` | 884 | 内联用户 DSL 中调用的其他 Python 函数 |
| `_BinOpSimplifier` | 1117 | `x + 0 → x`、`x * 1 → x` 等算术化简 |
| `_SimplifiedNameCollector` | 1134 | prettify 模式：收集长变量名，生成短别名 |
| `cache_source` | 873 | SHA256 去重写入 `~/.ninetoothed/` |

### 3. Mask 构建流程追踪

通过 monkey-patch trace（`deliverable/reports/2.codegen_analysis.ipynb`）实际追踪了 vector add 的 mask 构建过程。

#### 调用链

```
_generate_load / _generate_store
  -> _generate_pointers_and_mask (652)
       -> _complete_indices: indices = (pid,) + () + (arange(0, BLOCK_SIZE),)
       -> _generate_overall_offsets_and_mask (725)
            -> _generate_offsets_and_mask (747)
                 -> 遍历 dtype 链，设置每层 _inputs
                 -> 逆序遍历 _levels，调用 Tensor.offsets()
                 -> 累加 offsets 到 source._outputs[0]
            -> overall_offsets = sum(offsets[dim] * stride[dim])
       -> pointers = pointer_name + overall_offsets
```

#### 三次 offsets() 调用

| 调用 | _inputs | _offsets 结果 | mask 追加 |
|------|---------|---------------|-----------|
| outer | `[[pid]]` | `(pid * BLOCK,)` | `pid < num_blocks`, `pid >= 0` |
| inner | `[[arange]]` | `(arange * 1,) = (arange,)` | `arange < BLOCK`, `arange >= 0` |
| source | `[[pid*BLOCK], [arange]]` | `(pid*BLOCK+arange,)` | `pid*BLOCK+arange < N`, `pid*BLOCK+arange >= 0` |

#### 冗余分析

实际生成的 mask：7 个子条件，6 个永真。

```
True                                                   ← 占位符，永真
& pid < ceil(N/BLOCK)                                  ← tail block 边界
& pid >= 0                                             ← 永真（无符号整数）
& arange < BLOCK                                       ← 永真（arange 范围就是 [0, BLOCK)）
& arange >= 0                                          ← 永真
& pid*BLOCK + arange < N                               ← 唯一有效条件
& pid*BLOCK + arange >= 0                              ← 永真
```

冗余根源：`Tensor.offsets()`（tensor.py:565-578）不加区分地对每层都追加 `index < size` 和 `index >= 0`，没有考虑 pid 和 arange 的天然取值范围。