# Ninetoothed 仓库代码检索索引

## 核心仓库结构

```
ninetoothed/
├── ninetoothed/
│   ├── __init__.py          # 公开 API：make, Symbol, Tensor, block_size
│   ├── language/
│   │   ├── __init__.py      # ntl 语言 API 导出
│   │   ├── core.py          # 核心操作 (load, store, cast, etc.)
│   │   └── math.py          # 数学操作 (sigmoid, tanh, exp, etc.)
│   ├── ir/
│   │   ├── __init__.py
│   │   ├── symbol.py        # Symbol 实现
│   │   ├── tensor.py        # Tensor 声明与 shape 推导
│   │   ├── arrangement.py   # Arrangement (tile/expand/squeeze)
│   │   └── application.py   # Application 计算定义
│   ├── codegen/
│   │   ├── __init__.py
│   │   ├── triton.py        # Triton codegen 入口
│   │   └── source.py        # 源码生成与检查
│   ├── autotune/
│   │   └── __init__.py      # 自动调优 (block_size meta 符号)
│   └── testing/
│       └── __init__.py      # 测试辅助
```

## 关键 API 定位

| 符号/API | 文件 | 行号 | 说明 |
|----------|------|------|------|
| `Symbol` | `ir/symbol.py` | — | 编译时常量符号 |
| `block_size` | `autotune/__init__.py` | — | 可 autotune 的 meta 符号 |
| `Tensor(n)` | `ir/tensor.py` | — | n 维张量声明 |
| `make()` | `__init__.py` | — | 构建 kernel 主入口 |
| `ntl.load` | `language/core.py` | — | 从指针加载数据 |
| `ntl.store` | `language/core.py` | — | 将数据写入指针 |
| `ntl.cast` | `language/core.py` | — | 类型转换 |
| `ntl.sigmoid` | `language/math.py` | — | sigmoid 激活 |
| `ntl.zeros` | `language/core.py` | — | 零初始化 |
| `tile()` | `ir/arrangement.py` | — | 数据分块 |
| `expand()` | `ir/arrangement.py` | — | 维度广播扩展 |
| `squeeze()` | `ir/arrangement.py` | — | 移除单维度 |

## Dot / Matmul 相关

| API | 文件 | 说明 |
|-----|------|------|
| `ntl.dot` | `language/core.py` | 矩阵乘法片段 |
| `ntl.softmax` | `language/math.py` | online softmax 原语 |

## 示例算子索引

| 算子 | 路径 | DSL 模式 |
|------|------|----------|
| RoPE | `ops/ninetoothed/kernels/rotary_position_embedding.py` | stride-dilation + 多分支 kernel |
| Scaled Dot-Product Attn | `ops/ninetoothed/kernels/scaled_dot_product_attention.py` | Flash Attention online softmax |
| 通用 ops | `ops/ninetoothed/torch.py` | Torch 包装层 |

## 搜索技巧

```bash
# 查找 ntl 所有导出函数
grep -rn "def " ninetoothed/language/ --include="*.py"

# 查找 make() 的使用
grep -rn "ninetoothed.make" examples/ --include="*.py"

# 查找 Symbol 的使用
grep -rn "Symbol(" examples/ --include="*.py"
```
