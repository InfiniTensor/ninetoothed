# 阶段 1：项目分析与环境搭建

## 完成内容

### 1. 环境搭建
- 修复 Triton 运行环境：原 `.venv` 缺少 `torch`，导致 `triton.runtime.driver` 无法检测 GPU
- 安装 `nvcc`（CUDA 12.0）以支持 AOT 编译测试
- 确认 GPU 可用（RTX 4090D, compute capability 8.9, CUDA 12.8 driver）

### 2. 测试基线

**全量测试结果**（216 个，1271.70s ≈ 21min，主要耗时在 AOT 编译）：
- ✅ 213 passed
- ⏭️ 1 skipped（`test_aot.py::test_add[True]` — 多设备测试需 ≥2 GPU）
- ❌ 2 failed

| 失败测试 | 原因 | 分析 |
|---------|------|------|
| `test_aot.py::test_addmm` | fp16 精度超 `atol=0.075` | Triton 与 PyTorch 的 fp16 归约顺序不同导致浮点累积误差，**baseline 已有问题** |
| `test_ipynb.py::test_ipynb` | `jupyter nbconvert` 找不到 `ninetoothed` 模块 | Notebook 执行环境未安装 `.venv` 中的 ninetoothed，**环境问题非代码问题** |

**分项结果**：

| 测试文件 | 结果 | 耗时 |
|---------|------|------|
| `test_generation.py` | 76/76 ✅ | ~9s |
| `test_aot.py` | 10/12 ✅, 1 ❌, 1 ⏭️ | ~21min（几乎全部 AOT 编译时间） |
| `test_add.py` | 1 ✅ | ~3s |
| `test_addmm.py` | 2 ✅ | ~3s |
| 其余 12 个测试文件 | 全部 ✅ | 几十秒 |

**结论**：2 个失败均为环境/基线问题，非代码改动导致。`test_generation.py` 76 个测试全部通过，是本赛题修改的核心回归依据。

### 3. 项目结构理解

```
src/ninetoothed/
├── generation.py    # 核心：CodeGenerator AST 变换 → Triton 源码
├── aot.py           # AOT 编译：变体枚举、dispatcher 生成、nvcc 编译
├── tensor.py        # 符号 Tensor：tile/expand/squeeze/permute
├── symbol.py        # Symbol 包装：AST 节点 + 算术运算 + 范围约束
├── language.py      # ntl.dot, ntl.zeros 等语言层
├── naming.py        # 命名工具
├── jit.py           # JIT 编译入口
├── make.py          # make() 入口
├── build.py         # build() 多配置构建
├── dtype.py         # 数据类型
├── torchifier.py    # Torch 调用适配
├── cudaifier.py     # CUDA 调用适配
└── utils.py         # 工具函数
```

### 4. 代码生成执行链路

```
make(arrangement, application, tensors)
  → jit() 或 aot() 取决于 caller
  → CodeGenerator(func):
      1. 解析 func AST → _get_tree()
      2. 遍历 AST 节点，替换为 Triton 对应代码
      3. _generate_load/store() → tl.load/tl.store
      4. _generate_pointers_and_mask() → pointer 表达式 + mask
      5. _generate_offsets_and_mask() → 递归遍历 tensor 层级
      6. _generate_overall_offsets_and_mask() → sum(offset * stride)
      7. _generate_autotune() → autotune 包装（可选）
      8. _generate_launch() → launch function
  → 输出 Triton 源码到 ~/.ninetoothed/<sha256>.py
  → JIT: importlib 加载执行
  → AOT: nvcc 编译为 .so + C++ dispatcher
```

### 5. 理解 Tile 与 Arrangement

通过可视化 notebook（`deliverable/reports/0.tile_and_arrangement.ipynb`）逐步拆解了矩阵乘法的 arrangement 过程。

#### 层级规则

| 操作 | 增层级？ | 作用 |
|------|---------|------|
| `tile(size)` | **是** | 当前层切块，新增一层 outer，原层变为 inner |
| `expand(shape)` | 否 | size-1 维度广播复制，对齐 outer shape |
| `squeeze(dim)` | 否 | 去掉中间层的 size-1 维度 |

#### Arrangement 三步法（矩阵乘法 A×B=C 为例）

1. **第一次 tile** — 决定内层数据块大小（硬件调优）：
   ```
   A.tile((2,2)) → outer(2,3), inner(2,2)
   B.tile((2,2)) → outer(3,4), inner(2,2)
   C.tile((2,2)) → outer(2,4), inner(2,2)
   ```
   outer shape 互不相同，无法配对。

2. **第二次 tile** — 按 K 维打包，每个 program 拿到完整累加链，进入 **3 层**：
   ```
   A.tile((1,-1)) → outer(2,1): 每组 = A 的一整行 block
   B.tile((-1,1)) → outer(1,4): 每组 = B 的一整列 block
   ```
   `tile((1,-1))` 不是"切最小公约"，而是**把 K 维完整迭代范围打包给每个 program**。

3. **expand** — 显式广播对齐 outer shape：
   ```
   A: expand((-1, 4)) → outer(2,4)
   B: expand((2, -1)) → outer(2,4)
   ```
   编译器启动 2×4 = 8 个 program，每个在 `for k in range(3)` 中累加小矩阵乘。

### 6. 赛题理解

**目标**：在 `generation.py`/`aot.py` 中实现 1-2 类特化，减少冗余 mask/stride/pointer 表达式。

**特化类别**：
1. Contiguous fast path — 连续 tensor 时生成线性 pointer
2. Divisible tile fast path — 无尾块时省略 mask
3. Broadcast/scalar fast path — 简化广播/标量表达式
4. Layout-known AOT variant — 利用 AOT 约束选择更专门 variant

**评分要点**：
- Correctness 30 分（≥ 29/30 门槛）
- Specialization Coverage 20 分（命中率 - 误命中率）
- Generated Code Metric 20 分（mask/stride/pointer 缩减比例）
- Runtime 20 分（speedup）
- 工程与报告质量 10 分

**测试要求**：
- 现有 test_generation.py 76 个测试必须通过
- 新增 ≥ 2 hit + ≥ 2 fallback + ≥ 2 generated source 结构测试
- Benchmark 输出 JSON/CSV

### 7. 代码生成与特化本质理解

#### 7.1 架构分层

```
用户 DSL（tensor 层面）:
    output = lhs + rhs                     ← 描述"做什么"
         ↓
CodeGenerator（generation.py）:
    AST 变换，自动推导 pointer/mask/offset  ← 翻译成"怎么做"
         ↓
Triton 代码（~/.ninetoothed/<sha256>.py）:
    tl.load/tl.store/tl.add 等              ← 标准 Triton JIT
         ↓
Triton JIT 编译 → PTX → SASS               ← 底层工具链
```

关键文件职责：
- **`generation.py`**：`CodeGenerator` AST 变换器，核心。把 Python AST 节点（`BinOp`、`Assign` 等）替换为 Triton 等价节点。`_generate_load/store`、`_generate_offsets_and_mask`、`_generate_pointers_and_mask` 在此。
- **`jit.py`**：薄壳。只做两件事：(1) 调 `CodeGenerator` 生成源码，(2) `importlib` 加载该文件。不参与代码生成。
- **`aot.py`**：变体枚举 + C++ dispatcher 生成，供 AOT 编译。

#### 7.2 生成的 Triton 代码示例（vector add）

以 `Tensor(1).tile((BLOCK_SIZE,))` 为例，用户写：

```python
def add_kernel(lhs, rhs, output):
    output = lhs + rhs
```

NineToothed 生成（简化后）：

```python
@triton.jit
def add_kernel(lhs_ptr, lhs_size, lhs_stride,
               rhs_ptr, rhs_size, rhs_stride,
               out_ptr, out_size, out_stride,
               BLOCK_SIZE: constexpr):
    pid = tl.program_id(0)
    mask = (pid < (lhs_size - (BLOCK_SIZE-1) - 1 + BLOCK_SIZE-1) // BLOCK_SIZE + 1)  # ①
         & (pid >= 0)                                                                 # ② 永真
         & (tl.arange(0, BLOCK_SIZE) < BLOCK_SIZE)                                    # ③ 永真
         & (tl.arange(0, BLOCK_SIZE) >= 0)                                            # ④ 永真
         & (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) < lhs_size)                   # ⑤
         & (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) >= 0)                         # ⑥ 永真
    lhs = tl.load(lhs_ptr + (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) * lhs_stride, mask=mask)
    rhs = tl.load(rhs_ptr + (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) * rhs_stride, mask=mask)
    tl.store(out_ptr + (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)) * out_stride, lhs + rhs, mask=mask)
```

其中②③④⑥永远为真，①和⑤在 `lhs_size % BLOCK_SIZE == 0` 时也永远为真。全是冗余。

#### 7.3 赛题本质：编译期特化 = 代码模板剪枝

赛题就是在代码生成阶段加剪枝条件：

```
if 条件是"编译期可判定为永真":
    跳过生成该 mask 子句

if 所有 mask 子句都被跳过:
    生成 tl.load(ptr) 不带 mask 参数
```

类比传统编译器优化：

| 特化操作 | 传统编译器对应 | 剪枝内容 |
|---------|--------------|---------|
| Contiguous fast path | strength reduction | `ptr + offset * 1` → `ptr + offset` |
| Divisible tile fast path | partial evaluation | 无 tail block → 不生成 mask |
| Broadcast/scalar fast path | copy propagation | size-1 维 → 跳过 offset 展开 |
| Layout-known AOT variant | partial evaluation | AOT 已知连续 → 生成线性访存 |

**不是运行时优化**，是**生成阶段的条件跳过**，跟传统编译器的"死代码消除后处理"相反——这里根本不生成冗余代码。

#### 7.4 测试验证方法

生成的代码质量通过三类可见测试验证：
1. **Specialization hit**（≥ 2）— 验证特化路径被触发（检查生成代码中无 mask/mask 更简洁）
2. **Fallback correctness**（≥ 2）— 验证不满足条件时正确回退通用路径
3. **Generated source 结构**（≥ 2）— 直接断言生成源码中的 mask/stride 表达式结构

## 下一步计划

1. 深入阅读 `generation.py` 的 `_generate_offsets_and_mask` 和 `_generate_pointers_and_mask` 实现
2. 追踪代码生成流程：以 vector add 为例，定位哪段代码生成上述 6 个 mask 条件
3. 确定选 1-2 类特化方向，设计启用条件与剪枝逻辑
4. 在 `generation.py`/`aot.py` 中实现特化
5. 编写 6 个可见测试 + benchmark