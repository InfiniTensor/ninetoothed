# T1-2-1 Weakness Analysis

**选择的特化类别**：Contiguous Fast Path + Divisible Tile Fast Path  
**分析对象**：NineToothed 指定 baseline  
**数据来源**：`benchmarks/bench_aot_speedup_compare_results.json`  
**分析日期**：2026-07-12

---

## 1. 分析目标

本分析识别 baseline 在可可靠判定连续布局和 tile 整除信息时的弱势生成代码，重点对应官方允许的低效类型：

- 冗余 mask；
- 冗余 stride/offset；
- 冗余 pointer arithmetic；
- 可判定布局没有命中更专门的 variant。

分析不以特定 benchmark 名称或固定输入尺寸作为生产代码的命中条件。报告中的 shape 仅用于复现和量化。

---

## 2. Weakness Case 1：连续且整除的 1D 场景

### 2.1 场景

- rank：1D；
- 三个 tensor shape 相同；
- 输入与输出均 contiguous；
- tile 能完整覆盖迭代范围；
- 数据类型：BF16。

### 2.2 Baseline 弱势

baseline 即使已经能够判定 contiguous 和 divisible，生成代码仍保留：

- 6 个 mask 相关表达式；
- 9 个 stride 相关表达式；
- 19 个 pointer arithmetic 相关表达式；
- 9 个 kernel 参数，其中 3 个为 stride 参数；
- 3449 bytes 生成源码。

这属于：

- **冗余 mask**：无尾块时仍保留边界 mask；
- **冗余 stride/offset**：连续布局仍通过通用 stride 参数计算地址；
- **冗余 pointer arithmetic**：线性连续访问仍保留多余的通用地址展开；
- **可判定布局未命中特化 variant**。

### 2.3 提交后结果

| 指标 | baseline | submitted | 改善 |
|---|---:|---:|---:|
| mask expressions | 6 | 0 | 100% |
| stride expressions | 9 | 0 | 100% |
| pointer expressions | 19 | 10 | 47.4% |
| source bytes | 3449 | 1577 | 54.3% |
| kernel params | 9 | 4 | 55.6% |
| stride params | 3 | 0 | 100% |

实际 dispatcher：

```text
expected = flatten_contiguous_divisible
actual   = flatten_contiguous_divisible
dispatch_match = true
correct = true
```

该 case 证明 Category 1 与 Category 2 可以组合：连续布局用于简化地址，整除条件用于安全删除边界 mask。

---

## 3. Weakness Case 2：连续且整除的 2D 场景

### 3.1 场景

- rank：2D；
- 输入与输出均 contiguous；
- shape 与 tile 满足无尾块条件；
- 数据类型：BF16。

### 3.2 Baseline 弱势

2D 场景中，baseline 的多维通用地址展开更加明显：

- mask expressions：6；
- stride expressions：18；
- pointer expressions：31；
- kernel params：15；
- stride params：6；
- source bytes：6421。

相比 1D，stride 和 pointer 表达式随维数进一步增长。已知 contiguous 时继续传递和展开全部 stride，会放大生成代码复杂度。

### 3.3 提交后结果

| 指标 | baseline | submitted | 改善 |
|---|---:|---:|---:|
| mask expressions | 6 | 0 | 100% |
| stride expressions | 18 | 0 | 100% |
| pointer expressions | 31 | 16 | 48.4% |
| source bytes | 6421 | 2886 | 55.1% |
| kernel params | 15 | 7 | 53.3% |
| stride params | 6 | 0 | 100% |

实际 dispatcher：

```text
expected = flatten_contiguous_divisible
actual   = flatten_contiguous_divisible
dispatch_match = true
correct = true
```

2D square profile 的 runtime speedup 为 1.0110x，说明该路径在部分 2D 场景中不仅改善源码结构，也可获得小幅运行收益。

---

## 4. Weakness Case 3：连续但存在尾块的 masked 场景

### 4.1 问题性质

masked 场景不能删除边界 mask，否则会发生越界访问。因此这里的 weakness 不是“mask 必须全部删除”，而是：

> 边界 mask 必须保留，但连续布局已经可判定，baseline 仍保留全部 stride metadata 和通用地址计算。

### 4.2 1D masked 数据

| 指标 | baseline | submitted | 改善 |
|---|---:|---:|---:|
| mask expressions | 6 | 6 | 正确保留 |
| stride expressions | 9 | 0 | 100% |
| pointer expressions | 19 | 16 | 15.8% |
| source bytes | 3498 | 3108 | 11.1% |
| kernel params | 9 | 6 | 33.3% |
| stride params | 3 | 0 | 100% |

### 4.3 2D masked 数据

| 指标 | baseline | submitted | 改善 |
|---|---:|---:|---:|
| mask expressions | 6 | 6 | 正确保留 |
| stride expressions | 18 | 0 | 100% |
| pointer expressions | 31 | 28 | 9.7% |
| source bytes | 6516 | 5871 | 9.9% |
| kernel params | 15 | 9 | 40.0% |
| stride params | 6 | 0 | 100% |

实际 dispatcher 能分别命中：

```text
flatten_contiguous_masked
```

并保持 `correct=true`、`max_diff=0.0`。这证明 contiguous fast path 不需要以错误删除 mask 为代价。

---

## 5. Weakness Case 4：生成参数未按活跃性裁剪

### 5.1 Baseline 行为

baseline 采用通用 kernel signature，为每个 tensor 保留 shape/stride metadata。连续特化后，其中部分参数已经不再被 kernel body 使用，但仍出现在 signature 和 launch 参数中。

### 5.2 风险

简单地按 kernel body 删除参数也不安全，因为 launch grid 可能仍依赖 size。若忽略 grid liveness，会生成引用未声明 size 参数的 C++ wrapper。

### 5.3 解决方法

提交实现同时分析：

- Triton kernel body；
- launch grid AST。

只有两者都不引用的 metadata 才被删除。结果：

- 1D divisible：kernel params 9→4；
- 2D divisible：kernel params 15→7；
- 1D/2D stride params 全部降为 0；
- grid 仍需要的 size 参数被保留。

---

## 6. 选择的特化条件

### 6.1 Contiguous Fast Path

启用要求：

- 支持的 tensor-only elementwise 参数；
- runtime shape 兼容；
- 所有参与访问的 tensor 均为完整 contiguous；
- size/stride 满足 int32 安全范围；
- rank 为当前允许的 1D/2D。

满足时：

- 使用连续线性地址；
- 删除 runtime stride 参数；
- 删除通用多维 stride arithmetic；
- 保留必要的 size 和 mask。

### 6.2 Divisible Tile Fast Path

在 contiguous 条件基础上，还要求 tile 无尾块。满足时：

- 删除 load/store 的边界 mask；
- 进入 `flatten_contiguous_divisible` variant。

不满足整除条件时：

- 进入 `flatten_contiguous_masked`；
- 保留边界 mask。

---

## 7. Fallback 条件

以下情况不进入 1D/2D flatten fast path：

- 3D runtime；
- noncontiguous；
- stride-0 broadcast；
- transpose、permute、as_strided；
- scalar argument；
- shape 不兼容；
- size/stride 超出 int32 范围。

最新结果中 9 个 legacy/fallback 场景全部：

```text
correct = true
actual_variant = legacy
dispatch_match = true
```

说明 guard 能拒绝不满足条件的输入，没有为了扩大 coverage 误命中 fallback case。

---

## 8. Runtime 影响

总体数据：

- overall median speedup：1.0000x；
- hit median speedup：0.9982x；
- fallback median speedup：1.0000x；
- 2D hit median speedup：1.0072x；
- 1D hit median speedup：0.9915x。

因此 baseline weakness 的主要证据来自 generated-code complexity，而不是所有微型 kernel 都出现显著 runtime 加速。部分 2D case 获得 1%～1.6% 提升；部分 1D masked case 存在轻微回退。

这是当前实现的已知限制，应在主报告中如实披露。

---

## 9. 结论

baseline 在 contiguous/divisible 可判定场景下主要存在以下弱势：

1. 无尾块时仍保留冗余 mask；
2. 连续布局仍传递并展开通用 stride；
3. 多维 pointer arithmetic 随 rank 增长；
4. 不使用的 shape/stride metadata 未被裁剪；
5. 已知布局缺少可由 runtime dispatcher 实际选择的专门 variant。

提交方案以严格 guard 实现 1D/2D contiguous divisible 和 contiguous masked 两类路径，并在不破坏 fallback 的前提下显著降低生成代码复杂度。
