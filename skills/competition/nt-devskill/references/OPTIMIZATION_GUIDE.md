# NineToothed Optimization & Debugging Guide

## Performance Optimization

### 1. Roofline Model

Understand where your kernel is bottlenecked:

- **Memory-bound** (low arithmetic intensity): optimize memory access patterns.
- **Compute-bound** (high arithmetic intensity): optimize tile sizes and ILP.

Estimate arithmetic intensity:

```
intensity = (FLOPs) / (bytes loaded + bytes stored)
```

| Operator       | Intensity (FP16, typical) | Bound By                        |
| -------------- | ------------------------- | ------------------------------- |
| Vector Add     | ~0.5 FLOP/byte            | Memory                          |
| SiLU / SwiGLU  | ~1-2 FLOP/byte            | Memory                          |
| Softmax        | ~2-4 FLOP/byte            | Memory                          |
| MatMul (large) | >100 FLOP/byte            | Compute                         |
| Attention      | ~10-50 FLOP/byte          | Compute (with large hidden dim) |

### 2. Tile Size Selection

Guidelines for choosing `BLOCK_M`, `BLOCK_N`, `BLOCK_K`:

- Set tile sizes as powers of 2.
- Each tile dimension should not exceed 128 (soft limit per SM).
- Product of tile dims ≤ 1024 elements per thread block (for most GPUs).
- **MatMul:** `BLOCK_M=BLOCK_N=128`, `BLOCK_K=32` or `64` are good starting points. Use `block_size()` (meta) for autotuning.
- **Element-wise:** `BLOCK_SIZE=1024` or `2048` works well.
- **Softmax:** `BLOCK_SIZE` = `input.shape[-1]` (the entire row fits in one tile).
- **Attention:** Use `shape_options={"constexpr": True, "upper_bound": 128}` for `head_dim` to help compiler optimize.

#### MetaX GPU Private Memory Constraint (CRITICAL)

MetaX C500 GPUs have a **4 KB private memory limit per thread**. This directly constrains tile sizes:

```
Private memory per thread ≈ tile_elements × bytes_per_element
For float32: 4096 bytes / 4 = 1024 elements max per tile
For float16: 4096 bytes / 2 = 2048 elements max per tile
```

**Tile size limits on MetaX:**

| Tile Shape (float32) | Elements | Private Memory | Safe?      |
| -------------------- | -------- | -------------- | ---------- |
| 32 × 32              | 1,024    | 4 KB           | ✅ At limit |
| 64 × 64              | 4,096    | 16 KB          | ❌ Exceeds  |
| 32 × 64              | 2,048    | 8 KB           | ❌ Exceeds  |
| 16 × 16              | 256      | 1 KB           | ✅ Safe     |
| 32 × 16              | 512      | 2 KB           | ✅ Safe     |

**Recommendations for MetaX:**
1. **2D tiles (non-matmul):** Use `BLOCK_SIZE_M ≤ 32`, `BLOCK_SIZE_N ≤ 32`. Prefer 16×16 for safety.
2. **Intermediate arrays count too:** Row indices, masks, comparison results all consume private memory. Leave 50% headroom.
3. **MatMul is exempt:** `ntl.dot` uses tensor cores and doesn't materialize full tiles in private memory. Standard `128×128` output tiles work.
4. **If you see "private memory exceed" error:** Reduce tile dimensions by half and retry. Check that no intermediate variable exceeds the budget.

> **Rule of thumb:** For non-dot 2D tiles on MetaX, keep total tile elements ≤ 512 (float32) to leave room for intermediates.

### 3. Memory Access Patterns

- **Coalesced access:** Ensure consecutive threads access consecutive memory addresses. The last dimension of the tile should map to the contiguous dimension of the tensor.
- **Vectorized loads:** NineToothed handles this automatically when tile shapes align with 16-byte boundaries.
- **Shared memory:** NineToothed manages shared memory implicitly through the arrangement system. Complex kernels may benefit from explicit shared memory usage via Triton's `tl.load(..., eviction_policy="evict_first")`.
- **Strided access:** Use `tile(shape, strides=...)` and `tile(shape, dilation=...)` for non-standard access patterns (e.g., interleaved RoPE).

### 4. Avoiding Warp Divergence

- Prefer `ntl.where(condition, a, b)` over explicit `if` statements inside `application`.
- Use `ntl.maximum` / `ntl.minimum` rather than conditional branches.
- Keep all threads in a warp executing the same path.

### 5. Partial Tile Boundary Handling

When tiling, the last tile may be partial (fewer elements than `BLOCK_SIZE`). Two standard approaches:

**Approach 1: Fill value via `Tensor(other=...)`**

For reduce operations (max, sum), set a fill value on the input tensor declaration:

```python
# Softmax: fill out-of-bounds with -inf so max is correct
tensors = (Tensor(2, other=float("-inf")), Tensor(2))

# MaxPool2D: fill with -inf so max ignores padding
kernel = ninetoothed.make(arrangement, application, (Tensor(4, other=float("-inf")), Tensor(4)))
```

This ensures partial tiles are padded with the fill value, so reduce operations produce correct results.

**Approach 2: Explicit mask via `.offsets()` + `.source.shape`**

For attention and other patterns where you need fine-grained control:

```python
# Mask out-of-bounds K positions with -inf
qk = ntl.where(k[i].offsets(-2) < k.source.shape[-2], qk, float("-inf"))
```

Here `k[i].offsets(-2)` gives the global position of each element in the tile, and `k.source.shape[-2]` gives the original (untiled) dimension size.

### 6. Precision & Accumulator

- Always accumulate MatMul and attention results in `float32` for numerical stability.
- Use `.to(dtype)` for the final store.
- Use `ntl.cast(x, ntl.float32)` for intermediate precision-sensitive computations (e.g., `sigmoid`, `rsqrt`).
- Online softmax should use `ntl.exp2` (base-2) instead of `ntl.exp` for better performance; convert scale factor: `q * scale * 1.44269504089`.

#### GPU-side Precision Control (CRITICAL)

**libdevice 函数只支持 float32/float64**，float16 输入必须显式 cast：

```python
from ninetoothed.language import libdevice

def application(input, output):
    # ✅ 显式 cast 到 float32，再调用 libdevice
    input_f32 = ntl.cast(input, ntl.float32)
    output = libdevice.lgamma(input_f32)
    # 结果会被自动 cast 回 input.dtype

    # ❌ 不 cast → float16 输入行为未定义
    output = libdevice.lgamma(input)  # WRONG for float16
```

**kernel 中的数学常量应在 GPU 侧用 `ntl.cast` 生成**，避免 Python float 精度差异：

```python
# ✅ GPU 侧常量
def application(input, output):
    PI = ntl.cast(3.141592653589793, ntl.float32)
    output = input * ntl.cast(180.0, ntl.float32) / PI

# ⚠️ Python 侧常量（可能有效，但不够显式）
_RAD_TO_DEG = 180.0 / math.pi
def application(input, output):
    output = input * _RAD_TO_DEG  # 依赖编译器提升
```

#### 函数选用速查表

| 需要的函数                      | ntl 有? | libdevice 有? | 正确做法                            |
| ------------------------------- | ------- | ------------- | ----------------------------------- |
| exp, exp2, sqrt, rsqrt, sigmoid | ✅       | ✅             | 用 `ntl.X`                          |
| log, log2, lgamma, tgamma, erf  | ❌       | ✅             | 用 `libdevice.X` + `ntl.cast`       |
| copysign, pow, floor, ceil      | ❌       | ✅             | 用 `libdevice.X` + `ntl.cast`       |
| SiLU (`x * sigmoid(x)`)         | 组合    | ❌             | 用 `x * ntl.sigmoid(x)`             |
| SwiGLU                          | 组合    | ❌             | 用 `a * (b * ntl.sigmoid(cast(b)))` |

### 7. Autotuning

Use `block_size()` or `Symbol(..., meta=True)` to enable autotuning:

```python
BLOCK_SIZE_M = block_size()       # searched over [16, 32, 64, 128]
BLOCK_SIZE_N = block_size()
BLOCK_SIZE_K = block_size()
```

The autotuner tries combinations to find the fastest configuration.

To disable autotuning (for faster development iteration), replace with fixed values:

```python
BLOCK_SIZE_M = 128
BLOCK_SIZE_N = 128
BLOCK_SIZE_K = 32
```

#### Autotuning 默认值陷阱（CRITICAL）

> **`block_size()` 的默认搜索范围可能选择次优 tile size，导致性能差 2x 甚至更多。**

**实测案例（linspace on MetaX C500）：**

| Tile Size             | GPU Time | Bandwidth | vs Optimal      |
| --------------------- | -------- | --------- | --------------- |
| 256 (autotuning 选中) | 0.062 ms | 65 GB/s   | **2.7x slower** |
| 512                   | 0.023 ms | 173 GB/s  | 1.0x            |
| 1024                  | 0.024 ms | 172 GB/s  | 1.0x            |
| 4096                  | 0.023 ms | 174 GB/s  | 1.0x            |

**推荐做法：**

1. **先用 tile sweep 找最优值**（`scripts/diag_tile_sweep.py`），再在 `premake` 中显式设置：
   ```python
   def premake(dtype=None, block_size=1024):  # 显式默认值，不用 autotuning
   ```

2. **如果必须用 autotuning**，限制搜索范围并验证结果：
   ```python
   BLOCK_SIZE = Symbol("BLOCK_SIZE", meta=True, upper_bound=4096)
   ```

3. **始终在优化阶段运行 tile sweep**，不要假设 autotuning 的结果就是最优的。

### 8. Kernel Composition for Performance

Reusing `arrangement`/`application` from existing kernels reduces code and can improve performance:

- **Less code** → smaller compilation time, easier to maintain.
- **Shared optimization** → improvements to the base kernel benefit all composed kernels.
- **Fused operations** → e.g., `addmm` fuses matmul + bias addition in one kernel launch.

See `references/PATTERNS.md` §6 for composition patterns.

---

## Common Errors & Fixes

### Error 1: Out-of-bounds memory access

```
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
```

**Fix:** Use `Tensor(other=...)` or explicit `ntl.where` masks. When tiling with `tile((-1, BLOCK_SIZE))`, the last tile may be partial. Check offsets:

```python
ntl.where(tensor.offsets(-2) < source.shape[-2], value, 0.0)
```

### Error 2: `dtype` dimension mismatch

```
ValueError: cannot squeeze dim X
```

**Fix:** The `dtype` manipulations after `expand()` need careful tracking. The dimension count increases with each `tile()` call. Verify the sequence:

```
tile((BM, BK))       # adds 1 dim (now 3)
tile((1, -1))        # adds 1 dim (now 4)
expand(...)          # broadcasts
squeeze(0)           # remove the extra dim (back to 3)
```

For 3D+ tiling (bmm, attention), you may need **double-level** dtype access:

```python
arranged.dtype = arranged.dtype.squeeze((0, 1))
arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))  # second level
```

### Error 3: Kernel returns zeros

**Fix:** Check that `output` is correctly assigned in `application`. NineToothed uses the local variable name that corresponds to the output tensor:

```python
def application(..., output):
    output = result  # this assignment stores the result
```

Forgetting the assignment means the output buffer is never written.

### Error 4: Autotuning takes too long

**Fix:** Reduce the search space by:
1. Adding `upper_bound` to `Symbol`: `Symbol("BLOCK", meta=True, upper_bound=128)`
2. Temporarily using fixed values during development
3. Caching autotuning results (built-in Triton mechanism)

### Error 5: Broadcast shape mismatch

```
RuntimeError: The size of tensor a (X) must match the size of tensor b (Y)
```

**Fix:** Use `[:, None]` or `[:, None, :]` for proper broadcasting in reductions:

```python
row_max = ntl.max(x, 1)           # shape: [M]
normalized = x - row_max[:, None]  # shape: [M, BLOCK] -- correct
```

### Error 6: Kernel launch fails silently

**Fix:** Verify the grid dimensions. Use `triton.cdiv` for grid computation. Ensure the total grid size is not zero.

### Error 7: Numerical instability in softmax/attention

**Fix:** Always subtract the max before exp. For attention, use online softmax:

```python
# Softmax: use Tensor(other=float("-inf")) for stable max
# Attention: use online softmax with m_i tracking
m_ij = ntl.maximum(m_i, ntl.max(qk, 1))
p = ntl.exp2(qk - m_ij[:, None])  # base-2 exp for performance
```

---

## Debugging Workflow

1. **Start with torch:** Implement a naive PyTorch version and verify correctness.
2. **Simplify:** Begin with fixed tile sizes (no autotuning). Use a small input (e.g., `128x128`).
3. **Compare:** Run the NineToothed version against the torch reference.
4. **Iterate:** Gradually increase size, enable autotuning, add fusion.
5. **Profile:** Use `scripts/benchmark.py` to measure performance.
6. **Autotune:** Enable `meta=True` symbols and let the autotuner find optimal values.

### Quick Checklist

- [ ] Output tensor is assigned in `application`
- [ ] `ntl.where` masks or `Tensor(other=...)` cover partial tiles
- [ ] `dtype` squeeze dimensions match the expected rank
- [ ] Float32 accumulator for reductions and matmuls
- [ ] `ntl.cast` used for precision-sensitive operations (sigmoid, rsqrt)
- [ ] `BLOCK_SIZE` <= the actual dimension size (or handles partial tiles)
- [ ] Input tensors are on CUDA device
- [ ] CUDA is available (`torch.cuda.is_available()`)

---

## Correctness Matrix Testing

Before running performance benchmarks, verify correctness across a comprehensive matrix of shapes, dtypes, and layouts.

### Test Matrix Design

For each operator, test across three dimensions:

```
Shapes:   small, medium, large, non-power-of-2
Dtypes:   float32, float16
Layouts:  contiguous, transposed (.t()), strided ([::2, ::3])
```

**Example test matrix for a 2D operator:**

| Shape        | dtype | Layout            | Expected              |
| ------------ | ----- | ----------------- | --------------------- |
| (32, 64)     | fp32  | contiguous        | PASS                  |
| (32, 64)     | fp16  | contiguous        | PASS (atol=1e-3)      |
| (4096, 4096) | fp32  | contiguous        | PASS                  |
| (4096, 4096) | fp16  | contiguous        | PASS (atol=1e-3)      |
| (100, 333)   | fp32  | contiguous        | PASS (non-power-of-2) |
| (512, 256)   | fp32  | transposed        | PASS (non-contiguous) |
| (512, 512)   | fp32  | strided [::2,::3] | PASS                  |

### Precision Metrics: MERE & MARE

For numerical correctness, compute two error metrics:

```python
def compute_error_metrics(result, reference):
    """Compute MERE and MARE for correctness validation."""
    diff = (result - reference).abs()
    abs_err = diff.max().item()
    denom = reference.abs().clamp(min=1e-8)
    rel_err = (diff / denom).max().item()
    mean_abs = diff.mean().item()
    return {"max_abs_err": abs_err, "max_rel_err": rel_err, "mean_abs_err": mean_abs}
```

**Passing thresholds by dtype:**

| dtype    | MERE threshold | MARE threshold |
| -------- | -------------- | -------------- |
| float32  | ≤ 1e-4         | ≤ 1e-4         |
| float16  | ≤ 1e-2         | ≤ 1e-2         |
| bfloat16 | ≤ 1e-2         | ≤ 1e-2         |

> **禁止放宽容差：** 如果精度不达标，说明 kernel 实现有问题（如缺少 fp32 accumulator），不要通过放大 atol/rtol 来"通过"测试。

### Non-Contiguous Input Testing

**必须测试非连续输入。** 九齿通过 stride 信息自动处理非连续 tensor，但 kernel 实现可能假设连续内存。

```python
# 必须包含的非连续测试用例
input_c = torch.randn(256, 512, device="cuda")

test_cases = [
    ("contiguous", input_c),
    ("transposed", input_c.t()),
    ("strided",    input_c[::2, ::3]),
    ("sliced",     input_c[:, :100]),
]

for label, inp in test_cases:
    result = my_kernel(inp)
    expected = torch_reference(inp)
    assert torch.allclose(result, expected, atol=1e-5), f"FAIL on {label}"
```

---

## Performance Optimization Workflow

> **铁律：正确性通过后，必须进入性能优化阶段。不要停留在"能跑"的状态。**
> **No optimization without measurement — every change must be backed by diagnostic data.**

### Overview: 4-Round Optimization Loop

```
Round 1: DIAGNOSE → 写诊断脚本 → 运行 → 读输出 → 定位瓶颈类型
Round 2: SWEEP   → 写 tile 扫描脚本 → 运行 → 找最优 tile 配置
Round 3: OPTIMIZE → 根据诊断数据选方案 → 改代码 → 重测正确性
Round 4: VERIFY   → 重跑诊断 → 对比 before/after → 报告结果
            ↓
         speedup < 1.0x? → 回到 Round 2（最多 3 轮）
```

---

### Phase 1: Diagnostic Scripts (Round 1-2)

#### Diagnostic 1: Overhead Breakdown (`diag_overhead.py`)

**Already available:** `python scripts/diag_overhead.py --op <name>`

For operators not in the built-in list, the AI must write a custom diagnostic:

```python
#!/usr/bin/env python3
"""Custom overhead breakdown for operator <name>."""
import time, torch

def overhead_breakdown(nt_fn, torch_fn, args_list, warmup=20, trials=100):
    """
    Decompose timing into E2E, GPU-only, and host overhead.
    args_list: list of (shape_label, args_tuple) pairs for multi-shape testing.
    """
    for shape_label, args in args_list:
        # Warmup
        for _ in range(warmup):
            nt_fn(*args)
        torch.cuda.synchronize()

        # E2E (wall clock: Python dispatch + GPU compute)
        t0 = time.perf_counter()
        for _ in range(trials):
            nt_fn(*args)
        torch.cuda.synchronize()
        e2e_ms = (time.perf_counter() - t0) * 1000 / trials

        # GPU-only (CUDA events: excludes Python dispatch)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(trials):
            nt_fn(*args)
        end.record()
        torch.cuda.synchronize()
        gpu_ms = start.elapsed_time(end) / trials

        # Compute bandwidth
        nbytes = sum(a.numel() * a.element_size()
                     for a in args if isinstance(a, torch.Tensor))
        bw_gpu = nbytes / (gpu_ms * 1e-3) / 1e9  # GB/s

        # Host overhead
        host_ms = e2e_ms - gpu_ms
        host_pct = host_ms / e2e_ms * 100 if e2e_ms > 0 else 0

        # Diagnosis
        if host_pct > 50:
            bottleneck = "LAUNCH-BOUND"
        elif host_pct > 20:
            bottleneck = "MIXED"
        else:
            bottleneck = "COMPUTE-BOUND"

        print(f"[{shape_label}]")
        print(f"  E2E={e2e_ms:.4f}ms  GPU={gpu_ms:.4f}ms  "
              f"host={host_ms:.4f}ms ({host_pct:.0f}%)  "
              f"BW={bw_gpu:.1f}GB/s  → {bottleneck}")
```

**Example output and interpretation:**

```
[(128, 128)]
  E2E=0.0520ms  GPU=0.0150ms  host=0.0370ms (71%)  BW=12.8GB/s  → LAUNCH-BOUND
[(4096, 4096)]
  E2E=0.1800ms  GPU=0.1400ms  host=0.0400ms (22%)  BW=480.0GB/s  → MIXED
```

Interpretation:
- Small shapes → launch overhead dominates (expected for elementwise)
- Large shapes → compute starts to matter, BW utilization tells if memory pattern is optimal

#### Diagnostic 2: Tile Size Sweep (`diag_tile_sweep.py`)

**Already available:** `python scripts/diag_tile_sweep.py --op <name>`

For custom operators, the AI must write:

```python
#!/usr/bin/env python3
"""Tile size sweep for operator <name>."""
import torch

def tile_sweep(build_kernel_fn, args, nbytes, candidates, warmup=10, trials=50):
    """
    Sweep tile sizes, report GPU time + bandwidth.
    build_kernel_fn(tile_size) → callable kernel
    """
    print(f"{'tile':>12s} | {'GPU ms':>10s} | {'BW GB/s':>10s} | {'ratio':>8s} | status")
    print("-" * 65)
    best_ms, best_tile = float("inf"), None

    for tile in candidates:
        try:
            kernel = build_kernel_fn(tile)
            # warmup
            for _ in range(warmup):
                kernel(*args)
            torch.cuda.synchronize()

            # measure
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(trials):
                kernel(*args)
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end) / trials

            bw = nbytes / (ms * 1e-3) / 1e9
            if ms < best_ms:
                best_ms, best_tile = ms, tile
            ratio = ms / best_ms
            label = str(tile) if isinstance(tile, int) else "x".join(map(str, tile))
            status = "★ BEST" if ms == best_ms else f"{ratio:.2f}x"
            print(f"{label:>12s} | {ms:>10.4f} | {bw:>10.2f} | {ratio:>7.2f}x | {status}")
        except Exception as e:
            label = str(tile) if isinstance(tile, int) else "x".join(map(str, tile))
            print(f"{label:>12s} | {'FAIL':>10s} | {'---':>10s} | {'---':>8s} | {e}")

    if best_tile:
        worst_ms = max(ms for _, ms in results if ms < float("inf"))
        spread = worst_ms / best_ms
        print(f"\n★ Optimal: tile={best_tile}, GPU={best_ms:.4f}ms, spread={spread:.1f}x")
        if spread > 2.0:
            print("  → Large spread: tile choice has major impact. Use optimal tile.")
        else:
            print("  → Small spread: tile choice has minor impact. Consider other optimizations.")
```

**Sweep ranges by operator family:**

| Family           | Candidates                                             | Rationale                     |
| ---------------- | ------------------------------------------------------ | ----------------------------- |
| Elementwise (1D) | `[256, 512, 1024, 2048, 4096]`                         | Larger tiles = fewer launches |
| Reduction (2D)   | `[128, 256, 512, 1024, 2048]`                          | Must fit one row              |
| MatMul (3D)      | `[(32,32,16), (64,64,32), (128,128,32), (128,128,64)]` | Standard GEMM configs         |
| Attention (4D)   | `[(1,1,64,64), (1,1,128,128)]`                         | Head dim constrained          |
| MetaX 2D non-dot | `[(16,16), (16,32), (32,16), (32,32)]`                 | 4KB private memory limit      |

#### Diagnostic 3: Multi-Shape Profiling

For operators used across different shapes, profile all relevant shapes:

```python
shapes = [
    ("small",  (128, 128)),
    ("medium", (1024, 1024)),
    ("large",  (4096, 4096)),
    ("tall",   (16384, 256)),
    ("wide",   (256, 16384)),
]
# Run overhead_breakdown for each shape to see if bottleneck changes with size
```

---

### Phase 2: Optimization Decision Tree (Round 3)

After collecting diagnostic data, follow this decision tree:

```
                        diagnostic output
                              │
                    ┌─────────┴─────────┐
                    │                   │
              host > 50%?          host < 20%?
                    │                   │
              ┌─────┴─────┐      ┌─────┴─────┐
              │           │      │           │
          tile sweep   BW > 50  BW < 50   speedup > 1x?
          spread > 2x?  GB/s?   GB/s?         │
              │          │      │       ┌─────┴─────┐
          ┌───┴───┐      │      │       │           │
          │       │      │      │     YES          NO
         YES     NO      │      │       │           │
          │       │      │      │    Report     tile sweep
     Use best  Try dtype │  Fix memory   result   + next
     tile      fp16      │  coalescing           direction
                         │
                    Already fast
```

#### Optimization Cards

Each card maps a specific diagnostic signature to a concrete action with expected gain.

**OC-1: Tile Size Fix (from sweep data)**
- **Trigger:** sweep spread > 2x, current tile is not optimal
- **Action:** Change `BLOCK_SIZE` in arrangement to the sweep's optimal value
- **Expected gain:** 20-200% depending on spread
- **Code change:**
  ```python
  # Before
  def arrangement(input, output, BLOCK_SIZE=1024):
  # After (sweep showed 2048 is optimal)
  def arrangement(input, output, BLOCK_SIZE=2048):
  ```

**OC-2: Dtype Downgrade (when BW-limited)**
- **Trigger:** host < 20%, BW < 50% of peak, speedup < 1x
- **Action:** Switch from float32 to float16 (halves bandwidth requirement)
- **Expected gain:** ~2x for memory-bound ops
- **Code change:**
  ```python
  # In torch wrapper: ensure input is float16
  input = input.to(torch.float16) if input.dtype == torch.float32 else input
  # In application: keep ntl.cast for precision-sensitive ops only
  x_fp32 = ntl.cast(x, ntl.float32)  # only for sigmoid/rsqrt/accumulator
  ```

**OC-3: Autotuning Enable (when shape varies)**
- **Trigger:** operator used across many shapes, fixed tile underperforms on some
- **Action:** Replace fixed `BLOCK_SIZE` with `block_size()` (meta=True)
- **Expected gain:** 10-50% (avoids suboptimal tile for specific shapes)
- **Code change:**
  ```python
  # Before
  BLOCK_SIZE = 1024
  # After
  from ninetoothed import block_size
  BLOCK_SIZE = block_size()  # autotuner searches [256, 512, 1024, 2048, ...]
  ```

**OC-4: Kernel Fusion (when launch-bound)**
- **Trigger:** host > 50%, multiple related kernels in sequence
- **Action:** Merge operations into one kernel's `application` function
- **Expected gain:** 5-20us per eliminated launch
- **Example:** `linear_relu` = matmul + bias + ReLU in one `application`

**OC-5: exp2 Optimization (when using ntl.exp)**
- **Trigger:** kernel uses `ntl.exp()` in hot loop (softmax, attention)
- **Action:** Replace `ntl.exp(x)` with `ntl.exp2(x * 1.44269504089)`
- **Expected gain:** 5-15% for exp-heavy kernels
- **Code change:**
  ```python
  # Before
  p = ntl.exp(qk - m_ij[:, None])
  # After
  LOG2E = 1.44269504089
  p = ntl.exp2((qk - m_ij[:, None]) * LOG2E)
  ```

**OC-6: Memory Coalescing Fix (when BW is low)**
- **Trigger:** BW < 50 GB/s, host < 20%
- **Action:** Ensure last tile dimension maps to contiguous tensor dimension
- **Code change:**
  ```python
  # Bad: tile (BLOCK_M, BLOCK_N) on row-major tensor → N should be last dim
  # Good: tile (BLOCK_M, BLOCK_N) where BLOCK_N maps to contiguous dim
  input = input.tile((BLOCK_M, BLOCK_N))  # BLOCK_N along dim[-1] (contiguous)
  ```

---

### Phase 3: Validation Protocol (Round 4)

After each optimization, execute this exact sequence:

```bash
# Step 1: Correctness regression (MUST PASS)
python scripts/validate.py --op <name>

# Step 2: Re-run diagnostics
python scripts/diag_overhead.py --op <name>

# Step 3: If tile changed, re-run sweep to confirm still optimal
python scripts/diag_tile_sweep.py --op <name>
```

**Record before/after in this format:**

```
=== Optimization Report: <op_name> ===
Optimization applied: <OC-N card name> — <brief description>

Before:
  GPU: 0.250 ms | E2E: 0.400 ms | host: 37% | BW: 120 GB/s | speedup: 0.72x

After:
  GPU: 0.150 ms | E2E: 0.220 ms | host: 32% | BW: 200 GB/s | speedup: 1.20x

Delta: GPU +66% | E2E +82% | BW +67%
Verdict: ✅ IMPROVED (speedup 0.72x → 1.20x)
```

**Iteration rules:**
- speedup ≥ 1.0x → report to user, optimization complete
- speedup < 1.0x → try next Optimization Card from the decision tree
- Maximum 3 optimization rounds per operator
- If all rounds exhausted and still < 1.0x → report honestly with diagnostic data

---

### MetaX-Specific Performance Notes

- **Host dispatch overhead**: MetaX MACA compatibility layer adds ~10-30us per kernel launch vs. CUDA. This makes kernel fusion and reducing launches more impactful.
- **Autotuning overhead**: First call with `block_size()` triggers autotuning (can take 30-120s). Use `upper_bound` on Symbol to reduce search space.
- **Private memory**: 4KB per thread limit means larger tiles can silently spill to slower memory. Always sweep tile sizes on MetaX — intuition from CUDA may not transfer.
- **TF32**: Add `torch.backends.cuda.matmul.allow_tf32 = False` if kernel involves matmul and you need bit-exact results.
- **Warp size**: MetaX C500 has warp size 32 (same as NVIDIA CUDA). Tile dimensions should be multiples of 32.
- **`.item()` trap**: Each `.item()` call triggers GPU sync (~20-40us). Avoid in hot paths.
