# Layout-sensitive family (non-contiguous / stride / offset)

Output is a re-indexing or re-view of the input: flip, narrow, permute,
space-to-depth (pixel_unshuffle), pooling windows, and any kernel that must
accept a non-contiguous / strided / offset input. This is the family most
commonly missing from existing skills — cover it explicitly.

> Provenance tags — calibrate trust per claim: **[E]** exercised end-to-end on
> GPU in this project's episode runs (raw run logs kept with the companion
> harness, not committed to this package — re-run to reproduce) · **[S]** verified
> against the `ninetoothed==0.25.0` source, not executed here · **[I]**
> inferred from adjacent patterns — re-verify before relying on it.

**Family boundary.** Re-indexing here means index maps that are pure functions
of position — computable at arrangement time from shapes. A scatter whose
write addresses come from tensor *values*, or where several sources may write
one output element (needs atomic RMW), is NOT expressible in this family or
this DSL: take the declared-fallback path (SKILL.md §3, `dsl_limit`) instead
of forcing it. Verify with
`scripts/inspect_generated_source.py --contract atomic` if in doubt — plain
stores on a colliding scatter are race-prone even when tests pass.

> **Verify the exact meta-op signature against your installed version.** This
> playbook is written for `ninetoothed==0.25.0` whose `Tensor` exposes:
> `tile(tile_shape, strides=None, dilation=None, floor_mode=False)`, `expand`,
> `squeeze(dim)`, `unsqueeze(dim)`, `permute(dims)`,
> `flatten(start_dim=None, end_dim=None)`, `ravel()`, `pad(pad)`,
> `offsets()` (no args). If your version differs, re-read
> `src/ninetoothed/tensor.py` before relying on a signature.

## Decision: handle layout in the arrangement, or in the wrapper?

Two legitimate strategies — pick deliberately and state which you used:

1. **Wrapper contiguity fast-path** (simplest, always correct): call
   `x = x.contiguous()` (or `.view()/.permute().contiguous()`) in the torch
   wrapper, then run a standard contiguous kernel. Use this when the layout
   transform is cheap relative to the compute, or as the correctness baseline.
2. **In-arrangement layout** (no extra copy): express the transform with
   `permute` / `tile(strides=, dilation=)` / `ravel` / `flatten` so the kernel
   reads the original storage directly. Use when avoiding the copy matters for
   performance.

Always implement (1) first as the correctness oracle, then (2) if perf needs it,
and benchmark (2) vs (1).

## Strided / windowed tiling — `tile(strides=, dilation=)` — [S: signature in `src/ninetoothed/tensor.py`]

`tile` accepts `strides` (interval at which each tile starts) and `dilation`
(spacing between elements in a tile). Default `strides=-1` means non-overlapping
(stride = tile size). For overlapping windows set `strides` smaller than the
tile; for dilated windows set `dilation>1`.

```python
# non-overlapping 2x2 windows over the last two dims (pooling-style)
input.tile((1, 1, 2, 2))
# overlapping windows: 3x3 tile, stride 1
input.tile((1, 1, 3, 3), strides=(1, 1, 1, 1))
```

## Space-to-depth / windowed collapse — ravel + flatten — [S: adapted from `tests/test_max_pool2d.py` source, not executed here]

This is the real `max_pool2d` arrangement: tile out the window dims, `ravel()`
to flatten the whole hierarchy, `flatten` to merge batch dims, then re-`tile`
into blocks. The application reduces along the collapsed window axis.

```python
def arrangement(input, output):
    input_arranged = input.tile((1, 1, WINDOW_HEIGHT, WINDOW_WIDTH))
    input_arranged = input_arranged.ravel()
    input_arranged = input_arranged.flatten(end_dim=4).flatten(start_dim=1)
    input_arranged = input_arranged.tile((BLOCK_SIZE, -1))

    output_arranged = output.tile((1, 1, 1, 1)).ravel()
    output_arranged = output_arranged.flatten(end_dim=4).flatten(start_dim=1)
    output_arranged = output_arranged.tile((BLOCK_SIZE, -1))
    output_arranged.dtype = output_arranged.dtype.squeeze(1)
    return input_arranged, output_arranged
```

For `pixel_unshuffle` (space-to-depth, factor r) **[I — shape algebra
extrapolated from the max_pool2d pattern, not executed here]**: the same
shape algebra — tile the H,W dims by r, ravel/permute so the r·r window lands
on the channel axis, write to a `(B, C·r·r, H/r, W/r)` output. Build the
correctness oracle from `torch.nn.functional.pixel_unshuffle` first.

## Transpose-style — `permute(dims)` — [S]

`permute` reorders dims at the arrangement level without a data copy. Compose
with `tile` (permute first, then tile) when you need a transposed access
pattern. Cross-check the result with `simulate_arrangement` (perf-diag.md).

## Padding — `pad(pad)` — [S]

`pad(pad)` pads each dim by `(left, right)`. Combine with `other=` on the source
`Tensor` so the padded region reads a defined fill value.

## Non-contiguous correctness test (required for this family)

The matrix MUST include a non-contiguous input, e.g. a transposed or sliced
tensor, compared against the PyTorch reference on the same logical values:

```python
x = torch.randn(64, 128, device="cuda")
x_nc = x.t()                       # non-contiguous view
# expected = torch.<op>(x_nc); got = my_op(x_nc); assert_close(got, expected)
```

## Verify the arrangement before running the kernel

`ninetoothed.debugging.simulate_arrangement(arrangement, tensors)` returns, for
each tensor, a source index grid and the target (arranged) grid filled with
source indices — so you can confirm the tiling maps elements where you intend,
deterministically, without a real kernel. See `perf-diag.md`.

## Pitfalls

- Assuming `offsets()` takes a dim argument — in 0.25.0 it does not. Re-check.
- Building the output arrangement to mirror the input's window blocks → may hit
  `RecursionError`; arrange the output independently (see common-errors.md).
- `unsqueeze` inside an arrangement can fail to eval on some versions; prefer
  reshaping in the wrapper (`view`/`unsqueeze`) before the kernel.
