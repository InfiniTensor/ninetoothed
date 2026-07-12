"""
scatter_add kernel -- pure Triton implementation using tl.atomic_add.

For every position (i_0, ..., i_{N-1}) in `src`, let
    idx = index[i_0, ..., i_{N-1}].
We atomically add src[...] into
    out[i_0, ..., i_{dim-1}, idx, i_{dim+1}, ..., i_{N-1}].

Implementation notes
--------------------
* We flatten the (src, index) iteration space to 1D and use `src.stride()`
  to recover per-dimension coordinates for each flat offset. This handles
  non-contiguous src transparently (e.g. transposes, strided slices).
* The output flat index is computed with the *working buffer's* strides,
  which are contiguous. The wrapper is responsible for copying the result
  back into a possibly non-contiguous `out`.
* A **single fp32 working buffer** is used by the wrapper regardless of
  self / src dtype. The kernel always does `tl.atomic_add` on fp32
  pointers, so the hardware-level atomic serialises concurrent writes
  correctly -- no load-add-store race, even for fp16 inputs.
* This also matches the user's requirement: "fp32 accumulator to
  guarantee precision".
"""

import triton
import triton.language as tl


@triton.jit
def _scatter_add_kernel(
    src_ptr,
    index_ptr,
    work_ptr,
    src_stride0,
    src_stride1,
    src_stride2,
    src_stride3,
    work_stride0,
    work_stride1,
    work_stride2,
    work_stride3,
    out_shape_dim,
    ndim: tl.constexpr,
    dim: tl.constexpr,
    n_src: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_src

    if ndim == 1:
        idx = tl.load(index_ptr + offs, mask=mask, other=0)
        val = tl.load(src_ptr + offs, mask=mask, other=0.0)
        flat = idx * work_stride0
        bounds = mask & (idx >= 0) & (idx < out_shape_dim)
    else:
        rem = offs
        c0 = rem // src_stride0
        rem = rem - c0 * src_stride0
        c1 = rem // src_stride1
        rem = rem - c1 * src_stride1
        c2 = rem // src_stride2
        rem = rem - c2 * src_stride2
        c3 = rem // src_stride3

        idx = tl.load(index_ptr + offs, mask=mask, other=0)

        if ndim == 2:
            if dim == 0:
                flat = idx * work_stride0 + c1 * work_stride1
            else:
                flat = c0 * work_stride0 + idx * work_stride1
        elif ndim == 3:
            if dim == 0:
                flat = idx * work_stride0 + c1 * work_stride1 + c2 * work_stride2
            elif dim == 1:
                flat = c0 * work_stride0 + idx * work_stride1 + c2 * work_stride2
            else:
                flat = c0 * work_stride0 + c1 * work_stride1 + idx * work_stride2
        else:
            if dim == 0:
                flat = (
                    idx * work_stride0
                    + c1 * work_stride1
                    + c2 * work_stride2
                    + c3 * work_stride3
                )
            elif dim == 1:
                flat = (
                    c0 * work_stride0
                    + idx * work_stride1
                    + c2 * work_stride2
                    + c3 * work_stride3
                )
            elif dim == 2:
                flat = (
                    c0 * work_stride0
                    + c1 * work_stride1
                    + idx * work_stride2
                    + c3 * work_stride3
                )
            else:
                flat = (
                    c0 * work_stride0
                    + c1 * work_stride1
                    + c2 * work_stride2
                    + idx * work_stride3
                )

        val = tl.load(src_ptr + offs, mask=mask, other=0.0)
        bounds = mask & (idx >= 0) & (idx < out_shape_dim)

    acc = val.to(tl.float32)
    tl.atomic_add(work_ptr + flat, acc, mask=bounds)


def launch(src, index, work_buf, dim, block_size=2048):
    """Scatter-add src[index] into the fp32 contiguous work_buf."""
    import torch

    ndim = src.ndim
    if not (1 <= ndim <= 4):
        raise ValueError(f"scatter_add: only ndim 1..4 supported, got {ndim}")
    if not (src.is_contiguous() and index.is_contiguous()):
        raise RuntimeError(
            "scatter_add: src and index must be contiguous before launch "
            "(wrapper is responsible for .contiguous())"
        )
    if work_buf.dtype != torch.float32:
        raise RuntimeError(
            "scatter_add: work_buf must be float32 (fp32 accumulator)"
        )
    if not work_buf.is_contiguous():
        raise RuntimeError("scatter_add: work_buf must be contiguous")
    if dim < 0:
        dim += ndim
    if not (0 <= dim < ndim):
        raise ValueError(f"scatter_add: invalid dim={dim} for ndim={ndim}")

    src_strides = list(src.stride()) + [0] * (4 - ndim)
    work_strides = list(work_buf.stride()) + [0] * (4 - ndim)

    n_src = src.numel()
    out_shape_dim = work_buf.shape[dim]

    grid = (triton.cdiv(n_src, block_size),)
    _scatter_add_kernel[grid](
        src,
        index.to(torch.int32),
        work_buf,
        src_strides[0],
        src_strides[1],
        src_strides[2],
        src_strides[3],
        work_strides[0],
        work_strides[1],
        work_strides[2],
        work_strides[3],
        out_shape_dim,
        ndim,
        dim,
        n_src,
        block_size,
    )
