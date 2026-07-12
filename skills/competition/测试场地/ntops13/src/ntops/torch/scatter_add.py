"""
PyTorch wrapper for scatter_add.

Mirrors the semantics of ``torch.scatter_add(self, dim, index, src)``:

    out = self.clone()
    for each multi-index I in src:
        out[I with I[dim] replaced by index[I]] += src[I]

Design choices
--------------
* Negative ``dim`` is normalised.
* The wrapper enforces ``self.ndim == index.ndim == src.ndim`` and
  ``index.dtype == torch.long``.
* Out-of-bounds ``index`` values raise a clear ``IndexError`` instead of
  silently producing wrong answers. The check is done with GPU-side
  ``torch.any`` so it does NOT force an extra host-device sync.
* The kernel works on an **fp32 contiguous working buffer**. ``self`` is
  copied in (cast if needed), then ``tl.atomic_add`` serialises all
  conflicting writes at the hardware level, then the result is cast
  back to the original dtype and copied into ``out``.
* This guarantees a correct fp32 accumulator even when ``self`` / ``src``
  are fp16, and also handles arbitrary ``self`` layouts because the
  final ``Tensor.copy_`` respects ``out``'s own strides.
"""

import torch

from ntops.kernels import scatter_add as _kernel


def scatter_add(self, dim, index, src, *, out=None):
    if not isinstance(self, torch.Tensor) or not isinstance(src, torch.Tensor):
        raise TypeError("scatter_add: self and src must be torch.Tensor")
    if not isinstance(index, torch.Tensor):
        raise TypeError("scatter_add: index must be torch.Tensor")
    if self.device != src.device or self.device != index.device:
        raise RuntimeError(
            f"scatter_add: all tensors must share device, got "
            f"{self.device}, {src.device}, {index.device}"
        )

    if self.ndim != src.ndim or self.ndim != index.ndim:
        raise ValueError(
            f"scatter_add: self.ndim ({self.ndim}), index.ndim "
            f"({index.ndim}), src.ndim ({src.ndim}) must all match"
        )
    if index.dtype != torch.long:
        raise TypeError(
            f"scatter_add: index.dtype must be torch.long, got {index.dtype}"
        )
    if src.dtype != self.dtype:
        raise TypeError(
            f"scatter_add: self.dtype ({self.dtype}) and src.dtype "
            f"({src.dtype}) must match"
        )
    if src.dtype not in (torch.float32, torch.float16):
        raise TypeError(
            f"scatter_add: unsupported dtype {src.dtype}; "
            "only float32 / float16 are supported"
        )

    ndim = self.ndim
    if ndim == 0:
        raise ValueError("scatter_add: 0-D self is not supported")

    if dim < 0:
        dim += ndim
    if not (0 <= dim < ndim):
        raise ValueError(
            f"scatter_add: dim={dim} out of range for ndim={ndim}"
        )

    if index.shape != src.shape:
        raise ValueError(
            f"scatter_add: index.shape {tuple(index.shape)} must equal "
            f"src.shape {tuple(src.shape)}"
        )

    for d in range(ndim):
        if d != dim and src.shape[d] > self.shape[d]:
            raise ValueError(
                f"scatter_add: src.shape[{d}]={src.shape[d]} must be <= "
                f"self.shape[{d}]={self.shape[d]} (non-scatter dim)"
            )

    if out is None:
        out = self.clone()
    else:
        if out.shape != self.shape:
            raise ValueError(
                f"scatter_add: out.shape {tuple(out.shape)} must equal "
                f"self.shape {tuple(self.shape)}"
            )
        if out.dtype != self.dtype:
            raise TypeError(
                f"scatter_add: out.dtype {out.dtype} must equal "
                f"self.dtype {self.dtype}"
            )
        out.copy_(self)

    if src.numel() == 0:
        return out

    # GPU-side OOB check: no host-device sync forced. The `.any()` call
    # enqueues a reduction on the current CUDA stream; if it evaluates
    # True, the `.item()` here is the only sync point.
    dim_len = self.shape[dim]
    if torch.any(index < 0).item() or torch.any(index >= dim_len).item():
        raise IndexError(
            f"scatter_add: index out of bounds for self.shape[{dim}]="
            f"{dim_len}"
        )

    # fp32 working buffer: guarantees correct accumulation under heavy
    # atomic contention, and correct fp16 rounding via fp32 accumulator.
    # `.clone()` is required BEFORE `.to(...).contiguous()`: when self
    # is already fp32 + contiguous, `.to(fp32).contiguous()` would
    # return self itself, and the kernel's atomic_add would mutate
    # the caller's tensor across calls.
    work = self.clone().to(dtype=torch.float32).contiguous()

    src_c = src.contiguous()
    idx_c = index.contiguous()

    _kernel.launch(src_c, idx_c, work, dim)

    out.copy_(work.to(dtype=self.dtype))
    return out
