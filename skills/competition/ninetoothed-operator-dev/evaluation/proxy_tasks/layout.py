"""Layout-sensitive proxy tasks (6): train 4, holdout 2.

Holdout (pixel_unshuffle / pixel_shuffle) uses operators not in train, so
passing holdout demonstrates the skill generalises beyond the tuning set.
"""

from __future__ import annotations

from schema import TaskSpec


def _ref_contig(x):
    return x.contiguous()


def _ref_flip_last(x):
    import torch

    return torch.flip(x, dims=[-1])


def _ref_narrow_half(x):
    return x[..., : x.shape[-1] // 2].contiguous()


def _ref_strided_gather(x):
    return x[:, ::2].contiguous()


def _ref_pixel_unshuffle(x):
    import torch

    return torch.nn.functional.pixel_unshuffle(x, downscale_factor=2)


def _ref_pixel_shuffle(x):
    import torch

    return torch.nn.functional.pixel_shuffle(x, upscale_factor=2)


def _in_noncontig_transpose(device="cpu", dtype="float32"):
    import torch

    dt = getattr(torch, dtype)
    x = torch.randn(128, 64, device=device, dtype=dt)

    return (x.t(),)  # Non-contiguous (64, 128) view.


def _in_contig_2d(device="cpu", dtype="float32"):
    import torch

    dt = getattr(torch, dtype)

    return (torch.randn(64, 256, device=device, dtype=dt),)


def _in_nchw(device="cpu", dtype="float32"):
    import torch

    dt = getattr(torch, dtype)

    return (torch.randn(2, 4, 16, 16, device=device, dtype=dt),)


def _in_nchw_depth(device="cpu", dtype="float32"):
    import torch

    dt = getattr(torch, dtype)
    # (B, C*r*r, H, W) for pixel_shuffle with r=2.

    return (torch.randn(2, 16, 8, 8, device=device, dtype=dt),)


TASKS = [
    TaskSpec(
        id="ly01",
        category="layout",
        split="train",
        difficulty="medium",
        name="contig_transpose",
        kind="operator",
        prompt="把一个转置后的非连续输入物化为连续张量（数值不变，存储变连续）。输入是 x.t()。",
        reference=_ref_contig,
        make_inputs=_in_noncontig_transpose,
    ),
    TaskSpec(
        id="ly02",
        category="layout",
        split="train",
        difficulty="medium",
        name="flip_last",
        kind="operator",
        prompt="沿最后一维翻转：out[..., i] = x[..., N-1-i]。",
        reference=_ref_flip_last,
        make_inputs=_in_contig_2d,
    ),
    TaskSpec(
        id="ly03",
        category="layout",
        split="train",
        difficulty="medium",
        name="narrow_half",
        kind="operator",
        prompt="取最后一维前半段 x[..., :N//2] 并物化为连续张量。",
        reference=_ref_narrow_half,
        make_inputs=_in_contig_2d,
    ),
    TaskSpec(
        id="ly04",
        category="layout",
        split="train",
        difficulty="hard",
        name="strided_gather",
        kind="operator",
        prompt="从带步长的视图 x[:, ::2]（每隔一列取一列）物化为连续张量。",
        reference=_ref_strided_gather,
        make_inputs=_in_contig_2d,
    ),
    TaskSpec(
        id="ly05",
        category="layout",
        split="holdout",
        difficulty="hard",
        name="pixel_unshuffle",
        kind="operator",
        prompt="实现 pixel_unshuffle（space-to-depth），下采样因子 r=2，输入 (B,C,H,W)，需支持非连续输入。",
        reference=_ref_pixel_unshuffle,
        make_inputs=_in_nchw,
    ),
    TaskSpec(
        id="ly06",
        category="layout",
        split="holdout",
        difficulty="hard",
        name="pixel_shuffle",
        kind="operator",
        prompt="实现 pixel_shuffle（depth-to-space），上采样因子 r=2，输入 (B, C*r*r, H, W)。",
        reference=_ref_pixel_shuffle,
        make_inputs=_in_nchw_depth,
    ),
]
