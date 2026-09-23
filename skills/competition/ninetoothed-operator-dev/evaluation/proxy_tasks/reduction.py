"""Reduction / blocking proxy tasks (6): train 4, holdout 2."""

from __future__ import annotations

from schema import TaskSpec, randn_inputs


def _ref_sum_last(x):
    return x.sum(dim=-1)


def _ref_mean_last(x):
    return x.mean(dim=-1)


def _ref_softmax(x):
    import torch

    return torch.softmax(x.float(), dim=-1).to(x.dtype)


def _ref_rms_norm(x):
    import torch

    var = x.float().pow(2).mean(dim=-1, keepdim=True)

    return (x.float() * torch.rsqrt(var + 1e-6)).to(x.dtype)


def _ref_max_last(x):
    return x.max(dim=-1).values


def _ref_l2_norm(x):
    import torch

    return torch.sqrt(x.float().pow(2).sum(dim=-1)).to(x.dtype)


TASKS = [
    TaskSpec(
        id="rd01",
        category="reduction",
        split="train",
        difficulty="easy",
        name="sum_last",
        kind="operator",
        prompt="对最后一维求和：输入 (M,N)，输出 (M,)。fp16 输入需在 fp32 中累积。",
        reference=_ref_sum_last,
        make_inputs=randn_inputs((128, 1024)),
    ),
    TaskSpec(
        id="rd02",
        category="reduction",
        split="train",
        difficulty="easy",
        name="mean_last",
        kind="operator",
        prompt="对最后一维求均值：输入 (M,N)，输出 (M,)。fp16 输入需 fp32 累积后再除以 N。",
        reference=_ref_mean_last,
        make_inputs=randn_inputs((128, 1024)),
    ),
    TaskSpec(
        id="rd03",
        category="reduction",
        split="train",
        difficulty="medium",
        name="softmax",
        kind="operator",
        prompt="行 softmax：输入 (M,N)，对最后一维归一化。必须先减去行最大值以保证数值稳定。",
        reference=_ref_softmax,
        make_inputs=randn_inputs((64, 2048)),
    ),
    TaskSpec(
        id="rd04",
        category="reduction",
        split="train",
        difficulty="hard",
        name="rms_norm",
        kind="operator",
        prompt="RMS Norm，对最后一维归一化，eps=1e-6。中间统计量在 fp32 中累积。",
        reference=_ref_rms_norm,
        make_inputs=randn_inputs((64, 2048)),
    ),
    TaskSpec(
        id="rd05",
        category="reduction",
        split="holdout",
        difficulty="medium",
        name="max_last",
        kind="operator",
        prompt="对最后一维求最大值：输入 (M,N)，输出 (M,)。padding lane 用 -inf 填充。",
        reference=_ref_max_last,
        make_inputs=randn_inputs((96, 1023)),
    ),
    TaskSpec(
        id="rd06",
        category="reduction",
        split="holdout",
        difficulty="hard",
        name="l2_norm",
        kind="operator",
        prompt="对最后一维求 L2 范数：sqrt(sum(x^2))，输入 (M,N)，输出 (M,)。平方和在 fp32 中累积。",
        reference=_ref_l2_norm,
        make_inputs=randn_inputs((96, 1023)),
    ),
]
