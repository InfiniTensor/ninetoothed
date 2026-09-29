"""Elementwise / broadcast proxy tasks (6): train 4, holdout 2."""

from __future__ import annotations

from schema import TaskSpec, randn_inputs


def _ref_add(a, b):
    return a + b


def _ref_mul(a, b):
    return a * b


def _ref_relu(x):
    import torch

    return torch.relu(x)


def _ref_gelu(x):
    import torch

    return torch.nn.functional.gelu(x, approximate="tanh")


def _ref_silu(x):
    import torch

    return torch.nn.functional.silu(x)


def _ref_masked_add(a, b, mask):
    import torch

    return torch.where(mask, a + b, a)


def _in_mul_broadcast(device="cpu", dtype="float32"):
    import torch

    dt = getattr(torch, dtype)

    return (
        torch.randn(64, 1, device=device, dtype=dt),
        torch.randn(1, 128, device=device, dtype=dt),
    )


def _in_masked_add(device="cpu", dtype="float32"):
    import torch

    dt = getattr(torch, dtype)
    a = torch.randn(32, 128, device=device, dtype=dt)
    b = torch.randn(32, 128, device=device, dtype=dt)
    mask = torch.randint(0, 2, (1, 128), device=device, dtype=torch.bool)

    return (a, b, mask)


TASKS = [
    TaskSpec(
        id="ew01",
        category="elementwise",
        split="train",
        difficulty="easy",
        name="add",
        kind="operator",
        prompt="用 NineToothed 实现一维逐元素加法 out = a + b，a 和 b 同形状一维张量，支持 fp32 与 fp16。",
        reference=_ref_add,
        make_inputs=randn_inputs((98432,), (98432,)),
    ),
    TaskSpec(
        id="ew02",
        category="elementwise",
        split="train",
        difficulty="easy",
        name="mul_broadcast",
        kind="operator",
        prompt="实现逐元素乘法 out = a * b，支持 a 形状 (M,1) 与 b 形状 (1,N) 广播到 (M,N)。",
        reference=_ref_mul,
        make_inputs=_in_mul_broadcast,
    ),
    TaskSpec(
        id="ew03",
        category="elementwise",
        split="train",
        difficulty="medium",
        name="relu",
        kind="operator",
        prompt="实现 ReLU：out = max(x, 0)，逐元素，支持 fp32 与 fp16。",
        reference=_ref_relu,
        make_inputs=randn_inputs((4096, 1024)),
    ),
    TaskSpec(
        id="ew04",
        category="elementwise",
        split="train",
        difficulty="medium",
        name="gelu",
        kind="operator",
        prompt="实现 GELU（tanh 近似），逐元素。注意 fp16 下的数值表现。",
        reference=_ref_gelu,
        make_inputs=randn_inputs((4096, 1024)),
    ),
    TaskSpec(
        id="ew05",
        category="elementwise",
        split="holdout",
        difficulty="medium",
        name="silu",
        kind="operator",
        prompt="实现 SiLU：out = x * sigmoid(x)，逐元素，支持 fp32 与 fp16。",
        reference=_ref_silu,
        make_inputs=randn_inputs((4095, 1024)),
    ),
    TaskSpec(
        id="ew06",
        category="elementwise",
        split="holdout",
        difficulty="hard",
        name="masked_add",
        kind="operator",
        prompt="实现带掩码加法：mask 为真处 out = a + b，否则 out = a。mask 形状 (1,N) 广播到 (M,N)。",
        reference=_ref_masked_add,
        make_inputs=_in_masked_add,
    ),
]
