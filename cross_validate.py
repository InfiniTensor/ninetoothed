#!/usr/bin/env python3
"""Compare a CPU interpretation against a real Triton/CUDA execution.

This is the GPU half of the differential validation.  It runs the same
arrangement and application through the CPU reference interpreter and the real
Triton kernel, then diffs the two with the interpreter's own differ.

It needs a CUDA device and ``torch``, so it is a script and not a test; the
GPU-free suite in ``tests/`` covers the rest.

Usage::

    python cross_validate.py

A clean run prints ``MATCH`` and a ``max |cpu - numpy|`` around ``1e-8``.
"""

from __future__ import annotations

import dataclasses

import numpy as np

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.interpret import compare_interpretations, interpret

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - CPU-only environments
    torch = None

#: The tolerance the project uses for `float32` against a NumPy/PyTorch
#: reference.  Pass `rtol=1e-6, atol=1e-6` to see how much margin there is.
RTOL = 1e-3
ATOL = 1e-3

ROWS = 3
WIDTH = 11
BLOCK = 16


def arrangement(x, out):
    """Tile each row of the input into one program instance.

    The block width is a plain constant instead of a ``block_size()`` meta
    parameter.  A meta parameter is resolved at compile time under an
    auto-generated internal name, so there is no supported way to pin it to the
    same value on both sides of the comparison.  A mismatched block width would
    tile the rows differently and make the two runs incomparable.
    """
    return x.tile((1, BLOCK)), out.tile((1, BLOCK))


def application(x, out):
    """Row-wise softmax, masked with `-inf` so the tail drops out."""
    shifted = x - ntl.max(x, axis=1)[:, None]
    numerator = ntl.exp(shifted)
    out = numerator / ntl.sum(numerator, axis=1)[:, None]  # noqa: F841


def numpy_reference(x):
    """Return a second, independent reference to catch a shared misunderstanding."""
    shifted = x - x.max(axis=1, keepdims=True)
    numerator = np.exp(shifted)

    return numerator / numerator.sum(axis=1, keepdims=True)


def main():
    if torch is None:
        raise SystemExit(
            "This script needs PyTorch with CUDA. Install it on the GPU server, or "
            "run `python -m pytest tests/test_interpret.py` locally instead."
        )

    if not torch.cuda.is_available():
        raise SystemExit(
            "No CUDA device is visible. Run this on the GPU server, or use "
            "`python -m pytest tests/test_interpret.py` locally instead."
        )

    print("device        :", torch.cuda.get_device_name(0))

    cpu_x = np.random.default_rng(0).random((ROWS, WIDTH), dtype=np.float32)
    cpu_out = np.zeros_like(cpu_x)

    # -- CPU reference, using the interpreter ------------------------------
    reference = interpret(
        arrangement,
        application,
        tensors=(Tensor(2, other=float("-inf")), Tensor(2)),
        inputs=(cpu_x, cpu_out),
    )

    # -- GPU execution, using the same arrangement and application ---------
    kernel = ninetoothed.make(
        arrangement,
        application,
        (Tensor(2, other=float("-inf")), Tensor(2)),
    )
    gpu_x = torch.from_numpy(cpu_x).cuda()
    gpu_out = torch.zeros((ROWS, WIDTH), dtype=torch.float32, device="cuda")
    kernel(gpu_x, gpu_out)
    gpu_result = gpu_out.cpu().numpy()

    # -- Compare, using the interpreter's own differ -----------------------
    # `Interpretation` is a dataclass, so the GPU output can be substituted into
    # a copy of the CPU run and the two diffed like any other pair.
    on_gpu = dataclasses.replace(reference, outputs={"out": gpu_result})
    diff = compare_interpretations(
        reference, on_gpu, rtol=RTOL, atol=ATOL, label="cpu vs gpu"
    )
    print(diff.render())

    if not diff.matches:
        print(diff.to_json())
        print(reference.render_trace(limit=40))

        raise SystemExit(1)

    error = float(np.abs(reference.output("out") - numpy_reference(cpu_x)).max())
    print("max |cpu - numpy| =", error)
    print("MATCH: the GPU kernel agrees with the CPU reference")


if __name__ == "__main__":
    main()
