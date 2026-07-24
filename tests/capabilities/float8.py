"""Probe vendor Triton float8 lowering without importing NineToothed."""

import functools
import subprocess

from tests.capabilities.model import CapabilityResult
from tests.capabilities.runner import run_python_probe

_PROBE_SOURCE = r"""
import importlib.util
import json
import tempfile
from pathlib import Path

DEVICE = None
PHASE = None

try:
    import torch

    if DEVICE == "mlu":
        importlib.import_module("torch_mlu")

    directory = tempfile.TemporaryDirectory()
    module_path = Path(directory.name) / "float8_capability_probe.py"
    module_path.write_text(
        '''import triton
import triton.language as tl


@triton.jit
def copy_as_float32(input, output, size: tl.constexpr):
    offsets = tl.arange(0, size)
    values = tl.load(input + offsets).to(tl.float32)
    tl.store(output + offsets, values)
''',
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location(
        "float8_capability_probe",
        module_path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the float8 capability probe module.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def run(dtype):
        source = torch.arange(32, device=DEVICE, dtype=torch.float16).to(dtype)
        output = torch.empty(32, device=DEVICE, dtype=torch.float32)
        module.copy_as_float32[(1,)](source, output, 32)
        getattr(torch, DEVICE).synchronize()
        torch.testing.assert_close(output, source.float())
except Exception as error:
    print(json.dumps({"status": "error", "reason": f"probe setup failed: {error}"}))
else:
    if PHASE == "control":
        try:
            run(torch.float16)
        except Exception as error:
            print(json.dumps({"status": "error", "reason": f"FP16 control failed: {error}"}))
        else:
            print(json.dumps({"status": "supported"}))
    elif PHASE == "float8":
        try:
            run(torch.float8_e5m2)
        except Exception as error:
            print(json.dumps({"status": "unavailable", "reason": str(error)}))
        else:
            print(json.dumps({"status": "supported"}))
    else:
        print(json.dumps({"status": "error", "reason": f"invalid probe phase: {PHASE!r}"}))
"""


def _source_for(device, phase):
    return _PROBE_SOURCE.replace(
        "DEVICE = None",
        f"DEVICE = {device!r}",
        1,
    ).replace(
        "PHASE = None",
        f"PHASE = {phase!r}",
        1,
    )


def _run_float8_e5m2_probe(device, timeout=60):
    control_name = f"triton {device} FP16 control"
    control = run_python_probe(
        control_name,
        _source_for(device, "control"),
        timeout=timeout,
    )

    if not control.supported:
        raise RuntimeError(
            f"Capability probe {control_name!r} failed: {control.reason}."
        )

    float8_name = f"triton {device} float8_e5m2"

    try:
        return run_python_probe(
            float8_name,
            _source_for(device, "float8"),
            timeout=timeout,
        )
    except RuntimeError as error:
        if not isinstance(error.__cause__, subprocess.TimeoutExpired):
            raise

        return CapabilityResult(
            supported=False,
            reason=f"{float8_name}: probe timed out after {timeout} seconds",
        )


@functools.cache
def float8_e5m2_cuda():
    return _run_float8_e5m2_probe("cuda")


@functools.cache
def float8_e5m2_mlu():
    return _run_float8_e5m2_probe("mlu")


__all__ = ["float8_e5m2_cuda", "float8_e5m2_mlu"]
