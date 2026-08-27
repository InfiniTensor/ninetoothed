import contextlib
import importlib.util
import os
import shutil

import pytest
import torch


def get_available_devices():
    devices = []

    if torch.cuda.is_available():
        devices.append("cuda")

    if hasattr(torch, "mlu") and torch.mlu.is_available():
        devices.append("mlu")

    return tuple(devices)


def _platform_supports_backend(backend: str) -> bool | None:
    """Whether the active platform profile declares ``backend``.

    Returns ``None`` when no platform is pinned so callers can fall back to
    toolchain probing.
    """
    platform_name = os.environ.get("NINETOOTHED_PLATFORM")

    if not platform_name:
        return None

    with contextlib.suppress(Exception):
        from ninetoothed.targets import default_platform_registry

        profile = default_platform_registry().get(platform_name)

        return backend in profile.backend_modes
    return None


def _cambricon_platform_available():
    """MLU device, local neuware toolchain, or a configured remote cncc host."""
    if hasattr(torch, "mlu") and torch.mlu.is_available():
        return True

    if any(os.environ.get(name) for name in ("NEUWARE_HOME", "NEUWARE_ROOT")):
        return True

    if os.environ.get("NINETOOTHED_BANGC_COMPILER") or shutil.which("cncc"):
        return True

    return bool(os.environ.get("NINETOOTHED_BANGC_SSH"))


def _cuda_platform_available():
    """CUDA device or an nvcc/cucc toolchain.

    CUDA runs on most platforms, so the probe accepts either the toolkit or
    a CUDA-capable torch build; it only returns False on hosts that have
    neither (e.g. a pure Cambricon container without a CUDA cross-compiler).
    """
    if torch.cuda.is_available():
        return True

    if any(os.environ.get(name) for name in ("CUDA_HOME", "CUDA_PATH", "CUCC_PATH")):
        return True

    return shutil.which("nvcc") is not None or shutil.which("cucc") is not None


# Backends whose tests only run on the platform providing their toolchain.
# This table is the single source of truth for platform gating: adding a
# hardware backend means adding one probe entry here (e.g.
# `ascendc`: `_ascend_platform_available`).  Backends without an entry are
# considered available everywhere because their tests are either source-only
# (hermetic emission checks) or parameterized by `get_available_devices`
# at runtime.
_BACKEND_PLATFORM_PROBES = {
    "bangc": _cambricon_platform_available,
    "cuda": _cuda_platform_available,
}


def backend_platform_available(backend) -> bool:
    """Whether this host can run the tests of ``backend``.

    The check has two layers: the pinned platform profile (via
    ``NINETOOTHED_PLATFORM``) must declare the backend, and — for
    hardware-bound backends — the host must actually reach their toolchain.
    """
    backend = str(backend)

    declared = _platform_supports_backend(backend)

    if declared is False:
        return False

    probe = _BACKEND_PLATFORM_PROBES.get(backend)

    return True if probe is None else probe()


def requires_backend(backend):
    """Pytest mark that skips a test off the backend's native platform."""
    return pytest.mark.skipif(
        not backend_platform_available(backend),
        reason=f"`{backend}` tests require their native platform; the probe "
        "lives in tests/utils.py::_BACKEND_PLATFORM_PROBES.",
    )


def _nvcc_available() -> bool:
    try:
        from ninetoothed.backends.toolchain import find_nvcc

        find_nvcc()

        return True
    except Exception:
        return False


# End-to-end materialization tests (AOT builds, artifact reload) need the
# backend's full toolchain, not just source emission.  This table captures
# those extra requirements; entries default to the platform probe.
_BACKEND_RUNTIME_PROBES = {
    "triton": lambda: (
        importlib.util.find_spec("triton") is not None and _nvcc_available()
    ),
    "tilelang": lambda: importlib.util.find_spec("tilelang") is not None,
}


def backend_runtime_available(backend) -> bool:
    """Whether materializing ``backend`` artifacts works on this host."""
    if not backend_platform_available(backend):
        return False

    probe = _BACKEND_RUNTIME_PROBES.get(str(backend))

    return True if probe is None else probe()


def detect_default_backend() -> str:
    """Pick a backend whose full toolchain works on this host.

    Tests that do not pin a backend explicitly inherit ``NINETOOTHED_BACKEND``
    (triton by default).  On hosts without that default's toolchain — e.g.
    a Cambricon-only container — this picks the first backend that can
    actually run, so plain ``pytest`` invocations do not fail on unrelated
    platforms.
    """
    for backend in ("triton", "cuda", "tilelang", "bangc"):
        if backend_runtime_available(backend):
            return backend
    return "triton"


with contextlib.suppress(ImportError, ModuleNotFoundError):
    import torch_mlu  # noqa: F401
