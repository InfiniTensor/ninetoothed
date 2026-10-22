import contextlib

import torch


def get_available_devices():
    devices = []

    if torch.cuda.is_available():
        devices.append("cuda")

    if hasattr(torch, "mlu") and torch.mlu.is_available():
        devices.append("mlu")

    return tuple(devices)


def cuda_compiler_available():
    """Return whether an NVIDIA-compatible compiler can be located."""
    from ninetoothed.backends.toolchain import find_nvcc

    try:
        find_nvcc()
    except RuntimeError:
        return False

    return True


with contextlib.suppress(ImportError, ModuleNotFoundError):
    import torch_mlu  # noqa: F401
