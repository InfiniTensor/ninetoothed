import contextlib

import torch


def get_available_devices(*, backend=None):
    devices = []

    if torch.cuda.is_available():
        devices.append("cuda")

    if hasattr(torch, "mlu") and torch.mlu.is_available():
        devices.append("mlu")

    if backend is None or backend == "ascend":
        if hasattr(torch, "npu") and torch.npu.is_available():
            devices.append("npu")

    if backend == "ascend":
        devices = [device for device in devices if device == "npu"]

    return tuple(devices)


with contextlib.suppress(ImportError, ModuleNotFoundError):
    import torch_mlu  # noqa: F401


with contextlib.suppress(ImportError, ModuleNotFoundError):
    import torch_npu  # noqa: F401
