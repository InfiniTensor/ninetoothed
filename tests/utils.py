import torch


def get_available_devices():
    devices = []

    if torch.cuda.is_available():
        devices.append("cuda")

    return tuple(devices)
