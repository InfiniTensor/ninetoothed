import triton


def calculate_default_configs():
    num_warps = 8

    try:
        device = triton.runtime.driver.active.get_current_device()
        properties = triton.runtime.driver.active.utils.get_device_properties(device)
        max_shared_mem = properties["max_shared_mem"]
        num_stages = max(1, max_shared_mem // 2**15)
    except Exception:
        num_stages = 1

    return num_warps, num_stages
