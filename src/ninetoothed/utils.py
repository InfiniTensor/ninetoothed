from contextlib import contextmanager

import triton


@contextmanager
def device_scope(device):
    """Select a device through the active Triton driver and restore it afterward."""
    driver = triton.runtime.driver.active
    previous = driver.get_current_device()
    index = device if isinstance(device, int) else getattr(device, "index", None)
    changed = index is not None and index != previous

    if changed:
        driver.set_current_device(index)

    try:
        yield
    finally:
        if changed:
            driver.set_current_device(previous)


def get_device_properties(device=None):
    """Normalize device limits through the active driver's runtime interface."""
    driver = triton.runtime.driver.active

    with device_scope(device):
        properties = dict(
            driver.utils.get_device_properties(driver.get_current_device())
        )
        target = driver.get_current_target()
        interface = getattr(driver, "get_device_interface", lambda: None)()
        query = getattr(interface, "get_device_properties", None)

        if query is not None:
            runtime_properties = query(driver.get_current_device())

            for name, attribute in (
                ("max_threads_per_block", "max_threads_per_block"),
                ("max_threads_per_sm", "max_threads_per_multi_processor"),
                ("registers_per_sm", "regs_per_multiprocessor"),
                ("multiprocessor_count", "multi_processor_count"),
                ("name", "name"),
            ):
                value = getattr(runtime_properties, attribute, None)

                if value is not None:
                    properties[name] = value

    return properties | {
        "backend": target.backend,
        "arch": target.arch,
        "warp_size": target.warp_size,
    }


def calculate_default_configs():
    properties = get_device_properties()
    max_shared_mem = properties["max_shared_mem"]

    num_warps = 8
    num_stages = max_shared_mem // 2**15

    return num_warps, num_stages
