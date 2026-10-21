"""Plan disjoint CPU sets for accelerator workers and bind their main process."""

import os


def plan_cpu_affinity(devices, policy="auto", minimum=1):
    """Divide allowed CPUs among devices after reserving explicit assignments."""
    devices = list(devices)

    if (
        not devices
        or any(type(device) is not int or device < 0 for device in devices)
        or len(set(devices)) != len(devices)
    ):
        raise ValueError("CPU affinity requires distinct device identifiers.")

    if type(minimum) is not int or minimum < 1:
        raise ValueError("The minimum CPU count must be a positive integer.")

    if policy != "auto" and not isinstance(policy, dict):
        raise ValueError("CPU affinity must be 'auto' or a per-device mapping.")

    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        if policy != "auto":
            raise RuntimeError("Explicit CPU affinity requires Linux scheduling APIs.")

        return {device: None for device in devices}

    allowed = set(os.sched_getaffinity(0))
    assignments = {}
    reserved = set()

    for device in devices:
        cpus = (
            policy.get(str(device), policy.get(device))
            if isinstance(policy, dict)
            else None
        )

        if cpus is None:
            continue

        if (
            not isinstance(cpus, (list, tuple))
            or len(cpus) < minimum
            or any(type(cpu) is not int or cpu < 0 for cpu in cpus)
            or len(set(cpus)) != len(cpus)
        ):
            raise ValueError(
                f"Device {device} needs at least {minimum} distinct CPU identifiers."
            )

        selected = set(cpus)

        if not selected <= allowed:
            raise ValueError(
                f"Device {device} requests CPUs outside the allowed affinity."
            )

        if selected & reserved:
            raise ValueError("CPU affinity assignments for different devices overlap.")

        assignments[device] = sorted(selected)
        reserved.update(selected)

    automatic = [device for device in devices if device not in assignments]
    remaining = sorted(allowed - reserved)

    if automatic:
        if len(remaining) < minimum * len(automatic):
            raise ValueError("Not enough allowed CPUs for disjoint device assignments.")

        width, extra = divmod(len(remaining), len(automatic))
        offset = 0

        for index, device in enumerate(automatic):
            count = width + (index < extra)
            assignments[device] = remaining[offset : offset + count]
            offset += count

    return {device: assignments[device] for device in devices}


def apply_cpu_affinity(cpus):
    """Apply an explicit CPU set, or only read the current set when passed None."""
    if cpus is not None:
        if not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
            raise RuntimeError("Explicit CPU affinity requires Linux scheduling APIs.")

        if not cpus or any(type(cpu) is not int or cpu < 0 for cpu in cpus):
            raise ValueError("CPU affinity must contain nonnegative CPU identifiers.")

        selected = set(cpus)

        if not selected <= set(os.sched_getaffinity(0)):
            raise ValueError(
                "Requested CPUs are outside the worker's allowed affinity."
            )

        os.sched_setaffinity(0, selected)

        if set(os.sched_getaffinity(0)) != selected:
            raise RuntimeError("The requested CPU affinity was not applied exactly.")

    return sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
