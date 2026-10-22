"""Overflow-free positive-domain task arithmetic for BangC schedules."""

from ninetoothed.backends.emitters.task_launch import allocate_private_symbol


def ceil_divide(total, divisor):
    """Avoid the overflowing numerator in ``(total + divisor - 1) / divisor``."""
    return f"(({total}) / ({divisor}) + (({total}) % ({divisor}) != 0))"


def allocate_chunk_parameter(*fragments):
    """Allocate the host-bounded argument outside the public launch ABI."""
    return allocate_private_symbol("nt_launch_chunk", *fragments)
