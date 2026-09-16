"""Compatibility wrapper for the historical :mod:`ninetoothed.make` API."""

from ninetoothed.compiler.driver import make as _compiler_make


def make(
    arrangement,
    application,
    tensors,
    caller="torch",
    kernel_name=None,
    output_dir=None,
    num_warps=None,
    num_stages=None,
    max_num_configs=None,
    *,
    backend=None,
    platform=None,
    compute_arch=None,
    pipeline=None,
    pass_options=None,
    **backend_options,
):
    """Compile using SSA while preserving the legacy positional parameters."""
    return _compiler_make(
        arrangement,
        application,
        tensors,
        backend=backend,
        platform=platform,
        compute_arch=compute_arch,
        caller=caller,
        kernel_name=kernel_name,
        output_dir=output_dir,
        num_warps=num_warps,
        num_stages=num_stages,
        max_num_configs=max_num_configs,
        pipeline=pipeline,
        pass_options=pass_options,
        **backend_options,
    )


__all__ = ["make"]
