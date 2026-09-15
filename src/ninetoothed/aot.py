"""Compatibility wrapper for the historical :mod:`ninetoothed.aot` API."""

from ninetoothed.compiler.driver import aot as _compiler_aot


def aot(
    func,
    caller="cuda",
    kernel_name=None,
    output_dir=None,
    num_warps=None,
    num_stages=None,
    *,
    backend=None,
    platform=None,
    compute_arch=None,
    pipeline=None,
    pass_options=None,
    **backend_options,
):
    """Compile an annotated application through the SSA AOT path."""
    return _compiler_aot(
        func,
        backend=backend,
        platform=platform,
        compute_arch=compute_arch,
        caller=caller,
        kernel_name=kernel_name,
        output_dir=output_dir,
        num_warps=num_warps,
        num_stages=num_stages,
        pipeline=pipeline,
        pass_options=pass_options,
        **backend_options,
    )


__all__ = ["aot"]
