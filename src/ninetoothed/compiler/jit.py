"""JIT compilation interface for NineToothed applications."""

from ninetoothed.compiler.driver import DEFAULT_COMPILER, CompileRequest


def jit(
    func=None,
    *,
    backend=None,
    platform=None,
    compute_arch=None,
    caller="torch",
    kernel_name=None,
    num_warps=None,
    num_stages=None,
    max_num_configs=None,
    pipeline=None,
    pass_options=None,
    _prettify=False,
    **backend_options,
):
    """Compile an annotated application through SSA as a decorator or function."""
    if _prettify:
        raise NotImplementedError(
            "The legacy `_prettify` source rewrite is not supported by SSA emitters."
        )

    def wrapper(application):
        return DEFAULT_COMPILER.materialize(
            CompileRequest(
                application=application,
                backend=backend,
                platform=platform,
                compute_arch=compute_arch,
                caller=caller,
                kernel_name=kernel_name or application.__name__,
                num_warps=num_warps,
                num_stages=num_stages,
                max_num_configs=max_num_configs,
                pipeline=pipeline,
                pass_options=pass_options,
                backend_options=backend_options,
            )
        )

    return wrapper if func is None else wrapper(func)


__all__ = ["jit"]
