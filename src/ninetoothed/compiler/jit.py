"""JIT compilation interface for NineToothed applications."""

from ninetoothed.compiler.driver import DEFAULT_COMPILER, CompileRequest


def jit(
    func=None,
    *,
    backend=None,
    caller="torch",
    kernel_name=None,
    num_warps=None,
    num_stages=None,
    max_num_configs=None,
    pipeline=None,
    pass_options=None,
    tuning=False,
    _prettify=False,
    **backend_options,
):
    """Compile an annotated application through SSA as a decorator or function."""

    def wrapper(application):
        return JIT(
            application,
            caller=caller,
            backend=backend,
            kernel_name=kernel_name,
            num_warps=num_warps,
            num_stages=num_stages,
            max_num_configs=max_num_configs,
            pipeline=pipeline,
            pass_options=pass_options,
            tuning=tuning,
            _prettify=_prettify,
            backend_options=backend_options,
        )()

    return wrapper if func is None else wrapper(func)


class JIT:
    """Deferred materialization object used by :func:`jit`."""

    def __init__(
        self,
        func,
        *,
        backend,
        caller,
        kernel_name,
        num_warps,
        num_stages,
        max_num_configs,
        pipeline,
        pass_options,
        tuning,
        _prettify=False,
        backend_options=None,
    ):
        self.func = func
        self._caller = caller
        self._backend = backend
        self._kernel_name = kernel_name or func.__name__
        self._num_warps = num_warps
        self._num_stages = num_stages
        self._max_num_configs = max_num_configs
        self._pipeline = pipeline
        self._pass_options = pass_options
        self._tuning = tuning
        self._prettify = _prettify
        self._backend_options = dict(backend_options or {})

    def __call__(self):
        return DEFAULT_COMPILER.materialize(
            CompileRequest(
                application=self.func,
                backend=self._backend,
                caller=self._caller,
                kernel_name=self._kernel_name,
                num_warps=self._num_warps,
                num_stages=self._num_stages,
                max_num_configs=self._max_num_configs,
                pipeline=self._pipeline,
                pass_options=self._pass_options,
                tuning=self._tuning,
                backend_options=self._backend_options | {"prettify": self._prettify},
            )
        )


__all__ = ["JIT", "jit"]
