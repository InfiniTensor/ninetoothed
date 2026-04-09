import importlib
import sys

from ninetoothed.ir.pipeline import IRPipeline
from ninetoothed.utils import calculate_default_configs


def jit(
    func=None,
    *,
    caller="torch",
    kernel_name=None,
    num_warps=None,
    num_stages=None,
    max_num_configs=None,
    _prettify=False,
    _dump_ir=False,
):
    """A decorator for generating compute kernels.

    :param func: The function to be compiled.
    :param caller: Who will call the compute kernel.
    :param kernel_name: The name for the generated kernel.
    :param num_warps: The number of warps to use.
    :param num_stages: The number of pipeline stages.
    :param max_num_configs: The maximum number of auto-tuning
        configurations to use.
    :param _prettify: Whether to prettify the generated code.
    :param _dump_ir: Whether to print IR dump for each layer.
    :return: A handle to the compute kernel.

    """

    def wrapper(func):
        nonlocal num_warps, num_stages

        if num_warps is None or num_stages is None:
            try:
                default_num_warps, default_num_stages = calculate_default_configs()
            except Exception:
                default_num_warps, default_num_stages = (4, 8), (1, 2, 3, 4, 5)

            if num_warps is None:
                num_warps = default_num_warps

            if num_stages is None:
                num_stages = default_num_stages

        return JIT(
            func,
            caller=caller,
            kernel_name=kernel_name,
            num_warps=num_warps,
            num_stages=num_stages,
            max_num_configs=max_num_configs,
            _prettify=_prettify,
            _dump_ir=_dump_ir,
        )()

    if func is None:
        return wrapper

    return wrapper(func)


class JIT:
    def __init__(
        self,
        func,
        caller,
        kernel_name,
        num_warps,
        num_stages,
        max_num_configs,
        _prettify=False,
        _dump_ir=False,
    ):
        self.func = func

        self._caller = caller

        if kernel_name is not None:
            self._kernel_name = kernel_name
        else:
            self._kernel_name = func.__name__

        self._num_warps = num_warps

        self._num_stages = num_stages

        self._max_num_configs = max_num_configs

        self._prettify = _prettify

        self._dump_ir = _dump_ir

    def __call__(self):

        context = dict(self.func.__annotations__)
        args = list(context.values())

        pipeline = IRPipeline(
            context=context,
            args=args,
            caller=self._caller,
            kernel_name=self._kernel_name,
            num_warps=self._num_warps,
            num_stages=self._num_stages,
            max_num_configs=self._max_num_configs,
            prettify=self._prettify,
            dump_ir=self._dump_ir,
        )
        source_file = pipeline.run(self.func)

        module = import_from_path(source_file, source_file)
        module_vars = vars(module)

        handle = _Handle(
            module_vars[self._kernel_name],
            module_vars[code_generator.launch_func_name],
            source_file,
        )

        return handle


def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


class _Handle:
    def __init__(self, kernel, launch, source):
        self._kernel = kernel
        self._launch = launch
        self._source = source

    def __call__(self, *args, **kwargs):
        return self._launch(*args, **kwargs)
