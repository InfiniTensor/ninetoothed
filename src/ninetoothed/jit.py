import importlib
import sys

from ninetoothed.generation import CodeGenerator


def jit(
    func=None,
    *,
    caller="torch",
    kernel_name=None,
    num_warps=None,
    num_stages=None,
    max_num_configs=None,
    _prettify=False,
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
    :return: A handle to the compute kernel.

    .. note::

        The ``_prettify`` parameter is experimental, which might break
        the generated code.
    """

    def wrapper(func):
        return JIT(
            func,
            caller=caller,
            kernel_name=kernel_name,
            num_warps=num_warps,
            num_stages=num_stages,
            max_num_configs=max_num_configs,
            _prettify=_prettify,
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

    def __call__(self):
        code_generator = CodeGenerator()
        source_file = code_generator(
            self.func,
            self._caller,
            self._kernel_name,
            self._num_warps,
            self._num_stages,
            self._max_num_configs,
            self._prettify,
        )
        module = type(self)._import_from_path(source_file, source_file)
        module_vars = vars(module)

        handle = _Handle(
            module_vars[self._kernel_name],
            module_vars[code_generator.launch_func_name],
            source_file,
        )

        return handle

    @staticmethod
    def _import_from_path(module_name, file_path):
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
