import importlib
import sys

from ninetoothed.generation import CodeGenerator


def jit(func=None, *, caller="torch", _prettify=False):
    """A decorator for generating compute kernels.

    :param func: The function to be compiled.
    :param caller: Who will call the compute kernel.
    :param _prettify: Whether to prettify the generated code.
    :return: A handle to the compute kernel.

    .. note::

        The ``_prettify`` parameter is experimental, which might break
        the generated code.
    """

    def wrapper(func):
        return JIT(func, caller=caller, _prettify=_prettify)()

    if func is None:
        return wrapper

    return wrapper(func)


class JIT:
    def __init__(self, func, caller, _prettify=False):
        self.func = func

        self._caller = caller

        self._prettify = _prettify

    def __call__(self):
        code_generator = CodeGenerator()
        source_file = code_generator(self.func, self._caller, self._prettify)
        module = type(self)._import_from_path(source_file, source_file)
        module_vars = vars(module)

        handle = _Handle(
            module_vars[self.func.__name__],
            module_vars[f"launch_{self.func.__name__}"],
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
