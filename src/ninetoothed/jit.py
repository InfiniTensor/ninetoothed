import importlib
import sys
import torch
import os 

from ninetoothed.generation import CodeGenerator
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

    default_num_warps, default_num_stages = calculate_default_configs()

    if num_warps is None:
        num_warps = default_num_warps

    if num_stages is None:
        num_stages = default_num_stages

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
        
        handle = _Handle(
            source_file,
            code_generator,
        )

        return handle


def import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def get_target_device(*args, **kwargs):
    target_device = None
    for arg in args:
        if isinstance(arg, torch.Tensor):
            target_device = arg.device
            break
    
    if target_device is None:
        for val in kwargs.values():
            if isinstance(val, torch.Tensor):
                target_device = val.device
                break
                
    return target_device


def convert_to_cpu(source_file_path):
    if not os.path.exists(source_file_path):
        raise FileNotFoundError(f"源文件不存在: {source_file_path}")

    dir_name = os.path.dirname(source_file_path)
    base_name = os.path.basename(source_file_path)
    name, ext = os.path.splitext(base_name)
    
    new_file_name = f"{name}_cpu{ext}"
    new_file_path = os.path.join(dir_name, new_file_name)

    with open(source_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    new_content = content.replace("triton", "triton_cpu")
    with open(new_file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    return new_file_path


class _Handle:
    def __init__(self, source, code_generator):
        self._source = source
        self._code_generator = code_generator

    def __call__(self, *args, **kwargs):
        target_device = get_target_device(*args, **kwargs)

        if target_device is not None and str(target_device) == "cpu":
            cpu_path=convert_to_cpu(self._source)
            self._source = cpu_path

        module = import_from_path(self._source, self._source)
        module_vars = vars(module)
        self._launch = module_vars[self._code_generator.launch_func_name]

        return self._launch(*args, **kwargs)