import concurrent.futures
import functools
import inspect
import itertools
import multiprocessing
import pathlib

import ninetoothed
from ninetoothed.aot import (
    _DTYPE_MAPPING,
    _HEADER_PATH,
    _MACRO_MAPPING,
    _generate_launch_func,
)
from ninetoothed.auto_tuner import AutoTuner
from ninetoothed.tensor import Symbol


def build(
    premake,
    configs,
    *,
    meta_parameters=None,
    caller=None,
    kernel_name=None,
    output_dir=None,
):
    """Build a kernel from a ``premake`` function and ``configs``.

    :param premake: A callable that returns the ``arrangement``,
        ``application``, and ``tensors`` for a given configuration.
    :param configs: An iterable of configurations where each
        configuration is a tuple of
        ``(args, kwargs, compilation_configs)``.
        ``args`` and ``kwargs`` are passed to ``premake``, and
        ``compilation_configs`` contains compilation configurations for
        ``ninetoothed.make`` (e.g., ``num_warps`` and ``num_stages``).
    :param meta_parameters: An iterable of meta-parameters that should
        be auto-tuned.
    :param caller: Who will call the compute kernel.
    :param kernel_name: The name for the generated kernel.
    :param output_dir: The directory to store the generated files.
    """

    if caller is None:
        caller = "cuda"

    output_dir = pathlib.Path(output_dir)

    headers = []
    all_param_names = []
    combinations = []
    launches = []
    all_tensors = []

    with concurrent.futures.ProcessPoolExecutor(
        mp_context=multiprocessing.get_context("spawn")
    ) as executor:
        futures = []

        for config in configs:
            future = executor.submit(
                _make,
                premake,
                config,
                caller=caller,
                kernel_name=kernel_name,
                output_dir=output_dir,
            )

            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            header, param_names, combination, launch, tensors = future.result()

            headers.append(header)
            all_param_names.append(param_names)
            combinations.append(combination)
            launches.append(launch)
            all_tensors.append(tensors)

    includes = "\n".join(f'#include "{header}"' for header in headers)

    param_names = list(
        functools.reduce(
            lambda x, y: dict.fromkeys(x) | dict.fromkeys(y),
            sorted(all_param_names, key=len, reverse=True),
            {},
        )
    )
    param_types = [
        "NineToothedStream",
    ] + ["NineToothedTensor" for _ in range(len(param_names) - 1)]

    for param_name in functools.reduce(lambda x, y: x | y, combinations, {}):
        param_names.append(param_name)
        param_types.append("int")

    param_decls = ", ".join(
        f"{type} {param}" for param, type in zip(param_names, param_types)
    )

    source_file_name = f"{kernel_name}.cpp"
    header_file_name = f"{kernel_name}.h"

    func_sig = f"NineToothedResult launch_{kernel_name}({param_decls})"

    joined_launches = "\n".join(launches)

    op_decl = f'#ifdef __cplusplus\nextern "C" {func_sig};\n#else\n{func_sig};\n#endif'
    op_def = f"""extern "C" {func_sig} {{
{joined_launches}
    return 1;
}}"""

    source_content = f"""#include "{header_file_name}"

{includes}\n\n{op_def}\n"""
    header_content = f"""#include "{_HEADER_PATH}"
\n{op_decl}\n"""

    (output_dir / source_file_name).write_text(source_content)
    (output_dir / header_file_name).write_text(header_content)

    kernel = _generate_launch_func(kernel_name=kernel_name, output_dir=output_dir)

    if meta_parameters is not None:
        _auto_tune(
            kernel,
            configs,
            all_tensors,
            meta_parameters,
            caller=caller,
            kernel_name=kernel_name,
            output_dir=output_dir,
        )

    return kernel


_DEFAULT_SIZES = tuple(2**i for i in range(10))


class _MetaTensor:
    def __init__(self, shape, dtype):
        self.shape = []

        for size in shape:
            if isinstance(size, Symbol):
                self.shape.append(None)
            else:
                self.shape.append(size)

        self.dtype = dtype


def _auto_tune(
    kernel, configs, all_tensors, meta_parameters, *, caller, kernel_name, output_dir
):
    key = str(output_dir / kernel_name)

    auto_tuner = AutoTuner(funcs=(kernel,), keys=(key,))

    for config, tensors in zip(configs, all_tensors):
        _warm_up(auto_tuner, config, tensors, caller=caller)

    config_to_all_meta_parameters = {}

    for config in configs:
        args, kwargs, compilation_configs = config

        meta_params = {
            param: kwargs[param] for param in meta_parameters if param in kwargs
        } | compilation_configs

        kwargs_ = {
            key: value for key, value in kwargs.items() if key not in meta_params
        }

        config_ = (args, tuple(kwargs_))

        if config_ not in config_to_all_meta_parameters:
            config_to_all_meta_parameters[config_] = []

        config_to_all_meta_parameters[config_].append(meta_params)


def _warm_up(kernel, config, meta_tensors, *, caller):
    import torch

    dtype_mapping = {
        ninetoothed.int8: torch.int8,
        ninetoothed.int16: torch.int16,
        ninetoothed.int32: torch.int32,
        ninetoothed.int64: torch.int64,
        ninetoothed.uint8: torch.uint8,
        ninetoothed.uint16: torch.uint16,
        ninetoothed.uint32: torch.uint32,
        ninetoothed.uint64: torch.uint64,
        ninetoothed.float16: torch.float16,
        ninetoothed.bfloat16: torch.bfloat16,
        ninetoothed.float32: torch.float32,
        ninetoothed.float64: torch.float64,
    }

    args, kwargs, compilation_configs = config

    all_shapes = []

    for meta_tensor in meta_tensors:
        all_sizes = []

        for size in meta_tensor.shape:
            if size is None:
                all_sizes.append(_DEFAULT_SIZES)
            else:
                all_sizes.append((size,))

        shapes = tuple(itertools.product(*all_sizes))

        all_shapes.append(shapes)

    for shapes in tuple(itertools.product(*all_shapes)):
        tensors = []

        for meta_tensor, shape in zip(meta_tensors, shapes):
            dtype = dtype_mapping[meta_tensor.dtype]

            if len(shape) == 0:
                device = None
            else:
                device = caller

            tensor = torch.empty(shape, dtype=dtype, device=device)

            tensors.append(tensor)

        kernel(*tensors, *args, *kwargs.values(), *compilation_configs.values())


def _make(premake, config, caller, kernel_name, output_dir):
    args, kwargs, compilation_configs = config

    arrangement, application, tensors = premake(*args, **kwargs)

    premake_signature = inspect.signature(premake)
    bound_arguments = premake_signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()
    combination = bound_arguments.arguments
    combination = {f"{name}_": value for name, value in combination.items()}
    combination |= compilation_configs

    for name, value in combination.items():
        if isinstance(value, bool):
            combination[name] = _MACRO_MAPPING[value][0]

        if value in _DTYPE_MAPPING:
            combination[name] = _DTYPE_MAPPING[value]

    kernel_name_ = f"{kernel_name}_{_generate_suffix(combination.values())}"

    ninetoothed.make(
        arrangement,
        application,
        tensors,
        caller=caller,
        kernel_name=kernel_name_,
        output_dir=output_dir,
        **compilation_configs,
    )

    header = output_dir / f"{kernel_name_}.h"
    application_signature = inspect.signature(application)
    param_names = ("stream",) + tuple(application_signature.parameters.keys())
    launch = f"""    if ({_generate_condition(combination)})
        return launch_{kernel_name_}({", ".join(param_names)});"""
    tensors = tuple(
        _MetaTensor(shape=tensor.shape, dtype=tensor.dtype) for tensor in tensors
    )

    return header, param_names, combination, launch, tensors


def _generate_condition(combination):
    return " && ".join(f"{param} == {value}" for param, value in combination.items())


def _generate_suffix(values):
    return "_".join(f"{value}" for value in values)
