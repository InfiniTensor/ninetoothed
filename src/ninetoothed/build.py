import concurrent.futures
import csv
import enum
import functools
import inspect
import itertools
import multiprocessing
import pathlib

import ninetoothed
from ninetoothed.aot import (
    _DTYPE_MAPPING,
    _HEADER_PATH,
    _INDENTATION,
    _MACRO_MAPPING,
    _generate_launch_func,
    _KernelLaunchError,
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

    kernel_names = []
    all_param_names = []
    combinations = []
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

        configs = []

        for future in concurrent.futures.as_completed(futures):
            kernel_name_, param_names, combination, config, tensors = future.result()

            kernel_names.append(kernel_name_)
            all_param_names.append(param_names)
            combinations.append(combination)
            configs.append(config)
            all_tensors.append(tensors)

    tensor_param_names = tuple(
        functools.reduce(
            lambda x, y: dict.fromkeys(x) | dict.fromkeys(y),
            sorted(all_param_names, key=len, reverse=True),
            {},
        )
    )
    tensor_param_types = tuple("NineToothedTensor" for _ in tensor_param_names)

    non_tensor_param_names = tuple(
        functools.reduce(lambda x, y: x | y, combinations, {})
    )
    non_tensor_param_types = tuple("int" for _ in non_tensor_param_names)

    param_names = ("stream",) + tensor_param_names + non_tensor_param_names
    param_types = ("NineToothedStream",) + tensor_param_types + non_tensor_param_types

    param_decls = _generate_declaration_expressions(param_types, param_names)

    headers = []
    launches = []

    for kernel_name_, param_names_, combination in zip(
        kernel_names, all_param_names, combinations
    ):
        header = f"{kernel_name_}.h"
        launch = f"""    if ({_generate_condition(combination)})
        return launch_{kernel_name_}({", ".join((param_names[0],) + param_names_)});"""

        headers.append(header)
        launches.append(launch)

    source_file_name = f"{kernel_name}.cpp"
    header_file_name = f"{kernel_name}.h"

    includes = "\n".join(f'#include "{header}"' for header in headers)

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
        config_to_best_meta_arguments = _auto_tune(
            kernel,
            configs,
            all_tensors,
            meta_parameters,
            caller=caller,
            kernel_name=kernel_name,
            output_dir=output_dir,
        )

        return _generate_kernel_with_auto_tuning(
            config_to_best_meta_arguments,
            non_tensor_param_names,
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


def _generate_kernel_with_auto_tuning(
    config_to_best_meta_arguments, non_tensor_param_names, *, kernel_name, output_dir
):
    num_non_meta_premake_params = len(tuple(config_to_best_meta_arguments.keys())[0])

    config_param_names = non_tensor_param_names[:num_non_meta_premake_params]
    meta_param_names = non_tensor_param_names[num_non_meta_premake_params:]
    meta_param_types = tuple("int" for _ in meta_param_names)

    csv_path = output_dir / f"{kernel_name}.csv"
    config_args = ", ".join(f"static_cast<int>({name})" for name in config_param_names)
    cache_line = (
        f'{_INDENTATION}static ninetoothed::AutoTuningCache cache{{"{csv_path}"}};'
    )
    lookup_line = f"{_INDENTATION}auto meta{{cache.lookup({{{config_args}}})}};"
    meta_assignments = "\n".join(
        f"{_INDENTATION}auto {name}{{meta[{i}]}};"
        for i, name in enumerate(meta_param_names)
    )
    meta_param_initialization = f"{cache_line}\n{lookup_line}\n{meta_assignments}"

    meta_param_decl_exprs = _generate_declaration_expressions(
        meta_param_types, meta_param_names
    )

    source_file_name = f"{kernel_name}.cpp"
    header_file_name = f"{kernel_name}.h"

    source_path = output_dir / source_file_name
    header_path = output_dir / header_file_name

    source_content = source_path.read_text().replace(
        f", {meta_param_decl_exprs}) {{", f") {{\n{meta_param_initialization}"
    )
    header_content = header_path.read_text().replace(f", {meta_param_decl_exprs}", "")

    source_path.write_text(source_content)
    header_path.write_text(header_content)

    return _generate_launch_func(kernel_name=kernel_name, output_dir=output_dir)


def _auto_tune(
    kernel, configs, all_tensors, meta_parameters, *, caller, kernel_name, output_dir
):
    config_to_all_meta_arguments = {}

    for config in configs:
        args, kwargs, compilation_configs = config

        meta_args = {
            param: kwargs[param] for param in meta_parameters if param in kwargs
        } | compilation_configs

        kwargs_ = {key: value for key, value in kwargs.items() if key not in meta_args}

        config_ = (args, tuple(kwargs_.items()))

        if config_ not in config_to_all_meta_arguments:
            config_to_all_meta_arguments[config_] = []

        config_to_all_meta_arguments[config_].append(meta_args)

    csv_path = output_dir / f"{kernel_name}.csv"
    cached = _read_auto_tuning_cache(csv_path)

    if cached is not None:
        return cached

    key = str(output_dir / kernel_name)

    auto_tuner = AutoTuner(funcs=(kernel,), keys=(key,))

    for config, tensors in zip(configs, all_tensors):
        _warm_up(auto_tuner, config, tensors, caller=caller)

    config_to_best_meta_arguments = {}

    for config, all_meta_arguments in config_to_all_meta_arguments.items():
        args, kwargs_items = config

        config_ = (*args, *(item[1] for item in kwargs_items))
        all_meta_arguments_ = tuple(
            tuple(meta_args.values()) for meta_args in all_meta_arguments
        )

        meta_args_to_timing = {}
        func_timings = auto_tuner._timings[key]

        for meta_args in all_meta_arguments_:
            arg_key = auto_tuner._make_arg_key((*config_, *meta_args), {})

            for full_arg_key, timing in func_timings.items():
                if not full_arg_key.endswith(arg_key):
                    continue

                meta_args_to_timing[meta_args] = timing

                break

        best_meta_arguments = sorted(
            meta_args_to_timing.items(), key=lambda item: item[1]
        )[0][0]

        int_configs = tuple(_arg_to_int(arg) for arg in config_)

        config_to_best_meta_arguments[int_configs] = best_meta_arguments

    _write_auto_tuning_cache(csv_path, config_to_best_meta_arguments)

    return config_to_best_meta_arguments


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

        try:
            kernel(*tensors, *args, *kwargs.values(), *compilation_configs.values())
        except _KernelLaunchError:
            pass


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
        combination[name] = _arg_to_int(value)

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

    application_signature = inspect.signature(application)
    param_names = tuple(application_signature.parameters.keys())
    tensors = tuple(
        _MetaTensor(shape=tensor.shape, dtype=tensor.dtype) for tensor in tensors
    )

    return kernel_name_, param_names, combination, config, tensors


def _read_auto_tuning_cache(path):
    if not path.exists():
        return None

    config_to_best_meta_arguments = {}

    with open(path) as f:
        rows = list(csv.reader(f))

    for i in range(0, len(rows), 2):
        config = tuple(int(value) for value in rows[i])
        meta_args = tuple(int(value) for value in rows[i + 1])

        config_to_best_meta_arguments[config] = meta_args

    return config_to_best_meta_arguments


def _write_auto_tuning_cache(path, config_to_best_meta_arguments):
    with open(path, "w") as f:
        csv.writer(f).writerows(
            itertools.chain.from_iterable(config_to_best_meta_arguments.items())
        )


def _generate_declaration_statements(types, names):
    return "\n".join(
        f"{_generate_declaration(type, name)};" for type, name in zip(types, names)
    )


def _generate_assignment_statements(names, values):
    return "\n".join(f"{name} = {value};" for name, value in zip(names, values))


def _generate_declaration_expressions(types, names):
    return ", ".join(
        _generate_declaration(type, name) for type, name in zip(types, names)
    )


def _generate_declaration(type, name):
    return f"{type} {name}"


def _generate_condition(combination):
    return " && ".join(f"{param} == {value}" for param, value in combination.items())


def _generate_suffix(values):
    return "_".join(f"{value}" for value in values)


def _arg_to_int(arg):
    if isinstance(arg, bool) or arg is None:
        return _MACRO_MAPPING[arg][1]

    if arg in _DTYPE_MAPPING:
        return tuple(_DTYPE_MAPPING.keys()).index(arg)

    if isinstance(arg, enum.Enum):
        return arg.value

    return arg
