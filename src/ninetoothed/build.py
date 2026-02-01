import concurrent.futures
import functools
import inspect
import multiprocessing
import pathlib

import ninetoothed
from ninetoothed.aot import (
    _DTYPE_MAPPING,
    _HEADER_PATH,
    _MACRO_MAPPING,
    _generate_launch_func,
)


def build(premake, configs, *, caller=None, kernel_name=None, output_dir=None):
    """Build a kernel from a ``premake`` function and ``configs``.

    :param premake: A callable that returns the ``arrangement``,
        ``application``, and ``tensors`` for a given configuration.
    :param configs: An iterable of configurations where each
        configuration is a tuple of
        ``(args, kwargs, compilation_configs)``.
        ``args`` and ``kwargs`` are passed to ``premake``, and
        ``compilation_configs`` contains compilation configurations for
        ``ninetoothed.make`` (e.g., ``num_warps`` and ``num_stages``).
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
            kernel_name_, param_names, combination = future.result()

            kernel_names.append(kernel_name_)
            all_param_names.append(param_names)
            combinations.append(combination)

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

    param_decls = ", ".join(
        f"{type} {param}" for param, type in zip(param_names, param_types)
    )

    headers = []
    launches = []

    for kernel_name_, param_names_, combination in zip(
        kernel_names, all_param_names, combinations
    ):
        header = output_dir / f"{kernel_name_}.h"
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

    return _generate_launch_func(kernel_name=kernel_name, output_dir=output_dir)


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

    application_signature = inspect.signature(application)
    param_names = tuple(application_signature.parameters.keys())

    return kernel_name_, param_names, combination


def _generate_condition(combination):
    return " && ".join(f"{param} == {value}" for param, value in combination.items())


def _generate_suffix(values):
    return "_".join(f"{value}" for value in values)
