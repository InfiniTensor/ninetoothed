import ast
import ctypes
import pathlib
import re
import shutil
import subprocess
import tempfile
import textwrap
import uuid

import ninetoothed.dtype
import ninetoothed.naming as naming
from ninetoothed.generation import CACHE_DIR, CodeGenerator
from ninetoothed.tensor import Tensor
from ninetoothed.utils import calculate_default_configs


def aot(
    func,
    caller="cuda",
    kernel_name=None,
    output_dir=None,
    num_warps=None,
    num_stages=None,
):
    default_num_warps, default_num_stages = calculate_default_configs()

    if num_warps is None:
        num_warps = default_num_warps

    if num_stages is None:
        num_stages = default_num_stages

    output_dir = pathlib.Path(output_dir)

    output_contents = _aot(func, caller, kernel_name, num_warps, num_stages)

    for output_name, output_content in output_contents.items():
        output_path = output_dir / output_name

        with open(output_path, "w") as f:
            f.write(output_content)

    return _generate_launch_func(kernel_name=kernel_name, output_dir=output_dir)


def _aot(func, caller, kernel_name, num_warps, num_stages):
    def _find_tensor_by_source_name(tensors, name):
        name = naming.remove_prefixes(name)

        for tensor in tensors:
            if naming.remove_prefixes(tensor.source.name) == name:
                return tensor

    _HEADER_PATH.parent.mkdir(exist_ok=True)

    if not _HEADER_PATH.exists() or _HEADER_PATH.read_text() != _HEADER_CONTENT:
        _HEADER_PATH.write_text(_HEADER_CONTENT)

    code_generator = CodeGenerator()
    source_file = code_generator(
        func,
        caller=caller,
        kernel_name=kernel_name,
        num_warps=num_warps,
        num_stages=num_stages,
        max_num_configs=None,
        prettify=False,
    )

    tensors = code_generator.tensors
    kernel_func = code_generator.kernel_func
    launch_func = code_generator.launch_func

    grid_extractor = _GridExtractor()
    launch_func = grid_extractor.visit(launch_func)
    grid_extractor.visit(code_generator.raw_grid)
    grid = f"{ast.unparse(grid_extractor.grid[0])}, 1, 1"

    contiguous_outputs = _build_stride_variant(
        source_file,
        kernel_func,
        launch_func,
        tensors,
        _find_tensor_by_source_name,
        func,
        kernel_name=kernel_name,
        variant_suffix="contiguous",
        grid=grid,
        num_warps=num_warps,
        num_stages=num_stages,
        inner_stride_constexpr=True,
    )

    generic_outputs = _build_stride_variant(
        source_file,
        kernel_func,
        launch_func,
        tensors,
        _find_tensor_by_source_name,
        func,
        kernel_name=kernel_name,
        variant_suffix="generic",
        grid=grid,
        num_warps=num_warps,
        num_stages=num_stages,
        inner_stride_constexpr=False,
    )

    output_contents = {**contiguous_outputs, **generic_outputs}

    launch_arg_names = tuple(arg.arg for arg in launch_func.args.args)
    dispatcher_source, dispatcher_header = _generate_stride_dispatcher(
        kernel_name, launch_arg_names, tensors, _find_tensor_by_source_name
    )

    output_contents[f"{kernel_name}.cpp"] = dispatcher_source
    output_contents[f"{kernel_name}.h"] = dispatcher_header

    return output_contents


def _generate_stride_dispatcher(kernel_name, launch_arg_names, tensors, find_tensor):
    param_list = ", ".join(f"NineToothedTensor {name}" for name in launch_arg_names)
    call_args = ", ".join(launch_arg_names)

    check_exprs = []

    for name in launch_arg_names:
        tensor = find_tensor(tensors, name)

        if tensor is not None and tensor.source.ndim > 0:
            check_exprs.append(f"{name}.strides[{tensor.source.ndim - 1}] == 1")

    checks = " && ".join(check_exprs) or "true"

    signature = f"NineToothedResult launch_{kernel_name}(NineToothedStream stream{', ' + param_list if param_list else ''})"

    guard = f"NINETOOTHED_{kernel_name.upper()}_H"
    header = (
        f"#ifndef {guard}\n"
        f"#define {guard}\n\n"
        f'#include "{_HEADER_PATH}"\n\n'
        f'#ifdef __cplusplus\nextern "C" {signature};\n'
        f"#else\n{signature};\n#endif\n\n"
        f"#endif\n"
    )

    source = (
        f'#include "{_HEADER_PATH}"\n'
        f'\nextern "C" NineToothedResult launch_{kernel_name}_contiguous(NineToothedStream stream{", " + param_list if param_list else ""});\n'
        f'extern "C" NineToothedResult launch_{kernel_name}_generic(NineToothedStream stream{", " + param_list if param_list else ""});\n'
        f'\nextern "C" {signature} {{\n'
        f"{_INDENTATION}bool contiguous = {checks};\n"
        f"{_INDENTATION}if (contiguous) {{\n"
        f"{_INDENTATION}{_INDENTATION}return launch_{kernel_name}_contiguous(stream{', ' + call_args if call_args else ''});\n"
        f"{_INDENTATION}}} else {{\n"
        f"{_INDENTATION}{_INDENTATION}return launch_{kernel_name}_generic(stream{', ' + call_args if call_args else ''});\n"
        f"{_INDENTATION}}}\n"
        f"}}\n"
    )

    return source, header


def _build_stride_variant(
    source_file,
    kernel_func,
    launch_func,
    tensors,
    find_tensor,
    func,
    *,
    kernel_name,
    variant_suffix,
    grid,
    num_warps,
    num_stages,
    inner_stride_constexpr,
):
    param_strings = ["stream"]
    param_types = []
    constexpr_param_indices = []
    constexpr_inner_strides = []

    for arg in kernel_func.args.args:
        param = arg.arg

        param_strings.append(param)

        if match := Tensor.pointer_pattern().fullmatch(param):
            source_name = match.group(1)
            tensor = find_tensor(tensors, source_name)
            dtype = tensor.source.dtype

            param_types.append(f"*{dtype}:16")
        elif Tensor.size_pattern().fullmatch(param):
            param_types.append(ninetoothed.dtype.int64)
        elif match := Tensor.stride_pattern().fullmatch(param):
            source_name = match.group(1)
            dim_index = int(match.group(3))
            tensor = find_tensor(tensors, source_name)

            if inner_stride_constexpr and dim_index == tensor.source.ndim - 1:
                param_types.append("1")
                constexpr_param_indices.append(len(param_types) - 1)
                constexpr_inner_strides.append((source_name, dim_index))
            else:
                param_types.append(f"{ninetoothed.dtype.int64}:16")
        else:
            source_name = param
            tensor = find_tensor(tensors, source_name)
            dtype = tensor.source.dtype

            if tensor.constexpr:
                param_types.append(f"{tensor.value}")
                constexpr_param_indices.append(len(param_types) - 1)
            else:
                param_types.append(dtype)

    signature = ", ".join(param_types)

    for index in sorted(set(constexpr_param_indices), reverse=True):
        param_strings.pop(index + 1)
        param_types.pop(index)

    signature_hash, output_contents = _compile(
        source_file, kernel_name, signature, grid, num_warps, num_stages
    )

    c_source_file_name = f"{kernel_name}.{signature_hash}.c"
    c_source_file = output_contents[c_source_file_name]

    c_header_file_name = f"{kernel_name}.{signature_hash}.h"
    c_header_file = output_contents[c_header_file_name]

    pattern = rf"\({', '.join(rf'(.*) {param}' for param in param_strings)}\)"
    c_param_type_strings = re.search(pattern, c_header_file).groups()

    kernel_name_with_hash = f"{kernel_name}_{signature_hash}"

    unparser = _Unparser(c_param_type_strings, constexpr_inner_strides)

    launch_func_unparsed = unparser.unparse(launch_func)
    launch_func_unparsed_lines = launch_func_unparsed.splitlines()
    launch_func_unparsed_lines.insert(1, f"{_INDENTATION}cuCtxGetId(NULL, &ctx_id);\n")
    launch_func_unparsed_lines.insert(1, f"{_INDENTATION}unsigned long long ctx_id;")
    launch_func_unparsed = "\n".join(launch_func_unparsed_lines)
    launch_func_unparsed = launch_func_unparsed.replace(
        func.__name__, f"kernels_{variant_suffix}[ctx_id].{kernel_name_with_hash}"
    )
    launch_func_unparsed = launch_func_unparsed.replace(
        f"launch_{kernel_name}(", f"launch_{kernel_name}_{variant_suffix}(", 1
    )

    c_source_file = c_source_file.replace("<stdint.h>", f'"{_HEADER_PATH}"')
    output_contents[c_source_file_name] = c_source_file

    output_contents.pop(c_header_file_name, None)

    kernel_start = c_source_file.find("//")
    kernel_end = len(c_source_file)
    cpp_source_file = (
        c_source_file[:kernel_start]
        + f"namespace {kernel_name_with_hash} {{\n"
        + "struct Kernel {\n"
        + textwrap.indent(c_source_file[kernel_start:kernel_end], _INDENTATION)
        + "};\n"
        + textwrap.indent(c_source_file[kernel_end:], _INDENTATION)
        + "}\n"
        + f"\nstatic ninetoothed::ThreadSafeUnorderedMap<unsigned long long, {kernel_name_with_hash}::Kernel> kernels_{variant_suffix};\n"
        + f'\nextern "C" {launch_func_unparsed}\n'
    )
    cpp_source_file_name = f"{kernel_name}.{variant_suffix}.cpp"
    output_contents[cpp_source_file_name] = cpp_source_file
    output_contents.pop(c_source_file_name)

    return output_contents


_INDENTATION = "    "

_MACRO_MAPPING = {
    True: ("NINETOOTHED_TRUE", 1),
    False: ("NINETOOTHED_FALSE", 0),
    None: ("NINETOOTHED_NONE", 0),
}

_DTYPE_MAPPING = {
    ninetoothed.dtype.int8: "NINETOOTHED_INT8",
    ninetoothed.dtype.int16: "NINETOOTHED_INT16",
    ninetoothed.dtype.int32: "NINETOOTHED_INT32",
    ninetoothed.dtype.int64: "NINETOOTHED_INT64",
    ninetoothed.dtype.uint8: "NINETOOTHED_UINT8",
    ninetoothed.dtype.uint16: "NINETOOTHED_UINT16",
    ninetoothed.dtype.uint32: "NINETOOTHED_UINT32",
    ninetoothed.dtype.uint64: "NINETOOTHED_UINT64",
    ninetoothed.dtype.float16: "NINETOOTHED_FLOAT16",
    ninetoothed.dtype.bfloat16: "NINETOOTHED_BFLOAT16",
    ninetoothed.dtype.float32: "NINETOOTHED_FLOAT32",
    ninetoothed.dtype.float64: "NINETOOTHED_FLOAT64",
}

_DTYPE_TO_INDEX = {name: i for i, name in enumerate(_DTYPE_MAPPING.keys())}

_MACRO_CONTENT = "\n\n".join(
    f"#define {identifier} {replacement}"
    for identifier, replacement in _MACRO_MAPPING.values()
)

_DATA_TYPE_BODY_CONTENT = ",\n    ".join(_DTYPE_MAPPING.values())

_TEMPLATES_DIR = pathlib.Path(__file__).parent / "templates"

_AUTO_TUNING_CACHE_CONTENT = (
    (_TEMPLATES_DIR / "auto_tuning_cache.h").read_text().strip()
)

_THREAD_SAFE_UNORDERED_MAP_CONTENT = (
    (_TEMPLATES_DIR / "thread_safe_unordered_map.h").read_text().strip()
)

_HEADER_CONTENT = f"""#ifndef NINETOOTHED_H
#define NINETOOTHED_H

#include <stdint.h>

{_MACRO_CONTENT}

enum NineToothedDataType {{
    {_DATA_TYPE_BODY_CONTENT}
}};

typedef struct {{
    void *data;
    uint64_t *shape;
    int64_t *strides;
}} NineToothedTensor;

typedef void *NineToothedStream;

typedef int NineToothedResult;

#ifdef __cplusplus
{_AUTO_TUNING_CACHE_CONTENT}

{_THREAD_SAFE_UNORDERED_MAP_CONTENT}
#endif

#endif // NINETOOTHED_H
"""

_HEADER_PATH = CACHE_DIR / "ninetoothed.h"


class _Unparser:
    def __init__(self, param_types, constexpr_inner_strides=()):
        self._param_types = param_types

        self._constexpr_inner_strides = set(constexpr_inner_strides)

    def unparse(self, node):
        method_name = "_unparse_" + node.__class__.__name__

        if hasattr(self, method_name):
            return getattr(self, method_name)(node)

        return self._generic_unparse(node)

    def _generic_unparse(self, node):
        return ast.unparse(node)

    def _unparse_Expr(self, node):
        return self.unparse(node.value)

    def _unparse_Call(self, node):
        call = ast.Call(
            func=node.func,
            args=[ast.Name(id="stream", ctx=ast.Load())]
            + [arg for arg in node.args if not self._is_excluded(arg)],
            keywords=[],
        )

        unparsed = f"return {self._generic_unparse(call)};"

        pattern = rf"\((stream), {', '.join(r'([^,]*)' for _ in range(len(self._param_types) - 1))}\)"
        args = re.search(pattern, unparsed).groups()

        for i, (arg, type) in enumerate(zip(args, self._param_types)):
            if i != 0 and "." not in arg:
                new_arg = f"*({type} *){arg}.data"
            else:
                new_arg = f"({type}){arg}"

            unparsed = unparsed.replace(arg, new_arg)

        return unparsed

    def _unparse_FunctionDef(self, node):
        params = ["NineToothedStream stream"]
        params += [f"NineToothedTensor {arg.arg}" for arg in node.args.args]
        header = f"NineToothedResult {node.name}({', '.join(params)})"

        self.header = header

        body_lines = []

        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                continue

            stmt_unparsed = self.unparse(stmt)

            if isinstance(stmt, ast.Expr):
                stmt_unparsed = stmt_unparsed.strip()

            if not stmt_unparsed.endswith(";"):
                stmt_unparsed += ";"

            body_lines.append("    " + stmt_unparsed)

        body = "\n".join(body_lines)

        return f"{header} {{\n{body}\n}}"

    def _is_excluded(self, arg):
        if isinstance(arg, ast.Name) and naming.is_constexpr(arg.id):
            return True

        if (
            isinstance(arg, ast.Subscript)
            and isinstance(arg.value, ast.Attribute)
            and arg.value.attr == "strides"
            and isinstance(arg.slice, ast.Constant)
            and isinstance(arg.value.value, ast.Name)
        ):
            return (
                arg.value.value.id,
                arg.slice.value,
            ) in self._constexpr_inner_strides

        return False


class _GridExtractor(ast.NodeTransformer):
    def visit_BinOp(self, node):
        self.generic_visit(node)

        if isinstance(node.op, ast.FloorDiv):
            node.op = ast.Div()

        return node

    def visit_Call(self, node):
        self.generic_visit(node)

        node.func = node.func.value

        return node

    def visit_Lambda(self, node):
        self.generic_visit(node)

        self.grid = node.body.elts

        return node


class _ArgumentTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("shape", ctypes.POINTER(ctypes.c_uint64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
    ]

    @staticmethod
    def from_torch_tensor(tensor):
        ndim = tensor.ndim
        shape_array_type = _SHAPE_ARRAY_TYPES_BY_NDIM[ndim]
        strides_array_type = _STRIDES_ARRAY_TYPES_BY_NDIM[ndim]

        shape = shape_array_type(*tensor.shape)
        strides = strides_array_type(*tensor.stride())

        arg_tensor = _ArgumentTensor(tensor.data_ptr(), shape, strides)
        arg_tensor._torch_tensor = tensor

        return arg_tensor

    @staticmethod
    def from_scalar(value, ctype):
        buffer = ctype(value)
        arg_tensor = _ArgumentTensor(
            ctypes.addressof(buffer), _EMPTY_SHAPE_ARRAY, _EMPTY_STRIDES_ARRAY
        )
        arg_tensor._buffer = buffer

        return arg_tensor


_MAX_NUM_DIMS = 8

_SHAPE_ARRAY_TYPES_BY_NDIM = tuple(ctypes.c_uint64 * i for i in range(_MAX_NUM_DIMS))

_STRIDES_ARRAY_TYPES_BY_NDIM = tuple(ctypes.c_int64 * i for i in range(_MAX_NUM_DIMS))

_EMPTY_SHAPE_ARRAY = _SHAPE_ARRAY_TYPES_BY_NDIM[0]()

_EMPTY_STRIDES_ARRAY = _STRIDES_ARRAY_TYPES_BY_NDIM[0]()


class _KernelLaunchError(RuntimeError):
    def __init__(self, error_code):
        self._message = f"Kernel launch failed with error code: {error_code}."

        super().__init__(self._message)


def _compile(path, name, signature, grid, num_warps, num_stages):
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = pathlib.Path(temp_dir)
        output_name = uuid.uuid4().hex
        output_path = output_dir / output_name

        command = [
            "python",
            "-m",
            "triton.tools.compile",
            str(path),
            "--kernel-name",
            str(name),
            "--signature",
            str(signature),
            "--grid",
            str(grid),
            "--num-warps",
            str(num_warps),
            "--num-stages",
            str(num_stages),
            "--out-path",
            str(output_path),
        ]

        subprocess.run(command, check=True)

        matching_files = list(output_dir.glob(f"{output_name}.*"))

        signature_hash = matching_files[0].name.split(".")[1]

        output_contents = {}

        for file in matching_files:
            with file.open() as f:
                output_contents[file.name.replace(output_name, name)] = f.read()

    return signature_hash, output_contents


def _generate_launch_func(kernel_name, output_dir):
    output_dir = pathlib.Path(output_dir)

    _compile_library(kernel_name, output_dir)

    return _load_launch_func(kernel_name, output_dir)


def _load_launch_func(kernel_name, output_dir):
    import torch

    library = _load_library(kernel_name, output_dir)
    launch_func_name = f"launch_{kernel_name}"
    launch_func = getattr(library, launch_func_name)

    dtype_to_index = _DTYPE_TO_INDEX
    from_torch_tensor = _ArgumentTensor.from_torch_tensor
    from_scalar = _ArgumentTensor.from_scalar
    c_double = ctypes.c_double
    c_void_p = ctypes.c_void_p
    current_device = torch.cuda.current_device
    get_current_raw_stream = torch._C._cuda_getCurrentRawStream
    Tensor_cls = torch.Tensor

    def _run_launch_func(*args):
        arguments = [None] * len(args)

        for i, arg in enumerate(args):
            if isinstance(arg, Tensor_cls):
                arguments[i] = from_torch_tensor(arg)
            elif type(arg) is str:
                arguments[i] = dtype_to_index[arg]
            elif type(arg) is float:
                arguments[i] = from_scalar(arg, c_double)
            else:
                arguments[i] = arg

        stream = c_void_p(get_current_raw_stream(current_device()))
        result = launch_func(stream, *arguments)

        if result != 0:
            raise _KernelLaunchError(result)

    return _run_launch_func


def _compile_library(kernel_name, output_dir):
    command = [
        "nvcc",
        "-shared",
        "-arch",
        "native",
        "--threads",
        "0",
        "-Xcompiler",
        "-fPIC",
        # TODO: Remove the following 2 lines after the return value issue is resolved.
        "-Xcompiler",
        "-Wno-return-type",
        "-lcuda",
        "-o",
        output_dir / f"{kernel_name}.so",
    ] + list(output_dir.glob(f"{kernel_name}*.cpp"))

    subprocess.run(command, check=True)


def _load_library(kernel_name, kernel_dir):
    suffix = ".so"

    original_path = kernel_dir / f"{kernel_name}{suffix}"

    with tempfile.NamedTemporaryFile(suffix=suffix) as temp_file:
        temp_path = temp_file.name

        shutil.copy(original_path, temp_path)

        library = ctypes.CDLL(temp_path)

    return library
