import ast
import pathlib
import re
import subprocess
import tempfile
import uuid

import ninetoothed.naming as naming
from ninetoothed.dtype import int64
from ninetoothed.generation import CACHE_DIR, CodeGenerator
from ninetoothed.tensor import Tensor


def aot(
    func, caller="cuda", kernel_name=None, output_dir=None, num_warps=4, num_stages=3
):
    output_dir = pathlib.Path(output_dir)

    output_contents = _aot(func, caller, kernel_name, num_warps, num_stages)

    for output_name, output_content in output_contents.items():
        output_path = output_dir / f"{kernel_name}{output_name[-2:]}"

        with open(output_path, "w") as f:
            f.write(output_content)


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

    param_strings = ["stream"]
    param_types = []
    constexpr_param_indices = []

    for arg in kernel_func.args.args:
        param = arg.arg

        param_strings.append(param)

        if match := Tensor.pointer_pattern().fullmatch(param):
            source_name = match.group(0).removesuffix("_pointer")
            tensor = _find_tensor_by_source_name(tensors, source_name)
            dtype = tensor.source.dtype

            param_types.append(f"*{dtype}")
        elif Tensor.size_pattern().fullmatch(param):
            param_types.append(int64)
        elif Tensor.stride_pattern().fullmatch(param):
            param_types.append(int64)
        else:
            source_name = param
            tensor = _find_tensor_by_source_name(tensors, source_name)
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

    grid_extractor = _GridExtractor()
    launch_func = grid_extractor.visit(launch_func)
    grid_extractor.visit(code_generator.raw_grid)
    grid = f"{ast.unparse(grid_extractor.grid[0])}, 1, 1"

    signature_hash, output_contents = _compile(
        source_file, kernel_name, signature, grid, num_warps, num_stages
    )

    c_source_file_name = f"{kernel_name}.{signature_hash}.c"
    c_source_file = output_contents[c_source_file_name]

    c_header_file_name = f"{kernel_name}.{signature_hash}.h"
    c_header_file = output_contents[c_header_file_name]

    pattern = rf"\({', '.join(rf'(.*) {param}' for param in param_strings)}\)"
    c_param_type_strings = re.search(pattern, c_header_file).groups()

    unparser = _Unparser(c_param_type_strings)

    launch_func_unparsed = unparser.unparse(launch_func)
    launch_func_unparsed = launch_func_unparsed.replace(
        func.__name__, f"{kernel_name}_{signature_hash}"
    )

    c_source_file = f"{c_source_file}\n{launch_func_unparsed}\n"
    c_source_file = c_source_file.replace("<stdint.h>", f'"{_HEADER_PATH}"')
    output_contents[c_source_file_name] = c_source_file

    c_header_file = f'{c_header_file}\n#ifdef __cplusplus\nextern "C" {unparser.header};\n#else\n{unparser.header};\n#endif\n'
    c_header_file = c_header_file.replace("<stdint.h>", f'"{_HEADER_PATH}"')
    output_contents[c_header_file_name] = c_header_file

    return output_contents


_HEADER_CONTENT = """#ifndef NINETOOTHED_H
#define NINETOOTHED_H

#include <stdint.h>

typedef struct {
    void *data;
    uint64_t *shape;
    int64_t *strides;
} NineToothedTensor;

typedef void *NineToothedStream;

typedef int NineToothedResult;

#endif // NINETOOTHED_H
"""

_HEADER_PATH = CACHE_DIR / "ninetoothed.h"


class _Unparser:
    def __init__(self, param_types):
        self._param_types = param_types

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
            + [
                arg
                for arg in node.args
                if not isinstance(arg, ast.Name) or not naming.is_constexpr(arg.id)
            ],
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
