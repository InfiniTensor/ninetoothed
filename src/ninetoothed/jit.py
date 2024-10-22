import ast
import collections
import functools
import importlib.util
import inspect
import itertools
import math
import sys
import tempfile

import triton

from ninetoothed.language import attribute, call
from ninetoothed.symbol import Symbol
from ninetoothed.tensor import Tensor
from ninetoothed.torchifier import Torchifier


def jit(func):
    return JIT(func)()


class JIT:
    handles = collections.defaultdict(dict)

    def __init__(self, func):
        self.func = func

    def __call__(self):
        source_file = inspect.getsourcefile(self.func)
        source_line = inspect.getsourcelines(self.func)[1]

        if (
            source_file in type(self).handles
            and source_line in type(self).handles[source_file]
        ):
            return type(self).handles[source_file][source_line]

        tree = self._get_tree()

        CodeGenerator(inspect.get_annotations(self.func)).visit(tree)
        Tritonizer().visit(tree)
        ast.fix_missing_locations(tree)

        unparsed = ast.unparse(tree).replace("None:", ":").replace(":None", ":")
        dependencies = self._find_dependencies()
        source = "\n\n".join((unparsed, dependencies)).strip()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
            temp_file.write(source.encode("utf-8"))
            temp_file_name = temp_file.name

        module = type(self)._import_from_path(temp_file_name, temp_file_name)
        module_vars = vars(module)

        handle = _Handle(
            module_vars[self.func.__name__],
            module_vars[f"launch_{self.func.__name__}"],
            source,
        )

        type(self).handles[source_file][source_line] = handle

        return handle

    def _get_tree(self):
        module = ast.parse(inspect.getsource(inspect.getmodule(self.func)))

        _AliasRestorer().visit(module)
        finder = _FunctionDefFinder(self.func.__name__)
        finder.visit(module)

        return ast.Module(body=[finder.result], type_ignores=[])

    def _find_dependencies(self):
        dependencies = set()

        for obj in self.func.__globals__.values():
            if isinstance(obj, triton.runtime.JITFunction):
                dependencies.add(obj.src)

        return "\n".join(f"@triton.jit\n{dependency}" for dependency in dependencies)

    @staticmethod
    def _import_from_path(module_name, file_path):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        return module


class CodeGenerator(ast.NodeTransformer):
    def __init__(self, context):
        super().__init__()

        self._context = context

        self._args = list(self._context.values())

        self._POWER_OF_TWOS = tuple(2**n for n in range(5, 11))

        self._MIN_PRODUCT = 2**10

        self._MAX_PRODUCT = 2**20

    def visit_Module(self, node):
        self.generic_visit(node)

        node.body.append(self._launch)

        return node

    def visit_FunctionDef(self, node):
        self._func_def = node

        self.generic_visit(node)

        for arg in self._args:
            if not isinstance(arg, Tensor):
                continue

            offsets = arg.offsets()

            initializations = {
                type(self)._name_for_offsets(arg, dim): offs
                for dim, offs in enumerate(offsets)
            } | {
                type(self)._name_for_pointers(arg): arg.original.pointer_string()
                + sum(
                    type(self)._name_for_offsets(arg, dim)[
                        type(self)._generate_slices(arg, dim)
                    ]
                    * stride
                    for dim, stride in enumerate(arg.original.strides)
                )
            }

            for target, value in reversed(initializations.items()):
                node.body.insert(0, ast.Assign(targets=[target.node], value=value.node))

        return node

    def visit_arguments(self, node):
        self.generic_visit(node)

        names_of_args = [arg.names() - {"ninetoothed"} for arg in self._args]
        names = functools.reduce(lambda x, y: x | y, names_of_args)
        meta_names = {name for name in names if Symbol.is_meta(name)}
        non_meta_names = {name for name in names if name not in meta_names}

        node.args = [
            ast.arg(arg=name)
            if not Symbol.is_constexpr(name)
            else ast.arg(arg=name, annotation=attribute("constexpr").node)
            for name in non_meta_names
        ] + [
            ast.arg(arg=name, annotation=attribute("constexpr").node)
            for name in meta_names
        ]

        autotune = self._generate_autotune(non_meta_names, meta_names)
        self._func_def.decorator_list.insert(0, autotune)

        self._launch = self._generate_launch(non_meta_names, meta_names)

        return node

    def visit_Subscript(self, node):
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in self._context
            and isinstance(node.ctx, ast.Load)
        ):
            value = self._context[node.value.id]

            if isinstance(value, Tensor):
                return type(self)._generate_load(
                    value,
                    intermediate_indices=node.slice.elts
                    if isinstance(node.slice, ast.Tuple)
                    else (node.slice,),
                )

        self.generic_visit(node)

        return node

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id in self._context:
            value = self._context[node.value.id]

            if isinstance(value, Tensor):
                inner = value.dtype

                return Symbol(getattr(inner, node.attr)).node

        self.generic_visit(node)

        return node

    def visit_Name(self, node):
        self.generic_visit(node)

        if node.id in self._context and isinstance(node.ctx, ast.Load):
            return type(self)._generate_load(self._context[node.id])

        return node

    def visit_Assign(self, node):
        if len(node.targets) == 1:
            target = node.targets[0]

            if isinstance(target, ast.Name) and target.id in self._context:
                self.generic_visit(node)

                return ast.Expr(
                    type(self)._generate_store(self._context[target.id], node.value)
                )
            elif (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id in self._context
                and isinstance(target.ctx, ast.Store)
            ):
                value = self._context[target.value.id]

                if isinstance(value, Tensor):
                    self.generic_visit(node)

                    return ast.Expr(
                        type(self)._generate_store(
                            value,
                            node.value,
                            intermediate_indices=target.slice.elts
                            if isinstance(target.slice, ast.Tuple)
                            else (target.slice,),
                        )
                    )

        self.generic_visit(node)

        return node

    def _generate_autotune(self, params, meta):
        device = triton.runtime.driver.active.get_current_device()
        properties = triton.runtime.driver.active.utils.get_device_properties(device)
        max_shared_mem = properties["max_shared_mem"]

        num_warps = 8
        num_stages = max_shared_mem // 2**15

        configs = [
            ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="ninetoothed", ctx=ast.Load()),
                    attr="Config",
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Dict(
                        keys=[ast.Constant(value=param) for param in meta],
                        values=[ast.Constant(value=value) for value in values],
                    )
                ],
                keywords=[
                    ast.keyword(arg="num_warps", value=ast.Constant(value=num_warps)),
                    ast.keyword(arg="num_stages", value=ast.Constant(value=num_stages)),
                ],
            )
            for values in itertools.product(self._POWER_OF_TWOS, repeat=len(meta))
            if self._MIN_PRODUCT <= math.prod(values) <= self._MAX_PRODUCT
        ]

        return ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="ninetoothed", ctx=ast.Load()),
                attr="autotune",
                ctx=ast.Load(),
            ),
            args=[],
            keywords=[
                ast.keyword(
                    arg="configs",
                    value=ast.List(
                        elts=configs,
                        ctx=ast.Load(),
                    ),
                ),
                ast.keyword(
                    arg="key",
                    value=ast.List(
                        elts=[
                            ast.Constant(value=param)
                            for param in params
                            if not Tensor.pointer_pattern().fullmatch(param)
                        ],
                        ctx=ast.Load(),
                    ),
                ),
            ],
        )

    def _generate_launch(self, params, meta):
        constexpr_params = [param for param in params if Symbol.is_constexpr(param)]
        constexpr_params_without_prefixes = [
            Symbol.remove_prefix(param) for param in constexpr_params
        ]

        launch = ast.FunctionDef(
            name=f"launch_{self._func_def.name}",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=arg.original.name) for arg in self._args]
                + [ast.arg(arg=param) for param in constexpr_params_without_prefixes],
                kwonlyargs=[],
                defaults=[],
            ),
            body=[
                ast.Assign(
                    targets=[ast.Name(id=param, ctx=ast.Store())],
                    value=ast.Name(id=param_without_prefix, ctx=ast.Load()),
                )
                for param, param_without_prefix in zip(
                    constexpr_params, constexpr_params_without_prefixes
                )
            ]
            + [
                ast.Expr(
                    ast.Call(
                        func=ast.Subscript(
                            value=ast.Name(id=self._func_def.name, ctx=ast.Load()),
                            slice=self._generate_grid(),
                            ctx=ast.Load(),
                        ),
                        args=[ast.Name(id=param, ctx=ast.Load()) for param in params],
                        keywords=[],
                    )
                )
            ],
            decorator_list=[],
        )

        class MetaEncloser(ast.NodeTransformer):
            def __init__(self, meta):
                self._meta = meta

            def visit_Name(self, node):
                self.generic_visit(node)

                if node.id in self._meta:
                    return ast.Subscript(
                        value=ast.Name(id="meta", ctx=ast.Load()),
                        slice=ast.Constant(value=node.id),
                        ctx=ast.Load(),
                    )

                return node

        MetaEncloser(meta).visit(launch)

        Torchifier().visit(launch)

        return launch

    def _generate_grid(self):
        num_elements = functools.reduce(lambda x, y: x * y, self._args[0].shape)

        return ast.parse(f"lambda meta: ({num_elements},)", mode="eval").body

    @staticmethod
    def _generate_load(tensor, intermediate_indices=()):
        pointers, mask = CodeGenerator._generate_pointers_and_mask(
            tensor, intermediate_indices
        )
        other = CodeGenerator._generate_other(tensor)

        return call("load", pointers, mask=mask, other=other).node

    @staticmethod
    def _generate_store(tensor, value, intermediate_indices=()):
        pointers, mask = CodeGenerator._generate_pointers_and_mask(
            tensor, intermediate_indices
        )

        return call("store", pointers, value, mask=mask).node

    @staticmethod
    def _generate_pointers_and_mask(tensor, intermediate_indices):
        intermediate_offsets = CodeGenerator._generate_intermediate_offsets(
            tensor, intermediate_indices
        )
        offsets = [
            CodeGenerator._name_for_offsets(tensor, dim) + intermediate_offsets[dim]
            for dim in range(tensor.original.ndim)
        ]
        pointers = CodeGenerator._name_for_pointers(tensor) + sum(
            map(lambda x, y: x * y, intermediate_offsets, tensor.original.strides)
        )
        mask = functools.reduce(
            lambda x, y: x & y,
            (
                offs[CodeGenerator._generate_slices(tensor, dim)] < size
                for dim, (offs, size) in enumerate(zip(offsets, tensor.original.shape))
            ),
        )

        return pointers, mask

    @staticmethod
    def _generate_other(tensor):
        other = tensor.original.other

        if isinstance(other, float) and not math.isfinite(other):
            return f"float('{other}')"

        return other

    @staticmethod
    def _generate_slices(tensor, dim):
        return tuple(slice(None) if i == dim else None for i in range(tensor.ndim))

    @staticmethod
    def _generate_intermediate_offsets(tensor, intermediate_indices):
        return tuple(
            offs
            for offs in tensor.offsets(
                [0 for _ in range(tensor.ndim)]
                + list(intermediate_indices)
                + [0 for _ in range(tensor.inmost().ndim)]
            )
        )

    @staticmethod
    def _name_for_pointers(tensor):
        return Symbol(f"{tensor.original.name}_pointers")

    @staticmethod
    def _name_for_offsets(tensor, dim):
        return Symbol(f"{tensor.original.name}_offsets_{dim}")


class Tritonizer(ast.NodeTransformer):
    def visit_Module(self, node):
        self.generic_visit(node)

        node.body.insert(0, ast.Import(names=[ast.alias(name="triton.language")]))
        node.body.insert(0, ast.Import(names=[ast.alias(name="triton")]))

        return node

    def visit_Name(self, node):
        self.generic_visit(node)

        if node.id == "ninetoothed" or "ninetoothed." in node.id:
            node.id = node.id.replace("ninetoothed", "triton")

        return node

    def visit_Call(self, node):
        self.generic_visit(node)

        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "triton"
            and node.func.attr == "jit"
        ):
            return ast.Attribute(
                value=ast.Name(id="triton", ctx=ast.Load()), attr="jit", ctx=ast.Load()
            )

        return node


class _Handle:
    def __init__(self, kernel, launch, source):
        self._kernel = kernel
        self._launch = launch
        self._source = source

    def __call__(self, *args, **kwargs):
        return self._launch(*args, **kwargs)


class _AliasRestorer(ast.NodeTransformer):
    def __init__(self):
        super().__init__()

        self._aliases = {}
        self._redefined = set()

    def visit_Import(self, node):
        for alias in node.names:
            if alias.asname:
                self._aliases[alias.asname] = alias.name

        return node

    def visit_ImportFrom(self, node):
        for alias in node.names:
            full_name = f"{node.module}.{alias.name}"
            if alias.asname:
                self._aliases[alias.asname] = full_name

        return node

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._redefined.add(target.id)

        return self.generic_visit(node)

    def visit_FunctionDef(self, node):
        original_redefined = self._redefined.copy()

        self.generic_visit(node)

        self._redefined = original_redefined

        return node

    def visit_Name(self, node):
        if node.id in self._redefined:
            return node

        if node.id in self._aliases:
            return ast.Name(id=self._aliases[node.id], ctx=node.ctx)

        return node


class _FunctionDefFinder(ast.NodeVisitor):
    def __init__(self, name):
        self._name = name

        self.result = None

    def visit_FunctionDef(self, node):
        if node.name == self._name:
            self.result = node

        self.generic_visit(node)
