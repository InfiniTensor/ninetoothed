import ast
import collections
import copy
import functools
import importlib.util
import inspect
import itertools
import math
import subprocess
import sys
import tempfile

import triton

import ninetoothed.naming as naming
from ninetoothed.language import attribute, call
from ninetoothed.symbol import Symbol
from ninetoothed.tensor import Tensor
from ninetoothed.torchifier import Torchifier


def make(arrangement, application, tensors):
    """Integrate the arrangement and the application of the tensors.

    :param arrangement: The arrangement of the tensors.
    :param application: The application of the tensors.
    :param tensors: The tensors.
    :return: A handle to the compute kernel.
    """
    params = inspect.signature(application).parameters
    types = arrangement(*tensors)
    annotations = {param: type for param, type in zip(params, types)}
    application.__annotations__ = annotations

    return jit(application)


def jit(func=None, *, _prettify=False):
    """A decorator for generating compute kernels.

    :param func: The function to be compiled.
    :param _prettify: Whether to prettify the generated code.
    :return: A handle to the compute kernel.

    .. note::

        The ``_prettify`` parameter is experimental, which might break
        the generated code.
    """

    def wrapper(func):
        return JIT(func, _prettify=_prettify)()

    if func is None:
        return wrapper

    return wrapper(func)


class JIT:
    def __init__(self, func, _prettify=False):
        self.func = func

        self._prettify = _prettify

    def __call__(self):
        tree = self._get_tree()

        CodeGenerator(inspect.get_annotations(self.func)).visit(tree)
        Tritonizer().visit(tree)
        _BinOpSimplifier().visit(tree)
        ast.fix_missing_locations(tree)

        if self._prettify:
            name_collector = _SimplifiedNameCollector()
            name_collector.visit(tree)

        unparsed = ast.unparse(tree).replace("None:", ":").replace(":None", ":")
        dependencies = self._find_dependencies()
        source = "\n\n".join((unparsed, dependencies)).strip()

        if self._prettify:
            for original, simplified in name_collector.simplified_names.items():
                if simplified not in name_collector.simplified_names:
                    source = source.replace(original, simplified)

            source = subprocess.check_output(
                ["ruff", "format", "-"], input=source, encoding="utf-8"
            )

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

        return handle

    def _get_tree(self):
        module = ast.parse(inspect.getsource(inspect.getmodule(self.func)))

        _AliasRestorer().visit(module)
        collector = _ImportCollector()
        collector.visit(module)
        finder = _FunctionDefFinder(self.func.__name__)
        finder.visit(module)

        return ast.Module(body=collector.imports + [finder.result], type_ignores=[])

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

        self._invariants = {}

        self.generic_visit(node)

        for target, value in reversed(self._invariants.items()):
            node.body.insert(0, ast.Assign(targets=[target.node], value=value.node))

        return node

    def visit_arguments(self, node):
        self.generic_visit(node)

        names_of_args = [arg.names() - {"ninetoothed"} for arg in self._args]
        names = functools.reduce(lambda x, y: x | y, names_of_args)
        meta_names = {name for name in names if naming.is_meta(name)}
        non_meta_names = {name for name in names if name not in meta_names}
        non_meta_names |= {
            naming.make_next_power_of_2(name)
            for name in non_meta_names
            if naming.is_constexpr(name)
        }

        node.args = [
            ast.arg(arg=name)
            if not naming.is_constexpr(name)
            else ast.arg(arg=name, annotation=attribute("constexpr").node)
            for name in non_meta_names
        ] + [
            ast.arg(arg=name, annotation=attribute("constexpr").node)
            for name in meta_names
        ]

        autotune = self._generate_autotune(non_meta_names, meta_names)
        self._func_def.decorator_list = [autotune, Symbol("triton.jit").node]

        self._launch = self._generate_launch(non_meta_names, meta_names)

        return node

    def visit_Subscript(self, node):
        if self._in_context(node.value) and isinstance(node.ctx, ast.Load):
            value = self._context[node.value.id]

            if isinstance(value, Tensor):
                return self._generate_load(
                    value,
                    indices=node.slice.elts
                    if isinstance(node.slice, ast.Tuple)
                    else (node.slice,),
                )

        self.generic_visit(node)

        return node

    def visit_Attribute(self, node):
        if self._in_context(node.value):
            value = self._context[node.value.id]

            if isinstance(value, Tensor):
                inner = value.dtype

                return Symbol(getattr(inner, node.attr)).node

        self.generic_visit(node)

        return node

    def visit_Name(self, node):
        self.generic_visit(node)

        if self._in_context(node) and isinstance(node.ctx, ast.Load):
            return self._generate_load(self._context[node.id])

        return node

    def visit_Assign(self, node):
        if len(node.targets) == 1:
            target = node.targets[0]

            if self._in_context(target):
                self.generic_visit(node)

                return ast.Expr(
                    self._generate_store(self._context[target.id], node.value)
                )
            elif (
                isinstance(target, ast.Subscript)
                and self._in_context(target.value)
                and isinstance(target.ctx, ast.Store)
            ):
                value = self._context[target.value.id]

                if isinstance(value, Tensor):
                    self.generic_visit(node)

                    return ast.Expr(
                        self._generate_store(
                            value,
                            node.value,
                            indices=target.slice.elts
                            if isinstance(target.slice, ast.Tuple)
                            else (target.slice,),
                        )
                    )

        self.generic_visit(node)

        return node

    _NAME_FOR_PID = Symbol("ninetoothed_pid")

    def _in_context(self, node):
        return isinstance(node, ast.Name) and node.id in self._context

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
        non_next_power_of_2_constexpr_params = [
            param
            for param in params
            if naming.is_constexpr(param) and not naming.is_next_power_of_2(param)
        ]
        non_next_power_of_2_constexpr_params_without_prefixes = [
            naming.remove_prefixes(param)
            for param in non_next_power_of_2_constexpr_params
        ]
        next_power_of_2_params = [
            param for param in params if naming.is_next_power_of_2(param)
        ]
        next_power_of_2_params_without_prefixes = [
            naming.remove_prefixes(param) for param in next_power_of_2_params
        ]

        launch = ast.FunctionDef(
            name=f"launch_{self._func_def.name}",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=arg.source.name) for arg in self._args]
                + [
                    ast.arg(arg=param)
                    for param in non_next_power_of_2_constexpr_params_without_prefixes
                    if not Tensor.size_pattern().fullmatch(param)
                ],
                kwonlyargs=[],
                defaults=[],
            ),
            body=[
                ast.Assign(
                    targets=[ast.Name(id=param, ctx=ast.Store())],
                    value=ast.Name(id=param_without_prefixes, ctx=ast.Load()),
                )
                for param, param_without_prefixes in zip(
                    non_next_power_of_2_constexpr_params,
                    non_next_power_of_2_constexpr_params_without_prefixes,
                )
            ]
            + [
                ast.Assign(
                    targets=[ast.Name(id=param, ctx=ast.Store())],
                    value=Symbol(
                        f"triton.next_power_of_2({param_without_prefixes})"
                    ).node,
                )
                for param, param_without_prefixes in zip(
                    next_power_of_2_params,
                    next_power_of_2_params_without_prefixes,
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

    def _generate_load(self, tensor, indices=()):
        if tensor.ndim == 0:
            return Symbol(tensor.source.name).node

        pointers, mask = self._generate_pointers_and_mask(tensor, indices)
        other = type(self)._generate_other(tensor)

        return call("load", pointers, mask=mask, other=other).node

    def _generate_store(self, tensor, value, indices=()):
        pointers, mask = self._generate_pointers_and_mask(tensor, indices)

        return call("store", pointers, value, mask=mask).node

    def _generate_pointers_and_mask(self, tensor, indices):
        invariant_target_dims = type(self)._find_invariant_target_dims(tensor)

        indices = self._complete_indices(tensor, indices)
        offsets = type(self)._generate_offsets(tensor, indices)

        for source_dim in range(tensor.source.ndim):
            for target_dim in range(tensor.target.ndim):
                if target_dim not in invariant_target_dims:
                    continue

                name = type(self)._name_for_offsets(tensor, source_dim, target_dim)
                self._invariants[name] = offsets[source_dim][target_dim]
                offsets[source_dim][target_dim] = name

        name_for_pointers = type(self)._name_for_pointers(tensor)
        self._invariants[name_for_pointers] = Symbol(tensor.source.pointer_string())

        for source_dim in range(tensor.source.ndim):
            for target_dim in range(tensor.target.ndim):
                if target_dim not in invariant_target_dims:
                    continue

                self._invariants[name_for_pointers] += (
                    offsets[source_dim][target_dim][
                        type(self)._generate_slices(tensor, target_dim)
                    ]
                    * tensor.source.strides[source_dim]
                )

        pointers = name_for_pointers + sum(
            offsets[source_dim][target_dim][
                type(self)._generate_slices(tensor, target_dim)
            ]
            * tensor.source.strides[source_dim]
            for source_dim in range(tensor.source.ndim)
            for target_dim in range(tensor.target.ndim)
            if target_dim not in invariant_target_dims
            and offsets[source_dim][target_dim] != 0
        )
        mask = functools.reduce(
            lambda x, y: x & y,
            (
                offsets[source_dim][target_dim][
                    type(self)._generate_slices(tensor, target_dim)
                ]
                < tensor.source.shape[source_dim]
                for source_dim in range(tensor.source.ndim)
                for target_dim in range(tensor.target.ndim)
                if offsets[source_dim][target_dim] != 0
            ),
        )

        return pointers, mask

    def _complete_indices(self, tensor, indices):
        indices = list(self._generate_pid_indices(tensor) + tuple(indices))

        for size in tensor.innermost().shape:
            if Symbol.is_name(size):
                name = size.node.id
                if not naming.is_meta(name):
                    size = naming.make_next_power_of_2(name)

            indices.append(call("arange", 0, size))

        return tuple(indices)

    def _generate_pid_indices(self, tensor):
        self._invariants[type(self)._NAME_FOR_PID] = call("program_id", 0)

        indices = list(
            type(self)._unravel_index(type(self)._NAME_FOR_PID, tensor.shape)
        )

        for dim, index in enumerate(indices):
            name = type(self)._name_for_index(tensor, dim)
            self._invariants[name] = index
            indices[dim] = name

        return tuple(indices)

    @staticmethod
    def _generate_other(tensor):
        other = tensor.source.other

        if isinstance(other, float) and not math.isfinite(other):
            return f"float('{other}')"

        return other

    @staticmethod
    def _generate_slices(tensor, dim):
        return tuple(
            slice(None) if target_dim == dim else None
            for target_dim in tensor.innermost().target_dims
        )

    @staticmethod
    def _generate_offsets(tensor, indices):
        offsets = collections.defaultdict(
            lambda: collections.defaultdict(lambda: Symbol(0))
        )

        curr = tensor
        start = 0

        while isinstance(curr, type(tensor)):
            stop = start + curr.ndim
            curr_indices = indices[start:stop]

            for index, stride, source_dim, target_dim in zip(
                curr_indices, curr.strides, curr.source_dims, curr.target_dims
            ):
                offsets[source_dim][target_dim] += index * stride

            start = stop
            curr = curr.dtype

        for source_dim in tuple(offsets):
            for target_dim in tuple(offsets[source_dim]):
                if not isinstance(source_dim, tuple):
                    continue

                unraveled = CodeGenerator._unravel_index(
                    offsets[source_dim][target_dim],
                    tuple(tensor.source.shape[dim] for dim in source_dim),
                )

                for offs, dim in zip(unraveled, source_dim):
                    offsets[dim][target_dim] = offs

        for source_dim in range(tensor.source.ndim):
            for target_dim in range(tensor.target.ndim):
                offsets[source_dim][target_dim] = copy.deepcopy(
                    offsets[source_dim][target_dim]
                )
                offsets[source_dim][target_dim].find_and_replace(
                    Symbol(tensor.source.strides[source_dim]), Symbol(1)
                )

        return offsets

    @staticmethod
    def _find_invariant_target_dims(tensor):
        invariant_target_dims = set()

        curr = tensor.dtype

        while isinstance(curr.dtype, Tensor):
            for target_dim in range(curr.target.ndim):
                if target_dim not in curr.target_dims:
                    invariant_target_dims.add(target_dim)

            curr = curr.dtype

        return invariant_target_dims

    @staticmethod
    def _name_for_pointers(tensor):
        return Symbol(f"{tensor.source.name}_pointers")

    @staticmethod
    def _name_for_offsets(tensor, source_dim, target_dim):
        return Symbol(f"{tensor.source.name}_offsets_{source_dim}_{target_dim}")

    @staticmethod
    def _name_for_index(tensor, dim):
        return Symbol(f"{tensor.source.name}_index_{dim}")

    @staticmethod
    def _unravel_index(index, shape):
        indices = []

        for stride in Tensor(shape=shape).strides:
            indices.append(index // stride)
            index %= stride

        return tuple(indices)


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


class _BinOpSimplifier(ast.NodeTransformer):
    def visit_BinOp(self, node):
        self.generic_visit(node)

        if isinstance(node.op, ast.Mult):
            left = Symbol(node.left)
            right = Symbol(node.right)

            if left == 0 or right == 0:
                return Symbol(0).node

            if left == 1:
                return node.right

            if right == 1:
                return node.left

        return node


class _SimplifiedNameCollector(ast.NodeVisitor):
    def __init__(self):
        self.simplified_names = {}

    def visit_Name(self, node):
        self.generic_visit(node)

        self.simplified_names[node.id] = naming.remove_prefixes(node.id)


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


class _ImportCollector(ast.NodeVisitor):
    def __init__(self):
        super().__init__()

        self.imports = []

    def visit_Import(self, node):
        self.imports.append(node)

        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        self.imports.append(node)

        self.generic_visit(node)


class _FunctionDefFinder(ast.NodeVisitor):
    def __init__(self, name):
        self._name = name

        self.result = None

    def visit_FunctionDef(self, node):
        if node.name == self._name:
            self.result = node

        self.generic_visit(node)
