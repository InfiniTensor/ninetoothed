import ast
import collections
import copy
import functools
import hashlib
import inspect
import itertools
import math
import pathlib
import subprocess
import textwrap
import astpretty

import sympy
import triton

import ninetoothed.naming as naming
from ninetoothed.generation.node_transformer_utils import *
from ninetoothed.generation.tritonizer import Tritonizer
from ninetoothed.language import attribute, call
from ninetoothed.symbol import Symbol
from ninetoothed.tensor import Tensor

CACHE_DIR = pathlib.Path.home() / ".ninetoothed"
CACHE_DIR.mkdir(exist_ok=True)


class CodeGenerator(ast.NodeTransformer):
    def __init__(self):
        super().__init__()

        device = triton.runtime.driver.active.get_current_device()
        properties = triton.runtime.driver.active.utils.get_device_properties(device)

        self._min_num_elements = 1

        if "max_num_regs" in properties:
            max_innermost_size = 4 * properties["max_num_regs"]
        elif "max_nram_size" in properties:
            max_innermost_size = properties["max_nram_size"]
        else:
            max_innermost_size = 2**18

        self._max_num_elements = max_innermost_size // 8

    def __call__(
        self,
        func,
        caller,
        kernel_name,
        num_warps,
        num_stages,
        max_num_configs,
        prettify,
    ):
        def _get_tree(func):
            func_def = ast.parse(textwrap.dedent(inspect.getsource(func)))

            inliner = _Inliner(func.__globals__)
            inliner.visit(func_def)

            if inliner.libdevice_used:
                libdevice_alias = ast.alias(
                    name="libdevice", asname=inliner.LIBDEVICE_ALIAS
                )
                libdevice_import = ast.ImportFrom(
                    module="triton.language.extra",
                    names=[libdevice_alias],
                    level=0,
                )

                func_def.body.insert(0, libdevice_import)

            return func_def

        def _find_dependencies(func):
            dependencies = set()

            for obj in func.__globals__.values():
                if isinstance(obj, triton.runtime.JITFunction):
                    dependencies.add(obj.src)

            return "\n".join(
                f"@triton.jit\n{dependency}" for dependency in dependencies
            )

        self.launch_func_name = f"launch_{kernel_name}"

        self._caller = caller

        self._num_warps = num_warps

        self._num_stages = num_stages

        self._max_num_configs = max_num_configs

        self._context = inspect.get_annotations(func)

        self._args = list(self._context.values())

        tree = _get_tree(func)
        self.visit(tree)
        Tritonizer().visit(tree)
        _BinOpSimplifier().visit(tree)
        ast.fix_missing_locations(tree)

        if prettify:
            name_collector = _SimplifiedNameCollector()
            name_collector.visit(tree)

        unparsed = ast.unparse(tree).replace("None:", ":").replace(":None", ":")
        dependencies = _find_dependencies(func)
        source = "\n\n".join((unparsed, dependencies)).strip()
        source = source.replace(func.__name__, kernel_name)
        source += "\n"

        if prettify:
            for original, simplified in name_collector.simplified_names.items():
                if simplified not in name_collector.simplified_names:
                    source = source.replace(original, simplified)

            source = subprocess.check_output(
                ["ruff", "format", "-"], input=source, encoding="utf-8"
            )

        cache_file = cache_source(source)

        self.tensors = self._args
        self.kernel_func = self._func_def
        self.launch_func = self._launch

        return str(cache_file)

    def visit_Module(self, node):
        self.generic_visit(node)

        if self._autotune is not None:
            func_with_auto_tuning = f"{Symbol(self._autotune)}({self._func_def.name})"

            node.body.append(
                ast.parse(
                    f"{self._func_name_with_auto_tuning} = {func_with_auto_tuning}"
                )
            )

        node.body.append(self._launch)

        return node

    def visit_FunctionDef(self, node):
        self._func_def = node

        self._func_name_with_auto_tuning = f"{self._func_def.name}_with_auto_tuning"

        self._invariants = {}

        self.generic_visit(node)

        for target, value in reversed(self._invariants.items()):
            node.body.insert(0, ast.Assign(targets=[target.node], value=value.node))

        symbols = {
            name.node.id: name
            for arg in self._args
            for name in arg.names()
            if name != "ninetoothed"
        }
        names = symbols.keys()
        meta_names = {name for name in names if naming.is_meta(name)}
        non_meta_names = {name for name in names if name not in meta_names}
        non_meta_names |= {
            name.node.id
            for name in Symbol(node).names()
            if naming.is_next_power_of_2(name.node.id)
        }

        self._symbols = symbols

        non_meta_names = sorted(non_meta_names)
        meta_names = sorted(meta_names)

        node.args.args = [
            ast.arg(arg=name)
            if not naming.is_constexpr(name) or Tensor.size_pattern().fullmatch(name)
            else ast.arg(arg=name, annotation=attribute("constexpr").node)
            for name in non_meta_names
        ] + [
            ast.arg(arg=name, annotation=attribute("constexpr").node)
            for name in meta_names
        ]

        self._autotune = self._generate_autotune(non_meta_names, meta_names)

        if self._autotune is not None:
            self._func_name = self._func_name_with_auto_tuning
        else:
            self._func_name = self._func_def.name

        self._func_def.decorator_list = [Symbol("triton.jit").node]

        self._launch = self._generate_launch(non_meta_names, meta_names)

        return node

    def visit_Call(self, node):
        def _data_ptr(tensor):
            assert tensor is tensor.source, "Expected a source tensor."

            return Symbol(tensor.source.pointer_string()).node

        def _offsets(tensor, dim=None):
            if dim is None:
                return tensor._last_generated_overall_offsets.node

            offsets = tensor._last_generated_offsets

            if dim < 0:
                dim += tensor.source.ndim

            return _TupleSliceRemover().visit(offsets[dim].node)

        def _stride(tensor, dim):
            assert tensor is tensor.source, "Expected a source tensor."

            return Symbol(tensor.source.stride_string(dim)).node

        func = node.func
        args = node.args

        if isinstance(func, ast.Attribute):
            if func.attr in ("data_ptr", "offsets", "stride"):
                value = func.value

                if self._in_context(value):
                    tensor = self._context[value.id]
                elif isinstance(value, ast.Subscript) and self._in_context(value.value):
                    tensor = self._context[value.value.id]
                else:
                    tensor = self.visit(value)

                self.visit(value)

            if func.attr == "data_ptr":
                return _data_ptr(tensor)

            if func.attr == "offsets":
                # TODO: Add error handling.
                return _offsets(tensor, ast.literal_eval(args[0]) if args else None)

            if func.attr == "stride":
                # TODO: Add error handling.
                return _stride(tensor, ast.literal_eval(args[0]))

        self.generic_visit(node)

        return node

    def visit_Subscript(self, node):
        def _generate_load():
            return self._generate_load(
                tensor,
                indices=node.slice.elts
                if isinstance(node.slice, ast.Tuple)
                else (node.slice,),
            )

        if not hasattr(node, "ctx") or not isinstance(node.ctx, ast.Load):
            self.generic_visit(node)

            return node

        if self._in_context(node.value) and isinstance(
            tensor := self._context[node.value.id], Tensor
        ):
            return _generate_load()

        self.generic_visit(node)

        if isinstance(tensor := node.value, Tensor):
            assert tensor is tensor.source, "Expected a source tensor."

            return _generate_load()

        return node

    def visit_Attribute(self, node):
        value = node.value

        if isinstance(value, ast.Attribute):
            value = self.visit_Attribute(value)

        if self._in_context(value):
            value = self._context[value.id].dtype

        if isinstance(value, Tensor):
            attr = getattr(value, node.attr)

            if isinstance(attr, Tensor):
                return attr

            if node.attr == "dtype":
                return Symbol(f"{value.source.pointer_string()}.type.element_ty").node

            return Symbol(attr).node

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

            def _generate_store():
                return self._generate_store(
                    tensor,
                    self.visit(node.value),
                    indices=target.slice.elts
                    if isinstance(target.slice, ast.Tuple)
                    else (target.slice,),
                )

            if isinstance(target, ast.Subscript) and isinstance(target.ctx, ast.Store):
                if self._in_context(target.value) and isinstance(
                    tensor := self._context[target.value.id], Tensor
                ):
                    return ast.Expr(_generate_store())

                self.generic_visit(node)

                if isinstance(tensor := target.value, Tensor):
                    assert tensor is tensor.source, "Expected a source tensor."

                    return ast.Expr(_generate_store())

                return node

        self.generic_visit(node)

        return node

    _NAME_FOR_PID = Symbol("ninetoothed_pid")

    def _in_context(self, node):
        return isinstance(node, ast.Name) and node.id in self._context

    def _generate_autotune(self, params, meta):
        inequalities = True

        for arg in self._args:
            if arg.ndim == 0:
                continue

            num_elements = sympy.simplify(str(math.prod(arg.innermost().shape)))

            inequalities &= num_elements <= self._max_num_elements
            inequalities &= num_elements >= self._min_num_elements

        values_of_meta_params = []

        for param in meta:
            symbol = self._symbols[param]

            values = range(symbol.lower_bound, symbol.upper_bound + 1)

            if symbol.power_of_two:
                values = tuple(value for value in values if value & (value - 1) == 0)
            else:
                values = tuple(values)

            values_of_meta_params.append(values)

        max_values_of_non_meta_params = {}

        for free_symbol in inequalities.free_symbols:
            symbol_str = str(free_symbol)

            if symbol_str in meta:
                continue

            symbol = self._symbols[symbol_str]

            max_values_of_non_meta_params[symbol_str] = symbol.upper_bound

        block_size_configs = []

        for values in itertools.product(*values_of_meta_params):
            config = {param: value for param, value in zip(meta, values)}

            if sympy.logic.simplify_logic(
                inequalities.subs(config | max_values_of_non_meta_params)
            ):
                block_size_configs.append(config)

        if not block_size_configs:
            if meta:
                raise ValueError(
                    "Failed to generate auto-tuning. Please check the upper and lower bounds of the symbols."
                )
            else:
                block_size_configs.append({})

        if isinstance(self._num_warps, collections.abc.Iterable):
            num_warps_configs = self._num_warps
        else:
            num_warps_configs = (self._num_warps,)

        if isinstance(self._num_stages, collections.abc.Iterable):
            num_stages_configs = self._num_stages
        else:
            num_stages_configs = (self._num_stages,)

        compiler_configs = tuple(
            {"num_warps": num_warps, "num_stages": num_stages}
            for num_warps, num_stages in itertools.product(
                num_warps_configs, num_stages_configs
            )
        )

        configs = [
            ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id="ninetoothed", ctx=ast.Load()),
                    attr="Config",
                    ctx=ast.Load(),
                ),
                args=[
                    ast.Dict(
                        keys=[
                            ast.Constant(value=param)
                            for param in block_size_config.keys()
                        ],
                        values=[
                            ast.Constant(value=value)
                            for value in block_size_config.values()
                        ],
                    )
                ],
                keywords=[
                    ast.keyword(
                        arg="num_warps",
                        value=ast.Constant(value=compiler_config["num_warps"]),
                    ),
                    ast.keyword(
                        arg="num_stages",
                        value=ast.Constant(value=compiler_config["num_stages"]),
                    ),
                ],
            )
            for block_size_config, compiler_config in itertools.product(
                block_size_configs, compiler_configs
            )
        ]

        if self._max_num_configs is not None and len(configs) > self._max_num_configs:
            configs = [
                configs[i * len(configs) // self._max_num_configs]
                for i in range(self._max_num_configs)
            ]

        if len(configs) <= 1:
            return None

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

        arg_names = [naming.remove_prefixes(arg.source.name) for arg in self._args]

        arg_names += [
            param
            for param in non_next_power_of_2_constexpr_params_without_prefixes
            if not Tensor.size_pattern().fullmatch(param)
            and not Tensor.seq_len_pattern().fullmatch(param)
            and param not in arg_names
        ]

        launch = ast.FunctionDef(
            name=self.launch_func_name,
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=name) for name in arg_names],
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
                            value=ast.Name(id=self._func_name, ctx=ast.Load()),
                            slice=self._generate_grid(),
                            ctx=ast.Load(),
                        ),
                        args=[ast.Name(id=param, ctx=ast.Load()) for param in params],
                        keywords=[
                            ast.keyword(
                                arg="num_warps",
                                value=ast.Constant(value=self._num_warps),
                            ),
                            ast.keyword(
                                arg="num_stages",
                                value=ast.Constant(value=self._num_stages),
                            ),
                        ]
                        if self._autotune is None
                        else [],
                    )
                )
            ],
            decorator_list=[],
        )

        MetaEncloser(meta).visit(launch)

        if self._caller == "torch":
            Torchifier().visit(launch)
        elif self._caller == "cuda":
            Cudaifier().visit(launch)
        else:
            raise ValueError(f"Unsupported caller: `{self._caller}`.")

        return launch

    def _generate_grid(self):
        num_elements = functools.reduce(lambda x, y: x * y, self._args[0].shape)

        grid = ast.parse(f"lambda meta: ({num_elements},)", mode="eval").body

        self.raw_grid = copy.deepcopy(grid)

        return grid

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
        if tensor is not tensor.source:
            indices = self._complete_indices(tensor, indices)

        indices = tuple(Symbol(index) for index in indices)

        name_for_pointers = type(self)._name_for_pointers(tensor)
        self._invariants[name_for_pointers] = Symbol(tensor.source.pointer_string())

        overall_offsets, mask = type(self)._generate_overall_offsets_and_mask(
            tensor, indices
        )

        pointers = name_for_pointers + overall_offsets

        return pointers, mask

    def _complete_indices(self, tensor, indices):
        return (
            tuple(self._generate_pid_indices(tensor))
            + tuple(indices)
            + tuple(type(self)._generate_innermost_indices(tensor))
        )

    def _generate_pid_indices(self, tensor):
        self._invariants[type(self)._NAME_FOR_PID] = call("program_id", 0)

        indices = list(Tensor._unravel_index(type(self)._NAME_FOR_PID, tensor.shape))

        for dim, index in enumerate(indices):
            name = type(self)._name_for_index(tensor, dim)
            self._invariants[name] = index
            indices[dim] = name

        if tensor.source.jagged_dim is not None:
            seq_len_name = Symbol(tensor.source.seq_len_string())
            max_seq_len_name = Symbol(tensor.source.max_seq_len_string())

            for size in tensor.shape:
                size.find_and_replace(seq_len_name, max_seq_len_name)

            offsets_name = Symbol(tensor.source.offsets_string())
            batch_dim_index_name = type(self)._name_for_index(tensor, 0)
            seq_start_name = type(self)._name_for_seq_start(tensor)
            seq_end_name = type(self)._name_for_seq_end(tensor)

            self._invariants[seq_start_name] = call(
                "load", offsets_name + batch_dim_index_name
            )
            self._invariants[seq_end_name] = call(
                "load", offsets_name + batch_dim_index_name + 1
            )
            self._invariants[seq_len_name] = seq_end_name - seq_start_name

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
    def _generate_overall_offsets_and_mask(tensor, indices):
        indices = list(indices)

        offsets, mask = CodeGenerator._generate_offsets_and_mask(tensor, indices)

        tensor._last_generated_offsets = offsets

        overall_offsets = sum(
            offsets[source_dim] * Symbol(tensor.source.stride_string(source_dim))
            for source_dim in range(tensor.source.ndim)
        )

        if tensor.source.jagged_dim is not None:
            overall_offsets += CodeGenerator._name_for_seq_start(tensor) * Symbol(
                tensor.source.stride_string(tensor.source.jagged_dim)
            )

        tensor._last_generated_overall_offsets = overall_offsets

        return overall_offsets, mask

    @staticmethod
    def _generate_offsets_and_mask(tensor, indices):
        offsets = [Symbol(0) for _ in range(tensor.source.ndim)]

        tensor.source._mask = Symbol(True)

        curr = tensor
        start = 0

        while isinstance(curr, type(tensor)):
            stop = start + curr.ndim
            curr_indices = indices[start:stop]

            curr._inputs = [curr_indices]

            start = stop
            curr = curr.dtype

        for level in reversed(tensor._levels):
            for tensor_ in level:
                tensor_.offsets()

        for dim, offset in enumerate(tensor.source._outputs[0]):
            offsets[dim] += offset

        curr = tensor

        while isinstance(curr, type(tensor)):
            curr._inputs.clear()

            curr = curr.dtype

        return offsets, tensor.source._mask

    @staticmethod
    def _generate_innermost_indices(tensor, use_power_of_2_sizes=True):
        indices = []

        for size, target_dim in zip(
            tensor.innermost().shape, tensor.innermost().target_dims
        ):
            if use_power_of_2_sizes:
                size = _NextPowerOfTwoMaker().visit(Symbol(copy.deepcopy(size)).node)

            indices.append(
                call("arange", 0, size)[
                    CodeGenerator._generate_slices(tensor, target_dim)
                ]
            )

        return tuple(indices)

    @staticmethod
    def _name_for_pointers(tensor):
        return Symbol(f"{tensor.source.name}_pointers")

    @staticmethod
    def _name_for_offsets(tensor, source_dim, target_dim):
        return Symbol(f"{tensor.source.name}_offsets_{source_dim}_{target_dim}")

    @staticmethod
    def _name_for_seq_start(tensor):
        return Symbol(f"{tensor.source.name}_seq_start")

    @staticmethod
    def _name_for_seq_end(tensor):
        return Symbol(f"{tensor.source.name}_seq_end")

    @staticmethod
    def _name_for_index(tensor, dim):
        return Symbol(f"{tensor.source.name}_index_{dim}")


def cache_source(source):
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    cache_file = CACHE_DIR / f"{digest}.py"

    if not cache_file.exists():
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(source)

    return cache_file

