import ast
import functools
import inspect
import itertools
import math
import tempfile
import textwrap

from ninetoothed.language import attribute, call
from ninetoothed.symbol import Symbol
from ninetoothed.tensor import Tensor
from ninetoothed.torchifier import Torchifier


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
            else ast.arg(arg=name, annotation=attribute("constexpr"))
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
                if isinstance(node.slice, ast.Tuple):
                    indices = value.indices() + tuple(node.slice.elts)
                else:
                    indices = value.indices() + (node.slice,)
                offsets = value.offsets(indices)
                pointers = value.pointers(offsets)

                return call("load", pointers).node

        self.generic_visit(node)

        return node

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id in self._context:
            value = self._context[node.value.id]

            if isinstance(value, Tensor):
                inner = value.dtype

                return Symbol(inner.__dict__[node.attr]).node

        self.generic_visit(node)

        return node

    def visit_Name(self, node):
        self.generic_visit(node)

        if node.id in self._context and isinstance(node.ctx, ast.Load):
            return call("load", self._context[node.id].pointers().node).node

        return node

    def visit_Assign(self, node):
        if len(node.targets) == 1:
            target = node.targets[0]

            if isinstance(target, ast.Name) and target.id in self._context:
                self.generic_visit(node)

                return ast.Expr(
                    call(
                        "store",
                        self._context[target.id].pointers().node,
                        node.value,
                    ).node
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

                    indices = value.indices() + tuple(
                        target.slice.elts
                        if isinstance(target.slice, ast.Tuple)
                        else target.slice
                    )
                    offsets = value.offsets(indices)
                    pointers = value.pointers(offsets)

                    return ast.Expr(
                        call(
                            "store",
                            pointers.node,
                            node.value,
                        ).node
                    )

        self.generic_visit(node)

        return node

    def _generate_autotune(self, params, meta):
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
                keywords=[],
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
                            if not Tensor.is_pointer(param)
                        ],
                        ctx=ast.Load(),
                    ),
                ),
            ],
        )

    def _generate_launch(self, params, meta):
        launch = ast.FunctionDef(
            name=f"launch_{self._func_def.name}",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg.name) for arg in self._args],
                kwonlyargs=[],
                defaults=[],
            ),
            body=[
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


class Tritonizer(ast.NodeTransformer):
    def visit_Module(self, node):
        self.generic_visit(node)

        node.body.insert(0, ast.Import(names=[ast.alias(name="triton.language")]))
        node.body.insert(0, ast.Import(names=[ast.alias(name="triton")]))

        return node

    def visit_Name(self, node):
        self.generic_visit(node)

        if node.id == "ninetoothed":
            node.id = "triton"

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


def jit(func):
    source = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(source)

    CodeGenerator(func.__annotations__).visit(tree)
    Tritonizer().visit(tree)
    ast.fix_missing_locations(tree)

    unparsed = ast.unparse(tree).replace("None:", ":").replace(":None", ":")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
        temp_file.write(unparsed.encode("utf-8"))
        temp_file_name = temp_file.name

    with open(temp_file_name, "r") as temp_file:
        code = compile(source=temp_file.read(), filename=temp_file_name, mode="exec")

    namespace = {}
    exec(code, namespace)

    class Handle:
        def __init__(self, kernel, launch):
            self._kernel = kernel
            self._launch = launch

        def __call__(self, *args, **kwargs):
            return self._launch(*args, **kwargs)

    return Handle(namespace[func.__name__], namespace[f"launch_{func.__name__}"])
