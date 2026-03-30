import ast
import inspect

import ninetoothed.naming as naming
from ninetoothed.language import Symbol
from ninetoothed.symbol import Symbol as SymbolClass
from ninetoothed.tensor import Tensor
from triton.language.extra import libdevice

__all__ = [
    "_AliasRestorer",
    "_BinOpSimplifier",
    "_FunctionDefFinder",
    "_Inliner",
    "_NextPowerOfTwoMaker",
    "_SimplifiedNameCollector",
    "_TupleSliceRemover",
    "Cudaifier",
    "MetaEncloser",
    "Torchifier",
]


class _AliasRestorer(ast.NodeTransformer):
    def __init__(self, aliases):
        super().__init__()

        self._aliases = aliases

        self._redefined = set()

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


class _BinOpSimplifier(ast.NodeTransformer):
    def visit_BinOp(self, node):
        self.generic_visit(node)

        if isinstance(node.op, ast.Mult):
            left = Symbol(node.left)
            right = Symbol(node.right)

            if left == 1:
                return node.right

            if right == 1:
                return node.left

        return node


class _FunctionDefFinder(ast.NodeVisitor):
    def __init__(self, name):
        self._name = name

        self.result = None

    def visit_FunctionDef(self, node):
        if node.name == self._name:
            self.result = node

        self.generic_visit(node)


class _Inliner(ast.NodeTransformer):
    LIBDEVICE_ALIAS = naming.auto_generate("libdevice")

    def __init__(self, globals):
        self.libdevice_used = False

        self._globals = globals

        self._count = 0

    def visit(self, node):
        def _find_aliases():
            aliases = {}

            for name, value in self._globals.items():
                if inspect.ismodule(value):
                    if value is libdevice:
                        aliases[name] = self.LIBDEVICE_ALIAS
                        self.libdevice_used = True

                        continue

                    aliases[name] = value.__name__

            return aliases

        node = super().visit(node)

        alias_restorer = _AliasRestorer(_find_aliases())

        if isinstance(node, list):
            node = [alias_restorer.visit(item) for item in node]
        else:
            node = alias_restorer.visit(node)

        return node

    def visit_Expr(self, node):
        value, stmts = self._inline_expr(node.value)
        node.value = value
        node = self.generic_visit(node)

        if stmts:
            if isinstance(value, ast.Constant) and value.value is None:
                return stmts

            return stmts + [node]

        return node

    def visit_Assign(self, node):
        value, stmts = self._inline_expr(node.value)
        node.value = value
        node = self.generic_visit(node)

        if stmts:
            return stmts + [node]

        return node

    def visit_Return(self, node):
        if node.value:
            value, stmts = self._inline_expr(node.value)
            node.value = value

            if stmts:
                return stmts + [node]

        return node

    def _inline_expr(self, expr):
        def _inline_list(lst):
            new_list = []
            new_stmts = []

            for expr in lst:
                expr, stmts = self._inline_expr(expr)

                new_list.append(expr)
                new_stmts.extend(stmts)

            return new_list, new_stmts

        def _inline_field(field):
            if isinstance(field, ast.AST):
                return self._inline_expr(field)

            return field, []

        if isinstance(expr, ast.Call):
            new_expr, new_stmts = self._inline_call(expr)

            if new_expr is not None:
                return new_expr, new_stmts

        new_stmts = []

        for field, value in ast.iter_fields(expr):
            if isinstance(value, list):
                new_value, new_stmts = _inline_list(value)
            else:
                new_value, new_stmts = _inline_field(value)

            setattr(expr, field, new_value)
            new_stmts.extend(new_stmts)

        return expr, new_stmts

    def _inline_call(self, node):
        class _ParameterReplacer(ast.NodeTransformer):
            def __init__(self, mapping):
                self._mapping = mapping

            def visit_Name(self, node):
                return self._mapping.get(node.id, node)

        class _LocalVariableRenamer(ast.NodeTransformer):
            def __init__(self, prefix, local_vars):
                self._prefix = prefix

                self._local_vars = local_vars

            def visit_Name(self, node):
                if node.id in self._local_vars:
                    node.id = f"{self._prefix}{node.id}"

                return node

            def visit_arg(self, node):
                return node

        def _resolve_function(node, globals):
            if isinstance(node, ast.Name):
                return globals.get(node.id)

            if isinstance(node, ast.Attribute):
                obj = _resolve_function(node.value, globals)

                if obj is not None:
                    return getattr(obj, node.attr, None)

            return None

        def _get_source(func):
            try:
                return inspect.getsource(func)
            except TypeError:
                return None

        def _find_function_definition(source):
            finder = _FunctionDefFinder(func.__name__)
            finder.visit(ast.parse(source))

            return finder.result

        def _find_assigned_names(stmts):
            class _AssignedNameFinder(ast.NodeVisitor):
                def __init__(self):
                    self.result = set()

                def visit_Name(self, node):
                    if isinstance(node.ctx, ast.Store):
                        self.result.add(node.id)

            names = set()

            for stmt in stmts:
                finder = _AssignedNameFinder()
                finder.visit(stmt)
                names |= finder.result

            return names

        def _make_temporary():
            prefix = f"{naming.auto_generate(f'temporary_{self._count}')}_"
            self._count += 1

            return prefix

        func = _resolve_function(node.func, self._globals)

        if func is None:
            return None, []

        source = _get_source(func)

        if source is None:
            return None, []

        func_def = _find_function_definition(source)

        if func_def is None:
            return None, []

        if inspect.getmodule(func) is libdevice:
            return None, []

        param_names = [arg.arg for arg in func_def.args.args]

        mapping = {param: arg for param, arg in zip(param_names, node.args)}
        param_replacer = _ParameterReplacer(mapping)
        body = [param_replacer.visit(stmt) for stmt in func_def.body]

        local_vars = _find_assigned_names(body) - set(param_names)
        prefix = _make_temporary()
        local_var_renamer = _LocalVariableRenamer(prefix, local_vars)
        body = [local_var_renamer.visit(stmt) for stmt in body]

        inlined_body = []

        inliner = _Inliner(func.__globals__)

        for stmt in body:
            inlined_stmt = inliner.visit(stmt)

            if isinstance(inlined_stmt, list):
                inlined_body.extend(inlined_stmt)
            else:
                inlined_body.append(inlined_stmt)

        if not inlined_body or not isinstance(inlined_body[-1], ast.Return):
            return ast.Constant(value=None), inlined_body

        ret = inlined_body.pop()
        temp = _make_temporary()
        assignment = ast.Assign(
            targets=[ast.Name(id=temp, ctx=ast.Store())], value=ret.value
        )
        inlined_body.append(assignment)

        return ast.Name(id=temp, ctx=ast.Load()), inlined_body


class _SimplifiedNameCollector(ast.NodeVisitor):
    def __init__(self):
        self.simplified_names = {}

    def visit_Name(self, node):
        self.generic_visit(node)

        self.simplified_names[node.id] = naming.remove_prefixes(node.id)


class Cudaifier(ast.NodeTransformer):
    def visit_Name(self, node):
        self.generic_visit(node)

        source = node.id

        if naming.is_constexpr(source):
            if not Tensor.size_pattern().fullmatch(source):
                return node

            source = naming.remove_prefixes(source)

        def repl(match):
            return f"{match.group(1)}.data"

        source = Tensor.pointer_pattern().sub(repl, source)

        def repl(match):
            return f"{match.group(1)}.shape[{match.group(3)}]"

        source = Tensor.size_pattern().sub(repl, source)

        def repl(match):
            return f"{match.group(1)}.strides[{match.group(3)}]"

        source = Tensor.stride_pattern().sub(repl, source)

        source = source.removesuffix("_with_auto_tuning")

        if source != node.id:
            return ast.parse(source, mode="eval").body

        return node


class Torchifier(ast.NodeTransformer):
    def visit_Name(self, node):
        self.generic_visit(node)

        source = node.id

        if naming.is_constexpr(source):
            return node

        def repl(match):
            return f"{match.group(1)}"

        source = Tensor.pointer_pattern().sub(repl, source)

        def repl(match):
            return f"{match.group(1)}.{match.group(2)}({match.group(3)})"

        source = Tensor.size_pattern().sub(repl, source)
        source = Tensor.stride_pattern().sub(repl, source)

        def repl(match):
            return f"{match.group(1)}.{match.group(2)}()"

        source = Tensor.values_pattern().sub(repl, source)
        source = Tensor.offsets_pattern().sub(repl, source)

        def repl(match):
            return f"{match.group(1)}.offsets().diff().max().item()"

        source = Tensor.max_seq_len_pattern().sub(repl, source)
        source = Tensor.seq_len_pattern().sub(repl, source)

        if source != node.id:
            return ast.parse(source, mode="eval").body

        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)

        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "ninetoothed"
            and node.attr == "language"
        ):
            return node.value

        return node


class _TupleSliceRemover(ast.NodeTransformer):
    def visit_Subscript(self, node):
        self.generic_visit(node)

        if isinstance(node.slice, ast.Tuple):
            return node.value

        return node


class _NextPowerOfTwoMaker(ast.NodeTransformer):
    def visit_Name(self, node):
        name = node.id

        if not naming.is_meta(name):
            next_power_of_2_name = naming.make_next_power_of_2(name)

            return Symbol(next_power_of_2_name).node

        return self.generic_visit(node)


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


