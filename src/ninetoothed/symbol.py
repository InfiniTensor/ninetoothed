import ast
import inspect
import numbers
import types


class Symbol:
    def __init__(self, expr, constexpr=None, meta=None):
        if isinstance(expr, type(self)):
            self._node = expr._node
            return

        if isinstance(expr, ast.AST):
            self._node = expr
            return

        if isinstance(expr, types.CodeType):
            expr = inspect.getsource(expr)

        if not isinstance(expr, str):
            expr = str(expr)

        self._node = ast.parse(expr, mode="eval").body

        if (constexpr or meta) and not isinstance(self._node, ast.Name):
            raise ValueError("`constexpr` and `meta` are properties of name symbols.")

        if meta:
            if constexpr is False:
                raise ValueError("Non-constexpr meta symbol is not supported.")

            self._node.id = type(self)._create_meta(self._node.id)

        if constexpr:
            self._node.id = type(self)._create_constexpr(self._node.id)

    def __eq__(self, other):
        if isinstance(self._node, ast.Constant):
            if isinstance(other, Symbol) and isinstance(other._node, ast.Constant):
                return self._node.value == other._node.value

            if isinstance(other, numbers.Number):
                return self._node.value == other

        return False

    def __hash__(self):
        return id(self)

    def __add__(self, other):
        other = type(self)(other)

        if self == 0:
            return other

        if other == 0:
            return self

        return type(self)(ast.BinOp(left=self._node, op=ast.Add(), right=other._node))

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = type(self)(other)

        if self == 0 or other == 0:
            return type(self)(0)

        if self == 1:
            return other

        if other == 1:
            return self

        return type(self)(ast.BinOp(left=self._node, op=ast.Mult(), right=other._node))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __floordiv__(self, other):
        other = type(self)(other)

        if isinstance(other._node, ast.Constant) and other._node.value == 1:
            return self

        return type(self)(
            ast.BinOp(left=self._node, op=ast.FloorDiv(), right=other._node)
        )

    def __mod__(self, other):
        other = type(self)(other)

        return type(self)(ast.BinOp(left=self._node, op=ast.Mod(), right=other._node))

    def __lt__(self, other):
        other = type(self)(other)

        return type(self)(
            ast.Compare(left=self._node, ops=[ast.Lt()], comparators=[other._node])
        )

    def __and__(self, other):
        other = type(self)(other)

        return type(self)(
            ast.BinOp(left=self._node, op=ast.BitAnd(), right=other._node)
        )

    def __rand__(self, other):
        return self.__and__(other)

    def __getitem__(self, key):
        return type(self)(ast.Subscript(value=self._node, slice=type(self)(key)._node))

    def __repr__(self):
        return ast.unparse(self._node)

    def find_and_replace(self, target, replacement):
        _FindAndReplacer(target.node, replacement.node).visit(self._node)

    def names(self):
        class NameCollector(ast.NodeVisitor):
            def __init__(self):
                self.names = set()

            def visit_Name(self, node):
                self.generic_visit(node)

                self.names.add(node.id)

        name_collector = NameCollector()

        name_collector.visit(self._node)

        return name_collector.names

    @property
    def node(self):
        class SliceSimplifier(ast.NodeTransformer):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id == "slice":
                    return ast.Slice(*node.args)

                return node

        return SliceSimplifier().visit(self._node)

    @staticmethod
    def is_constexpr(name):
        return name.startswith(Symbol._constexpr_prefix()) or Symbol.is_meta(name)

    @staticmethod
    def is_meta(name):
        return name.startswith(Symbol._meta_prefix())

    @staticmethod
    def remove_prefix(name):
        if name.startswith(Symbol._constexpr_prefix()):
            return name.removeprefix(Symbol._constexpr_prefix())

        if name.startswith(Symbol._meta_prefix()):
            return name.removeprefix(Symbol._meta_prefix())

    @staticmethod
    def _create_constexpr(name):
        return f"{Symbol._constexpr_prefix()}{name}"

    @staticmethod
    def _create_meta(name):
        return f"{Symbol._meta_prefix()}{name}"

    @staticmethod
    def _constexpr_prefix():
        return f"{Symbol._ninetoothed_prefix()}constexpr_"

    @staticmethod
    def _meta_prefix():
        return f"{Symbol._ninetoothed_prefix()}meta_"

    @staticmethod
    def _ninetoothed_prefix():
        return "_ninetoothed_"


class _FindAndReplacer(ast.NodeTransformer):
    def __init__(self, target, replacement):
        self._target_id = target.id
        self._replacement = replacement

    def visit_Name(self, node):
        if node.id == self._target_id:
            return self._replacement

        return self.generic_visit(node)
