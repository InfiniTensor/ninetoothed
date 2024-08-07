import ast
import inspect
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

    def __add__(self, other):
        other = type(self)(other)

        if isinstance(self._node, ast.Constant) and self._node.value == 0:
            return other

        if isinstance(other._node, ast.Constant) and other._node.value == 0:
            return self

        return type(self)(ast.BinOp(left=self._node, op=ast.Add(), right=other._node))

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other = type(self)(other)

        if isinstance(self._node, ast.Constant) and self._node.value == 0:
            return type(self)(0)

        if isinstance(other._node, ast.Constant) and other._node.value == 0:
            return type(self)(0)

        if isinstance(self._node, ast.Constant) and self._node.value == 1:
            return other

        if isinstance(other._node, ast.Constant) and other._node.value == 1:
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
        return type(self)(
            ast.BinOp(left=self._node, op=ast.Mod(), right=type(self)(other)._node)
        )

    def __getitem__(self, key):
        return type(self)(ast.Subscript(value=self._node, slice=type(self)(key)._node))

    def __repr__(self):
        return ast.unparse(self._node)

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
        return name.startswith("_ninetoothed_constexpr_") or Symbol.is_meta(name)

    @staticmethod
    def is_meta(name):
        return name.startswith("_ninetoothed_meta_")

    @staticmethod
    def _create_constexpr(name):
        return f"_ninetoothed_constexpr_{name}"

    @staticmethod
    def _create_meta(name):
        return f"_ninetoothed_meta_{name}"
