import ast
import inspect
import types


class Symbol:
    def __init__(self, expr):
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

    def __add__(self, other):
        return type(self)(
            ast.BinOp(left=self._node, op=ast.Add(), right=type(self)(other)._node)
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        return type(self)(
            ast.BinOp(left=self._node, op=ast.Mult(), right=type(self)(other)._node)
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __floordiv__(self, other):
        return type(self)(
            ast.BinOp(left=self._node, op=ast.FloorDiv(), right=type(self)(other)._node)
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
