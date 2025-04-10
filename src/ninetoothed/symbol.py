import ast
import inspect
import numbers
import types

import ninetoothed.naming as naming


class Symbol:
    """A class uesed to represent a symbol.

    :param expr: The expression used to construct the symbol.
    :param constexpr: Whether the symbol is a constexpr.
    :param mata: Whether the symbol is a meta.
    :param lower_bound: The minimum value for the symbol's range.
    :param upper_bound: The maximum value for the symbol's range.
    :param power_of_two: Whether the value should be a power of two.
    """

    def __init__(
        self,
        expr,
        constexpr=None,
        meta=None,
        lower_bound=None,
        upper_bound=None,
        power_of_two=None,
    ):
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

            self._node.id = naming.make_meta(self._node.id)

        if constexpr:
            self._node.id = naming.make_constexpr(self._node.id)

        self._node.symbol = self

        DEFAULT_LOWER_BOUND_FOR_META_SYMBOLS = 2**5
        DEFAULT_UPPER_BOUND_FOR_META_SYMBOLS = 2**10
        DEFAULT_POWER_OF_TWO_FOR_META_SYMBOLS = True

        DEFAULT_LOWER_BOUND_FOR_NON_META_CONSTEXPR_SYMBOLS = 1
        DEFAULT_UPPER_BOUND_FOR_NON_META_CONSTEXPR_SYMBOLS = 2**20
        DEFAULT_POWER_OF_TWO_FOR_NON_META_CONSTEXPR_SYMBOLS = False

        if lower_bound is not None:
            self.lower_bound = lower_bound
        else:
            if meta:
                self.lower_bound = DEFAULT_LOWER_BOUND_FOR_META_SYMBOLS
            elif constexpr:
                self.lower_bound = DEFAULT_LOWER_BOUND_FOR_NON_META_CONSTEXPR_SYMBOLS

        if upper_bound is not None:
            self.upper_bound = upper_bound
        else:
            if meta:
                self.upper_bound = DEFAULT_UPPER_BOUND_FOR_META_SYMBOLS
            elif constexpr:
                self.upper_bound = DEFAULT_UPPER_BOUND_FOR_NON_META_CONSTEXPR_SYMBOLS

        if power_of_two is not None:
            self.power_of_two = power_of_two
        else:
            if meta:
                self.power_of_two = DEFAULT_POWER_OF_TWO_FOR_META_SYMBOLS
            elif constexpr:
                self.power_of_two = DEFAULT_POWER_OF_TWO_FOR_NON_META_CONSTEXPR_SYMBOLS

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

    def __sub__(self, other):
        other = type(self)(other)

        if self == 0:
            return -other

        if other == 0:
            return self

        return type(self)(ast.BinOp(left=self._node, op=ast.Sub(), right=other._node))

    def __rsub__(self, other):
        return type(self)(other).__sub__(self)

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
        if isinstance(target, tuple):
            targets = tuple(item.node for item in target)
        else:
            targets = (target.node,)

        return Symbol(_FindAndReplacer(targets, replacement.node).visit(self._node))

    def names(self):
        class NameCollector(ast.NodeVisitor):
            def __init__(self):
                self.names = set()

            def visit_Name(self, node):
                self.generic_visit(node)

                self.names.add(node.symbol)

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
    def is_name(object):
        return isinstance(object, Symbol) and isinstance(object.node, ast.Name)


def block_size(lower_bound=None, upper_bound=None):
    """Create a block size symbol that serves as a meta-parameter.

    :param lower_bound: The lower bound for the block size's range.
    :param upper_bound: The upper bound for the block size's range.
    :return: A block size symbol that serves as a meta-parameter.
    """

    name = naming.auto_generate(f"BLOCK_SIZE_{block_size._num_block_sizes}")

    block_size._num_block_sizes += 1

    return Symbol(name, meta=True, lower_bound=lower_bound, upper_bound=upper_bound)


block_size._num_block_sizes = 0


class _FindAndReplacer(ast.NodeTransformer):
    def __init__(self, targets, replacement):
        self._targets_unparsed = tuple(
            sorted({ast.unparse(target) for target in targets}, key=len, reverse=True)
        )
        self._replacement = replacement

    def visit(self, node):
        if ast.unparse(node) in self._targets_unparsed:
            return self._replacement

        return super().visit(node)
