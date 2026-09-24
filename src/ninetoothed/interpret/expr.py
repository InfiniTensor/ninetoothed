"""Safe evaluation of the symbolic expressions stored in SSA attributes.

The NineToothed frontend records layout and index mappings as text, for example
``"(outer_index // 4 + extract_0_0) * ninetoothed_x_stride_0"``.  The
interpreter has to turn that text back into numbers.  Instead of handing the
string to :func:`eval`, the text is parsed once, checked against a whitelist of
node types, and compiled.  Arbitrary attribute access, imports, and calls are
rejected.
"""

import ast
import math
from functools import lru_cache

import numpy as np

from .errors import InterpreterError

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.BitAnd,
    ast.BitOr,
    ast.BitXor,
    ast.LShift,
    ast.RShift,
    ast.And,
    ast.Or,
    ast.USub,
    ast.UAdd,
    ast.Invert,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)

_FUNCTIONS = {
    "abs": abs,
    "ceil": math.ceil,
    "floor": math.floor,
    "max": max,
    "min": min,
    "int": int,
    "float": float,
    "bool": bool,
    "round": round,
}


class ExpressionError(InterpreterError):
    """Raised when an SSA attribute expression cannot be evaluated."""


class Expression:
    """A parsed, whitelisted arithmetic expression."""

    __slots__ = ("_code", "_names", "_text")

    def __init__(self, text, *, code, names):
        self._text = text
        self._code = code
        self._names = names

    @property
    def text(self):
        """Return the original expression text."""
        return self._text

    @property
    def names(self):
        """Return the free variable names referenced by the expression."""
        return self._names

    def evaluate(self, namespace):
        """Evaluate the expression.

        :param namespace: A mapping providing the free variables.
        :return: The numeric result (possibly a ``numpy.ndarray``).
        """
        scope = dict(_FUNCTIONS)
        scope["np"] = np
        scope["math"] = math
        scope.update(namespace)

        missing = self._names - scope.keys()

        if missing:
            raise ExpressionError(
                f"Cannot evaluate `{self._text}`: undefined symbol(s) "
                f"{', '.join(sorted(missing))}."
            )

        try:
            return eval(self._code, {"__builtins__": {}}, scope)
        except Exception as exc:  # pragma: no cover - defensive
            raise ExpressionError(f"Cannot evaluate `{self._text}`: {exc}.") from exc

    def __repr__(self):
        return f"Expression({self._text!r})"


@lru_cache(maxsize=4096)
def parse_expression(text):
    """Parse an expression string into a reusable :class:`Expression`.

    :param text: The expression text.
    :return: A cached :class:`Expression`.
    """
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"Cannot parse expression `{text}`: {exc}.") from exc

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ExpressionError(
                f"Expression `{text}` uses unsupported syntax `{type(node).__name__}`."
            )

    names = frozenset(
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    ) - set(_FUNCTIONS)

    return Expression(
        text,
        code=compile(tree, "<ninetoothed-expression>", "eval"),
        names=names,
    )


def evaluate(text, namespace):
    """Parse and evaluate ``text`` in one call."""
    return parse_expression(str(text)).evaluate(namespace)


def names_of(text):
    """Return the free variable names of an expression without evaluating it."""
    return parse_expression(str(text)).names


__all__ = ["Expression", "ExpressionError", "evaluate", "names_of", "parse_expression"]
