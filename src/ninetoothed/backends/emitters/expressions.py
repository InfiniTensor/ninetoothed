"""Pure expression helpers shared by backend emitters."""

import re
from collections.abc import Mapping


def default_strides(shape: tuple[str, ...]) -> tuple[str, ...]:
    strides: list[str] = []
    acc = "1"

    for dim in reversed(shape):
        strides.append(acc)
        acc = dim if is_one_expr(acc) else f"({dim}) * ({acc})"
    return tuple(reversed(strides))


def is_zero_expr(expr: str) -> bool:
    return expr.strip("() ") == "0"


def is_one_expr(expr: str) -> bool:
    return expr.strip("() ") == "1"


def replace_index_symbol(expr: str, value: str) -> str:
    return re.sub(r"\bindex\b", f"({value})", expr)


def replace_symbols(expr: str, replacements: Mapping[str, str]) -> str:
    for name in sorted(replacements, key=len, reverse=True):
        expr = re.sub(rf"\b{re.escape(name)}\b", f"({replacements[name]})", expr)
    return expr


def shape_dim(axes: tuple[str, ...], dim) -> str:
    dim = int(dim or 0)

    if dim < 0:
        dim += len(axes)
    return axes[dim]


def stride_dim(axes: tuple[str, ...], dim) -> str:
    dim = int(dim or 0)

    if dim < 0:
        dim += len(axes)

    if dim >= len(axes):
        return "1"
    return product(axes[dim + 1 :])


def linearized_index(indices: tuple[str, ...], axes: tuple[str, ...]) -> str:
    if len(indices) == 1 and len(axes) <= 1:
        return indices[0]

    terms: list[str] = []

    for position, index in enumerate(indices):
        stride = product(axes[position + 1 :])
        terms.append(f"({index}) * ({stride})" if stride != "1" else f"({index})")
    return " + ".join(terms) if terms else "index"


def product(terms: tuple[str, ...]) -> str:
    items = tuple(str(term) for term in terms if str(term) not in {"", "1"})

    return " * ".join(factor(item) for item in items) if items else "1"


def factor(term: str) -> str:
    return term if valid_symbol(term) or term.isdecimal() else f"({term})"


def rewrite_index_math(expr: str, *, c_style: bool) -> str:
    previous = None
    current = expr

    while current != previous:
        previous = current
        current = _rewrite_named_call(
            current,
            "Mod",
            lambda args: f"(({args[0]}) % ({args[1]}))" if len(args) == 2 else None,
        )
        current = _rewrite_named_call(
            current,
            "floor",
            lambda args: (
                _rewrite_floor_arg(args[0], c_style=c_style) if len(args) == 1 else None
            ),
        )
    return current


def _rewrite_floor_arg(arg: str, *, c_style: bool) -> str:
    split = split_top_level_binary(arg, "/")

    if split is None:
        return f"floor({arg})"

    lhs, rhs = split
    operator = "/" if c_style else "//"

    return f"(({lhs}) {operator} ({rhs}))"


def _rewrite_named_call(expr: str, name: str, render) -> str:
    result: list[str] = []
    cursor = 0
    prefix = f"{name}("

    while True:
        start = expr.find(prefix, cursor)

        if start < 0:
            result.append(expr[cursor:])
            break

        result.append(expr[cursor:start])
        args_start = start + len(prefix)
        args_end = _matching_paren(expr, args_start - 1)

        if args_end is None:
            result.append(expr[start:])
            break

        args = _split_call_args(expr[args_start:args_end])
        rendered = render(args)
        result.append(expr[start : args_end + 1] if rendered is None else rendered)
        cursor = args_end + 1
    return "".join(result)


def _matching_paren(expr: str, open_index: int) -> int | None:
    depth = 0

    for index in range(open_index, len(expr)):
        if expr[index] == "(":
            depth += 1
        elif expr[index] == ")":
            depth -= 1

            if depth == 0:
                return index
    return None


def _split_call_args(args: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0

    for index, char in enumerate(args):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(args[start:index].strip())
            start = index + 1

    tail = args[start:].strip()

    if tail:
        parts.append(tail)
    return parts


def split_top_level_binary(expr: str, operator: str) -> tuple[str, str] | None:
    depth = 0

    for index, char in enumerate(expr):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == operator and depth == 0:
            return expr[:index].strip(), expr[index + 1 :].strip()
    return None


def integer_expr(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+", value))


def valid_symbol(value: str) -> bool:
    return value.isidentifier()


def symbols_in_text(value: str) -> tuple[str, ...]:
    excluded = {
        "True",
        "False",
        "None",
        "index",
        "outer_index",
        "floor",
        "ceil",
        "ceiling",
        "Mod",
    }

    return tuple(
        symbol
        for symbol in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", value)
        if symbol not in excluded
        and not re.fullmatch(r"value_\d+", symbol)
        and not re.fullmatch(r"extract_\d+_\d+", symbol)
    )


def normalize_dtype(dtype: str | None) -> str:
    if dtype is not None:
        dtype = dtype.strip().strip("'\"")

        if "." in dtype:
            dtype = dtype.split(".")[-1]

    mapping = {
        "fp16": "float16",
        "fp32": "float32",
        "fp64": "float64",
        "bf16": "bfloat16",
        "float": "float32",
    }

    return mapping.get(dtype or "float32", dtype or "float32")


__all__ = [
    "default_strides",
    "integer_expr",
    "is_one_expr",
    "is_zero_expr",
    "linearized_index",
    "normalize_dtype",
    "product",
    "replace_index_symbol",
    "replace_symbols",
    "rewrite_index_math",
    "shape_dim",
    "split_top_level_binary",
    "stride_dim",
    "symbols_in_text",
    "valid_symbol",
]
