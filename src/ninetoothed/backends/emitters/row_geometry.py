"""Source-row geometry required by scalar ownership and vector lowering."""

import re


def tensor_source_shape(info):
    attrs = info.attrs or {}

    return tuple(
        map(
            str,
            attrs.get("source_shape")
            or getattr(info, "source_shape", ())
            or info.shape,
        )
    )


def canonical_row_view(info, source_shape=None) -> bool:
    """Prove that an application row visits exactly one physical source row."""
    source_shape = (
        tensor_source_shape(info)
        if source_shape is None
        else tuple(map(str, source_shape))
    )

    if len(source_shape) != 2:
        return False

    attrs = info.attrs or {}
    templates = attrs.get("access_templates", ())

    if not templates:
        return (
            tuple(map(str, info.shape)) == source_shape
            and not attrs.get("view_offsets")
            and not attrs.get("view_linear_offset")
        )

    if len(templates) != 1:
        return False

    template = templates[0]

    if template.get("level", 0) != 0:
        return False

    tile_shape = tuple(map(str, template.get("shape", ())))
    view_shape = tuple(map(str, attrs.get("view_shape", info.shape)))
    offsets = template.get("offsets", ())

    if len(tile_shape) != 2 or len(view_shape) != 2 or len(offsets) != 2:
        return False

    import sympy

    texts = (
        *source_shape,
        *tile_shape,
        *view_shape,
        *map(str, offsets),
        str(template.get("mask", True)),
    )
    names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", " ".join(texts)))
    functions = {"floor", "ceiling", "Max", "Min", "Mod", "Abs", "True", "False"}
    symbols = {name: sympy.Symbol(name, integer=True) for name in names - functions}
    symbols.setdefault("outer_index", sympy.Symbol("outer_index", integer=True))
    symbols.setdefault("value_0", sympy.Symbol("value_0", integer=True))
    symbols.setdefault("value_1", sympy.Symbol("value_1", integer=True))

    def parse(text):
        return sympy.sympify(text, locals=symbols, evaluate=False)

    try:
        rows, cols = map(parse, source_shape)

        if sympy.simplify(parse(tile_shape[0]) - 1) != 0:
            return False

        if sympy.simplify(parse(view_shape[1]) - 1) != 0:
            return False

        if sympy.simplify(parse(view_shape[0]) - rows) != 0:
            return False

        if sympy.simplify(parse(tile_shape[1]) - cols).is_nonnegative is not True:
            return False

        if (
            attrs.get("other") is not None
            and sympy.simplify(parse(tile_shape[1]) - cols) != 0
        ):
            # Custom padding participates in a wider tile's computation.
            # Exact source-width rows never visit it, so any `other` is safe.
            return False

        substitution = {symbols["value_0"]: 0}

        for offset, expected in zip(
            offsets, (symbols["outer_index"], symbols["value_1"])
        ):
            if sympy.simplify(parse(str(offset)).subs(substitution) - expected) != 0:
                return False

        mask = sympy.simplify(parse(str(template.get("mask", True))).subs(substitution))

        return _row_mask_covers_source(mask, rows, cols, symbols)
    except (
        TypeError,
        ValueError,
        SyntaxError,
        sympy.SympifyError,
        sympy.PolynomialError,
    ):
        return False


def _row_mask_covers_source(mask, rows, cols, symbols):
    """Prove affine mask bounds at the corners of the complete source row domain."""
    import sympy
    from sympy.core.relational import Relational

    if mask == sympy.true:
        return True

    if isinstance(mask, sympy.And):
        return all(
            _row_mask_covers_source(term, rows, cols, symbols) for term in mask.args
        )

    if not isinstance(mask, Relational):
        return False

    if mask.rel_op == "!=":
        return False

    outer = symbols["outer_index"]
    column = symbols["value_1"]
    difference = sympy.simplify(mask.lhs - mask.rhs)

    if sympy.Poly(difference, outer, column).total_degree() > 1:
        return False

    for row in (0, rows - 1):
        for col in (0, cols - 1):
            value = sympy.simplify(difference.subs({outer: row, column: col}))
            proven = {
                "<": value.is_negative,
                "<=": value.is_nonpositive,
                ">": value.is_positive,
                ">=": value.is_nonnegative,
                "==": value.is_zero,
            }.get(mask.rel_op)

            if proven is not True:
                return False

    return True
