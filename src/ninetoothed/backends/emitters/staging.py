"""Shared staging analysis for C-style backends with on-chip memory.

Determines whether a kernel body benefits from staging data through
fast on-chip memory (CUDA shared memory, BangC NRAM, AscendC UB) before
computation, and prepares the body for staged rendering.
"""

import re

from ninetoothed.backends.emitters.base import ModuleRenderContext


def split_stride_guard(body: str) -> tuple[str | None, str]:
    """Split an ``if (stride == 1) { fast } else { generic }`` wrapper.

    Returns ``(predicate, fast_branch_body)`` or ``(None, body)``.
    """
    stripped = body.strip()

    if not stripped.startswith("if ("):
        return None, body

    open_paren = stripped.index("(")
    depth = 0
    predicate_end = None

    for i in range(open_paren, len(stripped)):
        if stripped[i] == "(":
            depth += 1
        elif stripped[i] == ")":
            depth -= 1

            if depth == 0:
                predicate_end = i
                break

    if predicate_end is None:
        return None, body

    predicate = stripped[open_paren + 1 : predicate_end]
    brace_start = stripped.index("{", predicate_end)
    depth = 0
    split_at = None

    for i in range(brace_start, len(stripped)):
        if stripped[i] == "{":
            depth += 1
        elif stripped[i] == "}":
            depth -= 1

            if depth == 0:
                remainder = stripped[i + 1 :]

                if remainder.startswith(" else {"):
                    split_at = i
                    break

                break

    if split_at is None:
        fast = stripped[brace_start + 1 : stripped.rindex("}")]

        return predicate, _dedent(fast)

    fast = stripped[brace_start + 1 : split_at]

    return predicate, _dedent(fast)


def _dedent(text: str) -> str:
    newline = chr(10)
    lines = text.split(newline)

    while lines and not lines[0].strip():
        lines.pop(0)

    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        return ""

    indent = len(lines[0]) - len(lines[0].lstrip())

    return newline.join(line[indent:] if len(line) >= indent else "" for line in lines)


_PRED_DECL = re.compile(r"bool (nt_pred_\d+) = ([^;]+);")


def canonical_flat_view(info) -> bool:
    """Prove that a staged view covers the complete physical 1-D prefix."""
    attrs = info.attrs or {}
    source_shape = attrs.get("source_shape") or info.source_shape or info.shape

    if info.ndim != 1 or len(source_shape) != 1:
        return False

    if attrs.get("jagged_offsets_param"):
        return False

    templates = attrs.get("access_templates", ())

    if attrs.get("dtype_level", 0) != 0:
        return False

    if not templates:
        return (
            tuple(map(str, info.shape)) == tuple(map(str, source_shape))
            and info.view_mask in {None, "True"}
            and not (info.view_linear_offset or attrs.get("view_offsets"))
        )

    if len(templates) != 1:
        return False

    template = templates[0]
    shape = template.get("shape", ())
    view_shape = attrs.get("view_shape", info.shape)
    offsets = template.get("offsets", ())

    if template.get("level", 0) != 0 or any(
        len(items) != 1 for items in (shape, view_shape, offsets)
    ):
        return False

    import sympy

    texts = (*source_shape, *shape, *view_shape, *offsets, template.get("mask", True))
    names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", " ".join(map(str, texts))))
    functions = {"floor", "ceiling", "Max", "Min", "Mod", "Abs", "True", "False"}
    positive_dimensions = {
        str(dimension)
        for dimension in (*source_shape, *shape)
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(dimension))
    }
    symbols = {
        name: sympy.Symbol(
            name,
            integer=True,
            **({"positive": True} if name in positive_dimensions else {}),
        )
        for name in names - functions - {"outer_index", "value_0"}
    }
    outer = symbols["outer_index"] = sympy.Symbol("outer_index", integer=True)
    inner = symbols["value_0"] = sympy.Symbol("value_0", integer=True)

    def parse(text):
        return sympy.sympify(str(text), locals=symbols)

    try:
        extent, block, count = map(parse, (source_shape[0], shape[0], view_shape[0]))

        if block.is_positive is not True:
            return False

        expected_count = sympy.floor((extent - 1) / block) + 1

        if sympy.simplify(count - expected_count) != 0:
            return False

        flat = outer * block + inner

        if sympy.simplify(parse(offsets[0]) - flat) != 0:
            return False

        allowed = (
            outer >= 0,
            outer < count,
            inner >= 0,
            inner < block,
            flat >= 0,
            flat < extent,
        )
        mask = parse(template.get("mask", True))

        if info.view_mask:
            mask &= parse(re.sub(r"\bindex\b", "outer_index", info.view_mask))

        terms = mask.args if isinstance(mask, sympy.And) else (mask,)

        return all(
            term == sympy.true
            or any(
                getattr(term, "rel_op", None) == getattr(bound, "rel_op", None)
                and sympy.simplify(term.lhs - bound.lhs) == 0
                and sympy.simplify(term.rhs - bound.rhs) == 0
                for bound in allowed
            )
            for term in terms
        )
    except (TypeError, ValueError, SyntaxError, AttributeError, sympy.SympifyError):
        return False


def _predicates_follow_templates(body, context):
    """Require each erased predicate to come from a proven layout bound."""
    import sympy

    expected = []

    for name in (*context.variables, *context.outputs):
        info = context.tensors.get(name)

        if info is None or info.ndim == 0:
            continue

        for template in (info.attrs or {}).get("access_templates", ()):
            expected.append(
                str(template.get("mask", True))
                .replace("outer_index", "nt_outer_index")
                .replace("value_0", "nt_inner_index")
            )

    expressions = [expression for _, expression in _PRED_DECL.findall(body)]
    names = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", " ".join(expected + expressions)))
    functions = {"floor", "ceiling", "Max", "Min", "Mod", "Abs", "True", "False"}
    symbols = {name: sympy.Symbol(name, integer=True) for name in names - functions}

    def terms(expression):
        value = sympy.sympify(expression, locals=symbols)

        return value.args if isinstance(value, sympy.And) else (value,)

    try:
        allowed = {term for expression in expected for term in terms(expression)}

        return all(
            term == sympy.true or term in allowed
            for expression in expressions
            for term in terms(expression.replace("/", "//"))
        )
    except (TypeError, ValueError, SyntaxError, sympy.SympifyError):
        return False


def flatten_tiled_accesses(
    body: str, context: ModuleRenderContext, *, allow_value_select: bool = False
) -> str | None:
    """Rewrite the tiled predicated body into the flat ``name[index]`` form.

    Contiguous 1-D tiled renders access tensors as
    ``name[(nt_outer_index) * BLOCK + (nt_inner_index)]`` guarded by
    ``nt_pred`` bounds checks.  Inside one staged chunk those predicates are
    subsumed by ``nt_j < nt_cnt`` only for views proven to cover the complete
    physical prefix, with canonical tile/source bounds and zero masked-load
    fallbacks. Interior masks, sliced views, and unknown mappings retain the
    generic scalar path.

    A caller with a complete typed def-use plan may retain value-select
    ternaries. Unresolved memory predicates and coordinates are still rejected.
    """
    if any(
        info.ndim != 0 and not canonical_flat_view(info)
        for name in (*context.variables, *context.outputs)
        if (info := context.tensors.get(name)) is not None
    ):
        return None

    if "nt_pred" not in body:
        return body

    # Inline coordinate aliases first: `nt_i0 = nt_inner_index` and
    # `nt_idx_N = (nt_outer_index) * B + X` are CSE locals for the plain
    # tiled coordinates; without inlining them the accesses below never
    # match the canonical forms.
    newline = chr(10)
    body = re.sub(
        r"int64_t nt_i\d+ = nt_inner_index;" + re.escape(newline),
        "",
        body,
    )
    body = re.sub(r"\bnt_i\d+\b", "nt_inner_index", body)

    alias_pattern = re.compile(
        r"int64_t (nt_idx_\d+) = \(nt_outer_index\) \* ([\w]+) \+ \((\w+)\);"
    )

    for alias, block_expr, var in alias_pattern.findall(body):
        decl = (
            "int64_t "
            + alias
            + " = (nt_outer_index) * "
            + block_expr
            + " + ("
            + var
            + ");"
        )
        body = body.replace(decl + newline, "")
        body = body.replace(
            alias,
            "(nt_outer_index) * " + block_expr + " + (" + var + ")",
        )

    if "nt_idx_" in body:
        return None

    allowed_symbols = {
        "nt_outer_index",
        "nt_inner_index",
        "index",
        "true",
        "false",
        *context.shape_params,
    }

    for _, expression in _PRED_DECL.findall(body):
        symbols = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression))

        if not symbols <= allowed_symbols:
            return None

    if not _predicates_follow_templates(body, context):
        return None

    tensor_names = sorted((*context.variables, *context.outputs), key=len, reverse=True)
    flattened = body

    for name in tensor_names:
        tiled = re.compile(
            rf"\b{re.escape(name)}\[+\(*nt_outer_index\) \* \(?([\w]+)\)? \+ \(+nt_inner_index\)\)*\]+"
        )

        if not tiled.search(flattened):
            continue

        flattened = tiled.sub(f"{name}[__nt_flat__]", flattened)

    # Masked loads with a zero fallback collapse to the load itself.
    # Guards appear either per operand (cast zero fallbacks) or as one
    # compound predicate bundling several accesses (bare zero fallback).
    flattened = re.sub(
        r"\(\(nt_pred_\d+\) \? \((\w+)\[__nt_flat__\]\) : \(static_cast<(?:float|double|int32_t|int64_t)>\(0(?:\.0)?f?\)\)\)",
        r"\1[__nt_flat__]",
        flattened,
    )
    flattened = re.sub(
        r"\(\(nt_pred_\d+\) \? \((\w+)\[__nt_flat__\]\) : \(\(\((?:float|int32_t|int64_t)\)\((?:0(?:\.0)?|0\.0f)\)\)\)\)",
        r"\1[__nt_flat__]",
        flattened,
    )
    flattened = re.sub(
        r"\(+\(nt_pred_\d+\)(?: && \(nt_pred_\d+\))*\)+ \? \((.+?)\) : 0(?:\.0)?f?(?:\)+)",
        r"(\1)",
        flattened,
    )
    flattened = re.sub(
        r"\(\(nt_pred_\d+\) \? \((\w+)\[__nt_flat__\]\) : (?:0(?:\.0)?f?)\)",
        r"\1[__nt_flat__]",
        flattened,
    )
    # Masked stores drop their (bounds-only) predicate.
    flattened = re.sub(
        r"if \(nt_pred_\d+\) \{\n(\s+)(\w+)\[__nt_flat__\] = ([^;]+);\n\s*\}",
        r"\1\2[__nt_flat__] = \3;",
        flattened,
    )
    # Predicate declarations are no longer referenced.
    flattened = _PRED_DECL.sub("", flattened)

    # Tiling coordinate declarations become dead once accesses are flat.
    flattened = re.sub(r"int64_t nt_outer_index = [^;]+;\n", "", flattened)
    flattened = re.sub(r"int64_t nt_inner_index = [^;]+;\n", "", flattened)
    flattened = re.sub(r"int64_t nt_i0 = nt_inner_index;\n", "", flattened)

    residue = flattened

    for name in tensor_names:
        residue = residue.replace(f"{name}[__nt_flat__]", "")

    if (
        "nt_outer_index" in flattened
        or "nt_inner_index" in flattened
        or "nt_pred" in flattened
        or ("? (" in residue and not allow_value_select)
        or "__nt_flat__" in residue
    ):
        return None

    return flattened.replace("__nt_flat__", "index")


def staging_extent(
    context: ModuleRenderContext,
    staged_tensors: list[tuple[str, str]],
) -> str | None:
    """Return one shared runtime extent for all staged tensors."""
    extents: set[str] = set()

    for name, _ in staged_tensors:
        info = context.tensors.get(name)

        if info is None or info.ndim != 1:
            continue

        attrs = info.attrs or {}
        shape = attrs.get("source_shape") or info.shape

        if not shape:
            return None

        extents.add(str(shape[0]))

    if not extents:
        return None

    if len(extents) == 1:
        return extents.pop()

    if len(extents) <= 3 and all(re.fullmatch(r"\w+", e) for e in extents):
        return sorted(extents)[0]

    return None


def staging_extent_guard(context, staged_tensors) -> str:
    """Require matching runtime extents before using a shared flat domain."""
    extents = []

    for name, _ in staged_tensors:
        info = context.tensors.get(name)

        if info is None or info.ndim != 1:
            continue

        attrs = info.attrs or {}
        shape = attrs.get("source_shape") or info.shape

        if shape:
            extents.append(str(shape[0]))

    unique = tuple(dict.fromkeys(extents))

    return (
        " && ".join(f"({unique[0]}) == ({extent})" for extent in unique[1:]) or "true"
    )


def dense_staging_memory_guard(context, inputs, outputs, *, pointer_suffix="") -> str:
    """Require complete dense read and write domains to be disjoint.

    Callers prove the original access maps before staging. This independent
    runtime obligation prevents bulk loads from crossing an earlier store,
    including stores in another packet or row. Physical source domains are
    conservative read envelopes, so broadcast inputs retain their real sizes.
    """
    from ninetoothed.backends.emitters.expressions import normalize_dtype

    widths = {"float32": 4, "float16": 2, "int32": 4, "bool": 1}
    input_names = tuple(dict.fromkeys(inputs))
    output_names = tuple(dict.fromkeys(outputs))
    names = tuple(dict.fromkeys((*input_names, *output_names)))
    spans = {}
    terms = []
    maximum = "~((uintptr_t)0)"

    for name in names:
        info = context.tensors.get(name)

        if info is None:
            return "false"

        if info.ndim == 0:
            if name in output_names:
                return "false"

            continue

        attrs = info.attrs or {}
        shape = tuple(
            map(str, attrs.get("source_shape") or info.source_shape or info.shape)
        )
        width = widths.get(normalize_dtype(info.dtype))
        strides = tuple(
            map(str, info.source_strides or attrs.get("source_strides", ()))
        )

        if (
            len(shape) not in {1, 2}
            or attrs.get("source_ndim", len(shape)) != len(shape)
            or attrs.get("jagged_offsets_param")
            or attrs.get("jagged_dim") is not None
            or width is None
            or any(not re.fullmatch(r"[A-Za-z_]\w*|\d+", axis) for axis in shape)
            or (strides and len(strides) != len(shape))
        ):
            return "false"

        # Check each positive factor before multiplying in uintptr_t. The
        # signed index range is stricter than the available pointer range.
        size = f"{width}ULL"

        for axis in shape:
            terms.extend(
                (
                    f"(({axis}) > 0)",
                    f"(((uintptr_t)({axis})) <= 9223372036854775807ULL / ({size}))",
                )
            )
            size = f"({size} * ((uintptr_t)({axis})))"

        if strides:
            terms.append(f"(({strides[-1]}) == 1)")

            if len(shape) == 2:
                terms.append(f"(({strides[0]}) == ({shape[1]}))")

        pointer = f"((uintptr_t){name}{pointer_suffix})"
        terms.append(f"({pointer} <= {maximum} - ({size} - 1ULL))")
        spans[name] = (pointer, size)

    for index, name in enumerate(output_names):
        if name not in spans:
            return "false"

        left, left_size = spans[name]

        for other in (*input_names, *output_names[index + 1 :]):
            if other not in spans:
                continue

            right, right_size = spans[other]
            terms.append(
                f"(({left} >= {right} && {left} - {right} >= {right_size}) || "
                f"({right} >= {left} && {right} - {left} >= {left_size}))"
            )

    return " && ".join(terms) or "false"


__all__ = [
    "split_stride_guard",
    "flatten_tiled_accesses",
    "canonical_flat_view",
    "staging_extent",
    "staging_extent_guard",
    "dense_staging_memory_guard",
]
