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


def flatten_tiled_accesses(body: str, context: ModuleRenderContext) -> str | None:
    """Rewrite the tiled predicated body into the flat ``name[index]`` form.

    Contiguous 1-D tiled renders access tensors as
    ``name[(nt_outer_index) * BLOCK + (nt_inner_index)]`` guarded by
    ``nt_pred`` bounds checks.  Inside one staged chunk those predicates are
    subsumed by ``nt_j < nt_cnt`` when (a) every predicate only references
    the tiling coordinates and size parameters and (b) every masked-load
    fallback value is zero.  Anything else (runtime flags, nonzero ``other``,
    strided access templates) bails out to the generic scalar path.
    """
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
        or "? (" in residue
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


__all__ = [
    "split_stride_guard",
    "flatten_tiled_accesses",
    "staging_extent",
]
