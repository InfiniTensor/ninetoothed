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
    "staging_extent",
]
