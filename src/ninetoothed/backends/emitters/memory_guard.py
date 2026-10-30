"""Physical nonoverlap proof for materialized native task storage."""

import re


def dense_staging_memory_guard(
    context, inputs, outputs, *, pointer_suffix="", dense_inputs=True
) -> str:
    """Require complete dense read and write domains to be disjoint.

    Callers prove the original access maps before staging. This independent
    runtime obligation prevents bulk loads from crossing an earlier store,
    including stores in another packet or row. Physical source domains are
    conservative read envelopes, so broadcast inputs retain their real sizes.
    """
    from ninetoothed.backends.emitters.expressions import normalize_dtype

    widths = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "int32": 4,
        "int64": 8,
        "uint64": 8,
        "int8": 1,
        "uint8": 1,
        "bool": 1,
    }
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
            not shape
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

        if strides and (dense_inputs or name in output_names):
            for axis, stride in enumerate(strides):
                trailing = " * ".join(f"({n})" for n in shape[axis + 1 :]) or "1"
                terms.append(f"(({shape[axis]}) == 1 || ({stride}) == ({trailing}))")
        elif strides:
            # Indexed tile loads retain the original access map. They need a
            # conservative physical envelope, rather than dense input rows.
            size = f"{width}ULL"

            for axis, stride in zip(shape, strides):
                scaled = f"((uintptr_t)({stride}) * {width}ULL)"
                terms.extend(
                    (
                        f"(({stride}) >= 0)",
                        f"((uintptr_t)({stride}) <= 9223372036854775807ULL / {width}ULL)",
                        f"(({stride}) == 0 || (uintptr_t)(({axis}) - 1) <= 9223372036854775807ULL / {scaled})",
                    )
                )
                extra = f"((uintptr_t)(({axis}) - 1) * {scaled})"
                terms.append(f"({extra} <= 9223372036854775807ULL - ({size}))")
                size = f"({size} + {extra})"

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
