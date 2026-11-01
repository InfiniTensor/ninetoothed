"""Launch-before-work guards for the versioned named layout contract."""

from ninetoothed.compiler.layout_runtime import (
    _tensor_has_non_overlapping_strides,
    memory_spans_overlap,
    tensor_memory_span,
)


def validate_named_call(values, abi):
    import torch

    from ninetoothed.compiler.runtime import _binding_value
    from ninetoothed.ir import IndexExpr
    from ninetoothed.ir.module import evaluate_index_expression

    contract = abi.named_layout

    if contract.get("version") != 1 or abi.outputs != (contract["output"],):
        raise ValueError("Invalid named launch contract.")

    groups = {}
    symbols = {
        binding.name: _binding_value(binding, values)
        for binding in abi.kernel_args
        if binding.kind == "meta"
    }

    for name, binding in contract["parameters"].items():
        tensor = values[name]

        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Named S1 requires Torch tensor arguments.")

        if len(tensor.shape) != len(binding["axes"]):
            raise ValueError(f"Named parameter {name!r} has the wrong logical rank.")

        if any(int(size) <= 0 for size in tensor.shape) or any(
            int(stride) <= 0 for stride in tensor.stride()
        ):
            raise ValueError(
                f"Named tensor {name!r} requires positive extents and strides."
            )

        for axis, dim in zip(binding["axes"], range(len(binding["axes"]))):
            groups.setdefault(axis, []).append((name, dim, int(tensor.shape[dim])))

        for size in binding["tile"]:
            try:
                actual = evaluate_index_expression(IndexExpr.parse(size), symbols)
            except (KeyError, ValueError) as exc:
                raise ValueError("Cannot bind a named tile parameter.") from exc

            if type(actual) is not int or actual <= 0:
                raise ValueError("Named tile sizes must remain positive integers.")

    for axis, observations in groups.items():
        if any(size != observations[0][2] for _, _, size in observations):
            rendered = ", ".join(
                f"{name}[{dim}]={size}" for name, dim, size in observations
            )
            raise ValueError(
                f"Named axis {axis!r} has conflicting extents: {rendered}."
            )

    output = values[contract["output"]]

    if not _tensor_has_non_overlapping_strides(output):
        raise ValueError("Named output has overlapping write coordinates.")

    span = tensor_memory_span(output)

    if span is None:
        raise ValueError("Cannot prove the named output memory span.")

    for name in contract["parameters"]:
        if name == contract["output"]:
            continue

        other = tensor_memory_span(values[name])

        if other is None or memory_spans_overlap(span, other):
            raise ValueError(f"Named output may overlap input {name!r}.")
