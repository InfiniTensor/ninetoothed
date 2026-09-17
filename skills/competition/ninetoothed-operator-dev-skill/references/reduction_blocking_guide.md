# Reduction And Blocking Guide

## Requirement Checks

- Define the reduction axes, negative-axis normalization, and `keepdim` rule.
- Record input/output shape, empty-domain behavior, and supported ranks.
- Define accumulator dtype, output dtype, tolerance, and stability requirements.
- Identify whether the reduction domain is static, dynamic, or multi-axis.

## Stable Algorithms

For softmax-like work, prefer a stable max-subtraction algorithm. For norms or
statistics, inspect whether accumulation should use a wider dtype. Preserve the
repository's existing reduction primitive and synchronization model.

Do not replace a correct local reduction pattern with a custom tree unless the
task demonstrates a real limitation.

## Block Design

- Map the independent output domain separately from the reduced domain.
- Choose a block or tile that covers tails with an explicit mask.
- Check intermediate storage and synchronization requirements.
- Avoid rereading inputs when an intermediate can be reused safely.
- Keep dynamic-shape fallback behavior explicit.

## Correctness Matrix

- Small and non-aligned reduction lengths.
- Singleton reduction domain.
- Large-magnitude values for numerical stability.
- Negative and positive axis forms.
- Contiguous input and the requested supported layout.
- Every supported dtype that changes accumulation.

Compare shape, dtype, values, and normalized-axis behavior with PyTorch.

## Benchmark Interpretation

Record both independent and reduction dimensions. A result for `64x1024`
does not establish behavior for tiny rows, very long rows, other axes, or other
dtypes. Keep compile cost separate and report selected-shape limits.

## Common Failures

- Reducing the wrong logical axis after a layout transform.
- Missing a tail mask.
- Overflow from unstable exponentiation.
- Comparing against a reference with different dtype accumulation.
- Timing before correctness or including compilation in one side only.
