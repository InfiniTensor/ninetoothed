# Layout, Stride, And Offset Guide

## Model The Tensor

Treat shape, stride, storage offset, and contiguity as separate facts. A tensor
can have the expected logical shape while indexing a sliced, transposed, or
offset storage region.

Record:

- Logical dimensions and output-shape formula.
- Element strides for every input and output.
- Storage offset and aliasing behavior.
- Padding, dilation, kernel size, and step size.
- Supported contiguous and non-contiguous classes.

## Implementation Choices

Use the repository's tensor metadata and indexing helpers. Derive addresses
from logical indices and recorded strides when the contract supports arbitrary
strides. If only contiguous input is supported, enforce that boundary at the
wrapper and test the error or conversion behavior.

For pooling and convolution-like operators, calculate effective kernel extent
before output shape. Check each boundary location and mask out-of-range loads.

## Correctness Matrix

- Contiguous baseline.
- Transpose with the same logical values when supported.
- Slice with a nonzero offset.
- Step slice with a larger stride.
- Odd spatial extents and non-aligned tails.
- Padding, dilation, and stride combinations required by the task.
- A clearly unsupported case that proves the boundary is enforced.

Compare output shape before comparing values. Report skip and expected-failure
counts separately from passed cases.

## Claim Boundary

One transpose or slice does not prove full non-contiguous support. State the
exact layout classes exercised. Keep diagnostic convolution failures separate
from pooling correctness and from performance conclusions.

## Common Failures

- Ignoring storage offset.
- Reusing contiguous flattening for a strided tensor.
- Mixing byte strides and element strides.
- Computing the wrong output extent with dilation.
- Treating skipped layout cases as passes.
