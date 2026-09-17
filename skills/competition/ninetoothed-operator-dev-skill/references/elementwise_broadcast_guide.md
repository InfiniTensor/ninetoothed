# Elementwise And Broadcast Guide

## Requirement Checks

- Define scalar semantics and the PyTorch reference.
- Record input ranks, output shape, dtype promotion, and accumulation dtype.
- Separate same-shape, scalar, singleton-dimension, and general broadcast cases.
- Identify mask and tail behavior for non-multiple extents.
- Decide whether in-place behavior, aliasing, or mixed dtypes are supported.

## Pattern Selection

Read a nearby unary operator for one-input load/store behavior and a nearby
binary operator for broadcast indexing. Reuse existing arrangement,
application, dtype, and tensor meta-operation helpers.

Avoid materializing broadcasted tensors when index mapping can express the
same semantics. Keep output allocation and public wrapper behavior consistent
with adjacent operators.

## Correctness Matrix

Cover the smallest decisive set:

- Equal shapes.
- Scalar or rank-zero input when supported.
- Leading-rank broadcast.
- Singleton dimension in each operand position.
- Tail length that is not aligned to the selected block.
- Positive, negative, zero, NaN, or infinity values when semantics require.
- Every supported dtype that changes promotion or tolerance.

Compare output shape, dtype, values, and documented failure behavior.

## Performance Checks

- Count avoidable loads and stores.
- Check whether broadcast index computation is repeated unnecessarily.
- Inspect generated source for redundant conversions or materialization.
- Separate compile cost from steady-state timing.
- Benchmark only after the exact timed input passes correctness.

## Common Failures

- Using physical shape instead of broadcasted logical shape.
- Applying a mask to the wrong tensor domain.
- Letting Python or PyTorch promotion differ from the kernel path.
- Testing only equal shapes while claiming broadcast support.
- Reporting a one-shape timing ratio as general speedup.
