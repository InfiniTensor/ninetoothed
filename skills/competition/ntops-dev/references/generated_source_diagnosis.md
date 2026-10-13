# Generated Source and Performance Diagnosis

Use this reference after correctness passes and benchmark shows a performance gap. The goal is not to claim a complete compiler-level proof, but to collect useful evidence for the next optimization step.

Source: diagnosis fields come from the local `inspect_generated_source.py`, generated files under the NineToothed cache, and recorded RTX 4090 D runs. Bottleneck statements are hypotheses until a follow-up benchmark confirms them.

## When To Inspect Generated Source

Inspect generated source when:

- a benchmark is much slower than PyTorch/Triton baseline,
- a performance-sensitive operator uses reduction, matmul, pooling, or attention,
- a failure may be related to dtype/precision, masks, or non-contiguous access,
- the task asks for performance diagnosis rather than a new operator.

## Evidence To Collect

Record:

- operator name,
- benchmark command and result,
- generated source path,
- file size and line count,
- counts of `tl.*` or `triton.language.*` operations such as load, store, dot, exp, erf, where, arange, sum, and max,
- whether `@triton.heuristics` or `@triton.autotune` appears,
- relevant `BLOCK_SIZE` / tile / constexpr names,
- obvious dtype casts such as `tl.float32`, `tl.float16`, or TF32 precision variants.

## Reading The Source

Start with these questions:

- Are there more loads/stores than the operator semantics suggest?
- Are masks generated inside hot loops?
- Is a reduction recomputing values that could be hoisted?
- Does the code cast to float32 for numerical stability, and is that expected?
- Does matmul use `tl.dot`, and does the precision setting match the reference?
- Is the input contiguous assumption explicit, or does the code rely on strides?
- Are tile sizes too small/large for the tested shape?

## Interpreting Existing Historical Results

The recorded historical benchmarks show:

- GELU is correct but slower than PyTorch eager for tested shapes.
- Softmax is correct but slower than PyTorch eager for `(1024, 1024)` float16.
- AddMM is correct but slower than PyTorch eager for `(512, 512, 512)` float16.
- Maximum is correct for the recorded same-shape and zero-dimensional cases but slower than PyTorch eager for `(1048576,)` float16.

These results are not final optimization claims. They identify where final-round work should inspect generated source, tile choices, load/store patterns, and dtype/precision behavior.

## Diagnosis Template

```text
Operator:
Correctness result:
Benchmark result:
Generated source path:
Source summary:
  lines:
  language_load:
  language_store:
  language_dot:
  language_exp:
  language_arange:
  notable casts:
  autotune/heuristics:
Likely bottleneck:
Next experiment:
Risk:
```
