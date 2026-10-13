# Feishu NineToothed Fix Notes

This note distills reusable guidance from the NineToothed troubleshooting documents distributed by the competition through Feishu. It is not a replacement for local `ntops` tests; use it as a checklist before changing operator code or declaring a diagnosis complete.

Source boundary: the reviewed material is competition-provided NineToothed documentation. Checklist wording is a condensed engineering summary prepared for this skill, not an official NineToothed API specification. The skill does not reproduce the original documents, and every recommendation still requires validation against the local repository and CUDA tests.

## Sources Reviewed

- `Symbol 问题`
- `自动注册｜图中断`
- `广播问题`
- `Mask & offsets 使用`
- `Store`
- `九齿 torch compile 解决方案`
- `expand 防呆修改`
- `Eval 问题`
- `Embedding kernel 问题`

This skill does not require Feishu access at runtime and must not depend on reopening external documents during evaluation.

## Reusable Lessons

### Symbol and constexpr inputs

Shape-derived constants such as tile sizes, hidden dimensions, or block sizes should be represented as `Symbol(..., constexpr=True)` or explicit make-time/call-time parameters when the generated kernel needs them. Do not assume a constant can be safely hard-coded inside `arrangement` just because it is known from the input shape.

When a shape-dependent value is needed in `arrangement`, pass it through the generated kernel call, for example `kernel(a, b, out, H=a.shape[-1])`. This keeps the arrangement expression symbolic while still giving the compiler a concrete constexpr.

Checklist:

- Identify every block size, reduction width, hidden dimension, and tile extent.
- Decide whether it is a Python constant, a `Symbol` constexpr, or a runtime tensor-derived value.
- If it is shape-derived, pass it through the make/kernel path rather than silently closing over one observed shape.
- Add a second shape in correctness tests to catch accidental single-shape specialization.

### Broadcast arrangement

NineToothed arrangement expressions can reject or mis-handle forms that look equivalent under PyTorch broadcasting. In the broadcast note, a `(B, 1)` temperature tensor is intended to broadcast over `(B, H)` logits. Some `expand` plus `tile` forms fail at `make`, while a simpler `tile` form can compile but produce an incorrect result.

Checklist:

- Treat PyTorch broadcasting semantics and NineToothed arrangement semantics as related but not identical.
- For each broadcast input, write down source shape, arranged shape, and output shape.
- Use a minimal `eval(subs)` or tiny CUDA check to inspect the arranged tensors before trusting a compiled kernel.
- Include at least one non-square or non-power-of-two shape so accidental symmetry does not hide a wrong broadcast.

### Expand guardrails

The expand note highlights a defensive check in offset construction: target dimensions that are not `-1` should not all be treated as broadcasted singleton dimensions. If the original size equals the requested new size, preserve the index instead of forcing that dimension's offset to zero.

Checklist:

- Distinguish "keep this dimension" from "broadcast this singleton dimension".
- Test `expand` with unchanged dimensions, singleton-to-larger dimensions, and mixed `-1` dimensions.
- For broadcast tasks, inspect both arranged shape and offset behavior, not shape alone.

### Eval as a diagnostic, not an oracle

The eval note shows that `.eval(subs)` can fail for some tensors declared with `shape_options`. Keep `eval` in the workflow because it is valuable for arrangement inspection, but record its limits.

Checklist:

- If `eval(subs)` fails, capture the tensor shape/options and the exact expression being evaluated.
- Confirm whether the same case also fails at `make` or CUDA execution before treating it as an operator failure.
- Use a tiny CUDA reference case when `eval` cannot represent the shape-options combination.

### Mask and offsets

Using `.data_ptr()` and `.offsets()` moves the implementation closer to handwritten Triton. That can be useful for low-level indexing, but it also moves responsibility for mask handling, multi-dimensional bounds, and address arithmetic into the operator code.

Checklist:

- For every `.offsets(axis)` use, state the logical axis and the physical address expression it feeds.
- Add masks for tail blocks and non-divisible dimensions before load/store, not after a value has already been used.
- Verify 1D examples separately from 2D or batched examples; a shape check that is enough for 1D can be incomplete for higher rank tensors.
- When a task is easier with built-in arrangement/load/store behavior, prefer that over raw pointer arithmetic.

### Store and scatter-like writes

Scatter-like stores such as `a[b[i], j] = c[i, j]` need explicit reasoning about index tensors, row strides, and output mutation. The logical output shape is not enough to prove that the write-back address is correct.

Checklist:

- Separate read layout, index layout, and write layout in the operator contract.
- Check whether the operation mutates an input, writes a fresh output, or returns a view-like result.
- Test repeated indices, boundary indices, and non-contiguous source or destination tensors if the contract allows them.
- Compare both final tensor values and untouched regions of the destination.

### Embedding and advanced indexing

Embedding is a lookup/scatter-adjacent pattern: input indices `(B, S)` select rows from `weight (num_emb, H)` to produce `output (B, S, H)`. A practical NineToothed decomposition can flatten `B*S` into a single logical dimension `M`, then tile/block that dimension while carrying the embedding dimension.

Checklist:

- State the index tensor dtype, valid index range, and out-of-range behavior.
- Decide whether to flatten batch/sequence dimensions, and record how to map the flattened offset back to `(B, S)`.
- Test boundary indices `0` and `num_emb - 1`; include repeated indices and non-contiguous output if supported.
- Compare against `torch.nn.functional.embedding` or direct `weight[input]` reference.

### torch.compile and custom op registration

The reviewed compile notes show two separate concerns:

- Wrapping a NineToothed operator as a `torch.op` can help avoid Dynamo inspecting the Python implementation directly.
- If the NineToothed kernel is still compiled with dynamic parameters during a traced region, graph breaks can still happen.

Checklist:

- Stabilize the custom op schema before testing `torch.compile`.
- Pre-build or cache the kernel outside the compiled hot path when possible.
- Keep constexpr/block-size choices explicit; distinguish static mode from dynamic `ninetoothed.block_size()` mode.
- Record whether the validation checks eager correctness only, `torch.compile` graph capture, or both.

## Skill Usage Rule

When an operator task touches shape-derived tiling, broadcasting, raw offsets, scatter/store, or `torch.compile`, first add a short "NineToothed risk note" to the task log. The note should state which issue pattern applies, the minimal reproduction or validation command, and what remains unverified.
