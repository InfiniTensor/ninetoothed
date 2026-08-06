# Failure Diagnosis

Use this reference for failing tests, generated source issues, AOT build problems, import failures, and benchmark regressions.

## Diagnosis Loop

Always record:

1. exact command
2. first relevant error line
3. suspected failing layer
4. smallest code or config change
5. rerun command
6. result

## Common Layers

- Task parsing: wrong shape, dtype, axis, broadcast, or layout contract.
- Arrangement: wrong rank, tile shape, flatten/ravel order, missing identity fill.
- Application: wrong operation, unstable formula, dtype mismatch, missing reduction axis.
- Wrapper: output allocation wrong shape/dtype/device, missing stride/offset parameter.
- Test: wrong PyTorch reference, tolerance too strict or too loose, skipped CUDA accidentally.
- Build/AOT: missing dependency, stale generated source, wrong compile flag, cache issue.
- Benchmark: measuring async CUDA without synchronization, correctness disabled without note, mismatched input sizes.

## Generated Source Review

If the repository exposes generated code or debug helpers:

- Generate code using the local documented path.
- Search for duplicated loads, unexpected casts, unnecessary stores, wrong masks, and wrong strides.
- Treat generated code as evidence, not as the primary source of truth.

Useful searches:

```shell
rg -n "generate|generated|source|debug|aot|build|cache" .
rg -n "load|store|stride|mask|BLOCK_SIZE|constexpr|meta" <generated-file-or-dir>
```

## AOT Build Checks

For AOT tasks:

- Locate existing AOT tests before editing. In the core repository, AOT is triggered via `ninetoothed.make(..., caller="cuda", output_dir=...)` and compiles with `nvcc`; check `nvcc --version` first. A working PyTorch CUDA runtime does not imply `nvcc` exists.
- Generated sources are cached under `ninetoothed.generation.CACHE_DIR`; inspect them there.
- Keep build commands reproducible from a clean checkout.
- Record compiler, Python, CUDA, PyTorch, Triton, and NineToothed versions when possible.
- If build is blocked by missing CUDA/toolchain, provide the exact missing command or package.

## Minimal Repair Examples

- Wrong softmax on large values: subtract row max before `exp`.
- Wrong tail tile: use `Tensor(..., other=identity)` and correct mask/fill behavior.
- Non-contiguous failure: pass stride metadata or add a documented contiguous-only guard and test.
- Benchmark flaky: synchronize CUDA, warm up, reuse benchmark harness, and avoid changing correctness code.

## Final Failure Note

Use this format:

```text
Failure:
Command:
Evidence:
Root cause:
Fix:
Verification command:
Verification result:
Remaining limitation:
```
