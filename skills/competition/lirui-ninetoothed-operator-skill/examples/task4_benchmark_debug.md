# Task 4: Benchmark / Debug / AOT Build Self-Test Plan

## Task goal

Validate that the skill guides an AI agent through benchmark recording, generated source inspection, AOT build verification, and failing test diagnosis without fabricating results.

This task focuses on the build path rather than a new mathematical operator.

## Related repository files

- `docs/source/build.rst`: primary AOT build reference. It explains `premake`, `configs`, `meta_parameters`, `output_dir`, `lazy=True`, generated artifacts, caching, and warm-up shape options.
- `tests/test_aot.py`: examples of `ninetoothed.make(..., caller=..., kernel_name=..., output_dir=...)` and direct AOT validation.
- `tests/test_aot_auto_tuning.py`: `ninetoothed.build`, `premake`, `configs`, `meta_parameters`, and auto-tuning tests.
- `src/ninetoothed/aot.py`: AOT generation, dispatcher generation, C++ launch wrappers, `.so` compilation, and launch function loading.
- `src/ninetoothed/build.py`: multi-config AOT build, auto-tuning, CSV cache, generated dispatcher, cache fingerprint, and fallback behavior.
- `src/ninetoothed/generation.py`: generated source cache and source generation.
- `src/ninetoothed/auto_tuner.py`: timing mechanism using `triton.testing.do_bench`.

## Input / output / shape / dtype / layout

Suggested operator for this self-test: AOT vector add or add with scalar alpha, because it keeps correctness simple and makes build behavior easier to inspect.

- Inputs: `input`, `other`, optional scalar `alpha`.
- Output: same shape as input.
- Shape: include one large shape and one non-power-of-two shape, such as `(20260128,)` and `(1127,)`, following `tests/test_aot_auto_tuning.py` style.
- Dtype: `ninetoothed.float32` and optionally `ninetoothed.float16`.
- Layout: contiguous for the first AOT test; optional non-contiguous layout only after correctness is stable.

## Expected arrangement / application / Tensor pattern

1. Read `docs/source/build.rst` before writing AOT code.
2. Define `arrangement` and `application` exactly as a normal `ninetoothed.make` kernel would.
3. Wrap setup in `premake(size=None, dtype=None, block_size=None)`.
4. Define `configs` as `(args, kwargs, compilation_configs)` tuples.
5. Put `block_size` in `meta_parameters` when comparing block sizes.
6. Ensure `output_dir` exists before calling `ninetoothed.build`.
7. Use a unique `kernel_name` to avoid cache confusion during testing.
8. Inspect generated `.cpp`, `.h`, `.so`, `.csv`, and `.fingerprint` files only in the intended output directory.
9. Do not commit generated artifacts unless explicitly required.

## PyTorch reference plan

Reference:

```python
expected = torch.add(input, other, alpha=alpha)
```

Required cases:

- AOT build with one dtype and one block size.
- AOT build with multiple block sizes and `meta_parameters=("block_size",)`.
- Cached second run, if environment allows.
- Failure diagnosis case, such as missing `output_dir` or an intentionally unsupported dtype, recorded honestly.

## Pytest command

Focused commands after real implementation exists:

```bash
pytest tests/test_aot.py -q
pytest tests/test_aot_auto_tuning.py -q
```

If a separate self-test file is created, record that exact command instead.

## Correctness test result

```text
待真实运行后填写
```

## Benchmark plan

At least two benchmark designs must be recorded:

1. Runtime baseline benchmark:
   - Baseline: PyTorch `torch.add`.
   - Candidate: NineToothed AOT vector add.
   - Shapes: `(1127,)` and `(20260128,)`.
   - Dtypes: `float32`, optional `float16`.
   - Layout: contiguous.

2. AOT configuration benchmark:
   - Baseline: one fixed block size.
   - Candidate: multiple `block_size` values with `meta_parameters=("block_size",)`.
   - Record generated CSV auto-tuning selection when available.

Use the following table for every real run. Record timing units and the exact repetition method in the command or evidence log.

| Case | Baseline | Shape | Dtype | Layout | Command | Mean time | Variance | Conclusion |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T4-B1 PyTorch vs AOT vector add | 待真实运行后填写 | `(1127,)` | `float32` | contiguous | 待真实运行后填写 | 待真实运行后填写 | 待真实运行后填写 | 待真实运行后填写 |
| T4-B2 large vector runtime | 待真实运行后填写 | `(20260128,)` | `float32` | contiguous | 待真实运行后填写 | 待真实运行后填写 | 待真实运行后填写 | 待真实运行后填写 |
| T4-B3 fixed block vs auto-tuned | 待真实运行后填写 | 待真实运行后填写 | 待真实运行后填写 | contiguous | 待真实运行后填写 | 待真实运行后填写 | 待真实运行后填写 | 待真实运行后填写 |

## Benchmark command

```text
待真实运行后填写
```

## Benchmark result

```text
待真实运行后填写
```

## Generated source inspection plan

Record:

- `output_dir` absolute or repo-relative path.
- Whether the directory existed before build.
- Generated `.cpp` files.
- Generated `.h` files.
- Generated `.so` file.
- Generated `.csv` auto-tuning file.
- Generated `.fingerprint` file.
- Whether the second run reused cache.

Current generated source result:

```text
待真实运行后填写
```

## Failure diagnosis plan

Record every failure with:

- Symptom.
- Error message.
- Suspected root cause.
- Minimal fix.
- Re-run command.
- Re-run result.

Likely root causes:

- `output_dir` missing.
- `kernel_name` collision with stale cache.
- Missing CUDA, NVCC, Triton, or GPU runtime.
- Auto-tuning CSV missing or stale.
- Shape or dtype mismatch between `premake`, runtime arguments, and configs.

Current diagnosis:

```text
待真实运行后填写
```

