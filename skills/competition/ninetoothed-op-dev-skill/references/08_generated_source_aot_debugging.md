# 08 — Generated source / AOT debugging

## When to use

Tasks mentioning generated source, Triton output, AOT build, compile artifacts, performance regression at codegen level.

## Generated source (JIT)

**Reference:** `tests/test_generation.py`

```python
kernel = ninetoothed.make(...)
source_file = kernel._source
with open(source_file) as f:
    contents = f.read()
```

Look for: redundant loads/stores, unexpected broadcasts, wrong block sizes.

Also: `import ninetoothed.generation` — `generation.CACHE_DIR` used in `test_aot.py`.

## AOT build

**Docs:** `<repo-root>/docs/source/build.rst`  
**Tests:** `<repo-root>/tests/test_aot.py`, `<repo-root>/tests/test_aot_auto_tuning.py`

Pattern:

- `ninetoothed.make(..., kernel_name=..., output_dir=...)`
- `ninetoothed.aot` module for compile/load steps (read test file for current API)

## Toolchain requirements

- **nvcc / CUDA toolkit** may be required for native AOT artifacts  
- If `nvcc` missing (`env_check.md`), report:

  > AOT compile not run — nvcc not available; documented config and expected steps only.

Do not fake `.so` files or compile logs.

## Debugging module

`tests/test_debugging.py`, `src/ninetoothed/debugging.py` — visualization / debug hooks.

## Report template

```markdown
## Generated source / AOT
- Kernel: ...
- Source path or build dir: ...
- Observation: (e.g. extra load in loop)
- Action taken: ...
- Re-verify command: ...
```
