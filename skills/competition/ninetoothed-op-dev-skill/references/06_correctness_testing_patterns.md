# 06 — Correctness testing patterns

## When to use

Every operator task — before benchmark or optimization.

## Standard harness (ninetoothed repo)

- **Runner:** `pytest`  
- **Devices:** `tests/utils.py` → `get_available_devices()` (CUDA if available)  
- **Seeds:** `tests/conftest.py` — deterministic per module/test  
- **Assertion:** `torch.allclose(output, expected)`  

## Minimal workflow

1. Implement kernel wrapper (like `test_add.py` top-level `add()`).  
2. Parametrize `device`, `dtype`, shapes.  
3. Compare to `torch` reference on same device/dtype/view.  
4. Log command + exit code to `logs/correctness/`.

```bash
cd <repo-root>
pytest tests/test_add.py -v --tb=short 2>&1 | tee logs/correctness/test_add.log
```

## Optional examples (`--examples-root`)

```bash
cd <examples-root>
pytest tests/test_ops.py -v --tb=short
```

Uses `bench.assert_match` pattern in `<examples-root>/bench.py` for multi-impl comparison.

## fp16 / bf16

- Use explicit tolerances if needed: `torch.allclose(a, b, rtol=1e-2, atol=1e-2)`  
- See `test_aot.py` for `bfloat16` / `ninetoothed.bfloat16` pairing

## Compliance

- Do **not** delete or skip failing tests to pass  
- Do **not** claim pass without saved pytest output  
- If CUDA unavailable, state unsupported — do not fake GPU results

## Report template

```markdown
## Correctness
- Command: `pytest ...`
- Exit code: 0
- Log: logs/correctness/xxx.log
- Cases: dtype, shape, layout (list)
```
