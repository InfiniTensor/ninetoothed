# Run log — example 05

## D1 Task card

See `task_card.md` (layout branch mandatory).

## D2–D3 Route + rg

Family: **layout / stride**.

```bash
cd "$REPO"
rg -n "is_contiguous|stride|as_strided" tests/test_clone.py tests/test_data_ptr.py
rg -n "ninetoothed.make|arrangement" tests/test_add.py
```

Nearest patterns: clone/data_ptr layout tests; elementwise `make` from add tests.

## D4–D5 Minimal patch + layout branch

- `solution/strided_add_kernel.py` — tile + add; no `.contiguous()` in the hot path
- `tests/test_example_strided_add.py` — transpose / empty_strided views + asserts

## D6 Correctness

```bash
python .../examples/05_non_contiguous_transpose_add/verify.py
pytest .../examples/05_non_contiguous_transpose_add/tests -v --tb=short
```

## D8 Benchmark

**N/A** — this example gates layout contract, not performance.
