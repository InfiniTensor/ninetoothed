# Run log — example 01

## D1 Task card

See `task_card.md`.

## D2–D3 Route + rg

Family: **elementwise / broadcast**.

```bash
cd "$REPO"   # NineToothed root (has src/ninetoothed/)
rg -n "arrangement|application|ninetoothed.make" tests/test_add.py
rg -n "expand|broadcast" tests/test_expand.py tests/test_add.py
```

Nearest patterns: `tests/test_add.py` (`@ninetoothed.jit` / make style), expand helpers.

## D4 Minimal patch

- `solution/broadcast_add.py` — `arrangement` tiles + expand, `application` add
- `tests/test_example_broadcast_add.py` — vs `torch.add`

## D6 Correctness

```bash
python skills/competition/ninetoothed-op-dev-skill/examples/01_elementwise_broadcast_add/verify.py
# or:
pytest skills/competition/ninetoothed-op-dev-skill/examples/01_elementwise_broadcast_add/tests -v --tb=short
```

## D8 Benchmark

Optional; see `benchmark_result.md` (verify.py prints ms/iter when CUDA available).
