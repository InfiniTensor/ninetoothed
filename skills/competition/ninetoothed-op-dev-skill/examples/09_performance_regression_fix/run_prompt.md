Use skill `ninetoothed-op-dev-skill` (SKILL.md decision tree D1–D9, especially **D8**).

## Task

Document a block-size performance regression: same add kernel at `BLOCK_SIZE=32` vs `1024`.

## Required process

1. Write a task card; `rg` nearest tunable / generation tests before coding.
2. Correctness PASS at **every** `BLOCK_SIZE` you will time.
3. Then benchmark: fixed inputs, warmup ≥5, repeated measurement, fair baseline, bounded conclusion.
4. If ratios look absurd, follow `failure_diagnosis.md` (autotuning / unfair baseline).
5. Do not claim peak performance under autotuning without disclosure.
