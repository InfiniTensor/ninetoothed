# Correctness — example 09

## Command (from NineToothed fork/clone root)

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
pytest skills/competition/ninetoothed-op-dev-skill/examples/09_performance_regression_fix/tests -q
# or:
python skills/competition/ninetoothed-op-dev-skill/examples/09_performance_regression_fix/verify.py
```

## Result summary

| Item | Value |
|------|-------|
| Verdict | **PASS** |
| NineToothed commit | `ef4c52899f5f836e3d77001b56c3544849cacf2c` |
| Evidence | `logs/benchmark_phase2b_gate_final_02.log`; Phase 4.1 `logs/phase41_release_acceptance_final_02.log` |
| Coverage | correctness for `BLOCK_SIZE` ∈ {32, 256, 1024} before any timing claim |

Stdout contains `correctness: PASS` on verify; pytest exit code `0`.
