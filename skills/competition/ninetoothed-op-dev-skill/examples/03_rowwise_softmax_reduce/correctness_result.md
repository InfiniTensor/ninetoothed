# Correctness — example 03

## Command (from NineToothed fork/clone root)

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
pytest skills/competition/ninetoothed-op-dev-skill/examples/03_rowwise_softmax_reduce/tests -q
# or:
python skills/competition/ninetoothed-op-dev-skill/examples/03_rowwise_softmax_reduce/verify.py
```

## Result summary

| Item | Value |
|------|-------|
| Verdict | **PASS** |
| NineToothed commit | `ef4c52899f5f836e3d77001b56c3544849cacf2c` |
| Evidence | `logs/benchmark_phase2b_gate_final_02.log`; Phase 4.1 `logs/phase41_release_acceptance_final_02.log` |
| Coverage | stable row-wise softmax / max-sum reduce |

Stdout contains `correctness: PASS` on verify; pytest exit code `0`.
