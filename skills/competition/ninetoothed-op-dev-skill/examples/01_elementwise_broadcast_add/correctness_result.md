# Correctness — example 01

## Command (from NineToothed fork/clone root)

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
pytest skills/competition/ninetoothed-op-dev-skill/examples/01_elementwise_broadcast_add/tests/test_example_broadcast_add.py -q
# or:
python skills/competition/ninetoothed-op-dev-skill/examples/01_elementwise_broadcast_add/verify.py
```

## Result summary

| Item | Value |
|------|-------|
| Verdict | **PASS** |
| NineToothed commit | `ef4c52899f5f836e3d77001b56c3544849cacf2c` |
| Evidence | `logs/benchmark_phase2b_gate_final_02.log` (examples section); Phase 4.1 reconfirm `logs/phase41_release_acceptance_final_02.log` |
| Coverage | broadcast shapes vs `torch.add`; `out=` reuse; float16 correctness (atol/rtol=1e-2) |

Stdout contains `correctness: PASS` on verify; pytest exit code `0`.
