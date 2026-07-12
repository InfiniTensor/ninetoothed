# Validating the skill itself

This directory documents how to verify that the skill produces correct, testable
NineToothed operators — i.e. that the skill *works*, not just that it reads well.

## What "the skill works" means

For each of the four lanes, an agent following `SKILL.md` should produce an
operator that (a) compiles, (b) matches a PyTorch reference, and (c) passes a
repo-style pytest. The `examples/` directory is the evidence: each self-test is an
operator authored by following the skill's workflow, plus its passing test.

## How to reproduce

From the repository root, with the dev environment installed
(`pip install -e .[all]` and `pip install -r requirements.txt`):

```bash
# Run all four self-test operators.
pytest skills/competition/ninetoothed-operator-skill/examples -v -p no:cacheprovider

# Run the performance benchmark (T4).
python skills/competition/ninetoothed-operator-skill/examples/t4-performance-diagnosis/benchmark_multiply.py

# Run the full repo CI check sequence the skill prescribes.
bash skills/competition/ninetoothed-operator-skill/scripts/run_ci_checks.sh
```

## Expected result

`4 passed` on a CUDA device. On a machine without a GPU,
`tests.utils.get_available_devices()` returns empty and the tests SKIP — that is
the honest, expected off-GPU behavior, not a failure to hide.

## Coverage matrix

| Lane | Self-test | Verifies |
|---|---|---|
| L1 elementwise/broadcast | T1 multiply | basic arrange-and-apply, `*` primitive |
| L2 reduction/block | T2 softmax | `constexpr` block, `ntl.max/exp/sum`, `-inf` fill |
| L3 layout-sensitive | T3 2-D multiply | non-contiguous (transposed) input correctness |
| L4 performance/diagnosis | T4 benchmark + diagnosis | `do_bench` reporting, real fix loop |

## With-skill vs no-skill

Without the skill, a general agent typically writes NineToothed as if it were
PyTorch/raw Triton, assumes contiguous inputs, and skips tests — the layout case
(T3 non-contiguous) is the most common silent failure. With the skill, the fixed
workflow forces reading a reference operator, testing a non-contiguous input, and
running the CI sequence, which is what makes T3 pass. The real diagnosis loop in
`examples/t4-performance-diagnosis/README.md` (Part B) shows the self-correction
behavior end to end.
