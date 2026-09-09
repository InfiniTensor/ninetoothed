# VERIFICATION.md

## Verification Goal

This file explains how to verify that `ninetoothed-operator-skill` is effective. The focus is not whether a single operator is fastest, but whether an AI agent can complete a NineToothed operator task correctly after reading `SKILL.md`.

## Environment

**Windows 11 + NVIDIA RTX 5060 Laptop GPU (8GB) + PyTorch 2.12.0.dev20260408+cu128 + ninetoothed 0.26.0**

All tests and benchmarks have been executed on this GPU environment.

| Environment | pytest | benchmark |
|-------------|--------|-----------|
| RTX 5060 + CUDA 12.8 | All PASSED | Results collected |
| No CUDA (fallback) | SKIP (expected) | SKIP and exit |

## Verification Steps

1. Install NineToothed:

```bash
pip install -e .
```

2. Ask the AI agent to read:

```text
skills/competition/ninetoothed-operator-skill/SKILL.md
```

3. Give the AI a task, for example:

```text
Implement a 2D elementwise add operator using NineToothed, with PyTorch reference tests, and CUDA skip when unavailable.
```

4. Check AI output against:

- Did it extract input/output shape, dtype, device first?
- Did it choose the right NineToothed implementation style?
- Does it include PyTorch reference comparison?
- Does it include CUDA skip?
- Does it mention benchmark or performance verification?
- Does it mention generated source / AOT build (for perf tasks)?
- Does it avoid hardcoding hidden evaluation answers?
- Does it record failure diagnosis or known limitations?

## Self-test Verification

```bash
cd skills/competition/ninetoothed-operator-skill
bash scripts/run_self_tests.sh
python scripts/collect_logs.py
```

All tests PASSED on RTX 5060 + CUDA 12.8. Log summary is generated at `reports/logs/summary.md`.

## Pass Criteria

The skill is considered effective if, after reading `SKILL.md`, an AI agent can stably generate task files with implementation, tests, benchmark or diagnosis, and the test logic matches PyTorch reference and CUDA skip conventions.

## Limitations

This verification does not guarantee all hardware, all dtypes, or dynamic shapes pass. Full validation was done on RTX 5060 + CUDA 12.8.
