# Validation Plan

## Purpose

This plan validates whether the skill improves AI-agent behavior on NineToothed operator tasks without relying on hidden answers or fabricated results.

## Correctness test validation

1. Select an operator task with a clear PyTorch reference.
2. Ask the agent to extract requirements using `skills/competition/lirui-ninetoothed-operator-skill/SKILL.md`.
3. Require tests for normal input, broadcast input when applicable, boundary input, and non-contiguous input when applicable.
4. Run focused pytest.
5. Record exact command and output.

Local T1 focused pytest command attempted:

```powershell
$env:PYTHONPATH = "$PWD\src"
pytest tests/test_add.py tests/test_addmm.py tests/test_pow.py -q
```

Local result:

```text
pytest collection failed because the local Windows Python environment could import ninetoothed from src, but ninetoothed imports triton and triton is not installed.
ModuleNotFoundError: No module named 'triton'
```

Interpretation: this is a local environment limitation, not an operator correctness failure.

TODO: rerun inside a supported Linux/WSL/CUDA/Triton environment with repository dependencies installed.

## Benchmark validation

1. Define baseline, input sizes, dtype, and layout before running.
2. Run a benchmark command in a reproducible environment.
3. Record exact command, result, unit, and conclusion.
4. Reject conclusions that do not include numeric evidence.

Result:

```text
待真实运行后填写
```

## Failure diagnosis validation

Recorded local failure diagnosis:

- Symptom: pytest collection failed.
- Error message: `ModuleNotFoundError: No module named 'triton'`.
- Root cause: local environment lacks Triton dependency required by the `ninetoothed` import path.
- Minimal fix: install a supported Triton environment or rerun inside Linux/WSL/CUDA environment with repository dependencies installed.
- Re-run command: `pytest tests/test_add.py tests/test_addmm.py tests/test_pow.py -q`.
- Re-run result: 待真实运行后填写.

The agent must not delete tests, bypass assertions, or fabricate a pass.

## Skill vs no-skill comparison

Compare two runs on the same operator task:

- No-skill baseline: agent receives only the operator request.
- Skill-guided run: agent follows `skills/competition/lirui-ninetoothed-operator-skill/SKILL.md` and fills the task template.

Compare:

- Whether requirements were extracted before implementation.
- Whether non-contiguous layout was tested.
- Whether PyTorch reference was used.
- Whether failures were diagnosed with minimal fixes.
- Whether benchmark claims included command and result.
- Whether PR integration rules were respected.

Result:

```text
待真实运行后填写
```
