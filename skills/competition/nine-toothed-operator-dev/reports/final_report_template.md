# <Team Name>_NineToothed skill Innovation_T3-1-1_Final Report

## 1. Skill Goal

This submission provides `nine-toothed-operator-dev`, a reusable skill for guiding AI agents through NineToothed operator implementation, correctness testing, benchmark analysis, and failure diagnosis.

## 2. Design Principles

- Correctness before performance.
- Minimal repository-style patches.
- Explicit task contracts for shape, dtype, broadcast, layout, and unsupported cases.
- Reproducible commands and results.
- Performance claims backed by benchmark evidence.

## 3. Package Structure

Describe `SKILL.md`, references, scripts, examples, tests, and report files.

## 4. Core Workflow

Summarize the required workflow from task parsing to final audit note.

Include how the skill forces the agent to verify:

- arranged outer tensor shape alignment
- per-program block shape and dtype nesting
- identity fill for tail tiles
- PyTorch reference correctness
- benchmark evidence for performance-sensitive work
- closed-loop diagnosis for generated source, AOT, and failing tests

## 5. Self-Test Task 1: Elementwise/Broadcast

- Task: Verify existing public examples add operator and benchmark workflow.
- Files changed: none for the public repository; results recorded under `examples/01-elementwise-add/task.md`.
- Correctness command: `/usr/local/miniconda3/envs/py312/bin/python -m pytest tests/test_ops.py::TestAdd::test_correctness -q`
- Correctness result: `1 passed in 5.58s`.
- Benchmark command: `/usr/local/miniconda3/envs/py312/bin/python -m pytest tests/test_benchmarks.py::TestAddBenchmark::test_benchmark -q -m benchmark`
- Benchmark result: `1 passed in 9.80s`; custom fp16 shape `(98432,)`: NineToothed 0.0512 ms, PyTorch 0.0205 ms.
- Conclusion: correctness passed; performance claims must be backed by benchmark data.

## 6. Self-Test Task 2: Reduction/Block

- Task: Verify softmax reduction correctness and diagnose public examples benchmark failure.
- Files changed: none for the public repository; results recorded under `examples/02-softmax-reduction/task.md`.
- Correctness command: `/usr/local/miniconda3/envs/py312/bin/python -m pytest tests/test_softmax.py -q`
- Correctness result: `1 passed in 1.86s`.
- Benchmark command: `/usr/local/miniconda3/envs/py312/bin/python -m pytest tests/test_benchmarks.py::TestSoftmaxBenchmark::test_benchmark -q -m benchmark`
- Benchmark result: blocked by exact Triton comparison precheck. Custom fp16 shape `(4096, 781)`: PyTorch allclose at `atol=1e-3`, max abs diff `1.52587890625e-05`, NineToothed 0.0543 ms, PyTorch 0.0215 ms.
- Conclusion: core softmax correctness passed; benchmark harness needs tolerance adjustment or separate provider comparison for Triton.

## 7. Self-Test Task 3: Layout-Sensitive

- Task: Probe non-contiguous add behavior with stepped views.
- Files changed: none for the public repository; diagnostic script result recorded.
- Correctness command: `/usr/local/miniconda3/envs/py312/bin/python /root/ninetoothed-skill-work/selftest_custom.py`
- Correctness result: `layout add allclose False`.
- Layout evidence: input shape `(256, 256)`, stride `(512, 2)`, `is_contiguous=False`.
- Unsupported cases: stepped views, transposed views, negative strides, overlapping storage, and arbitrary offsets are unproven for the public examples add wrapper.
- Conclusion: this validates the skill rule that layout-sensitive tasks must not assume contiguous input.

## 8. Self-Test Task 4: Performance/Diagnosis

- Task: Diagnose AOT failure and collect lightweight benchmark data.
- Failure or regression symptom: AOT test failed.
- Diagnosis path: first failure was `python` not on PATH; after adding conda env to PATH, AOT failed on missing `nvcc`.
- Fix or mitigation: use a full CUDA Toolkit image or install/expose `nvcc`, then verify `nvcc --version`.
- Correctness command: `/usr/local/miniconda3/envs/py312/bin/python -m pytest tests/test_add.py -q` and `tests/test_softmax.py -q`.
- Correctness result: add `1 passed in 8.69s`; softmax `1 passed in 1.86s`.
- Benchmark command: `/usr/local/miniconda3/envs/py312/bin/python /root/ninetoothed-skill-work/selftest_custom.py`.
- Benchmark result: add 0.0512 ms vs PyTorch 0.0205 ms; softmax 0.0543 ms vs PyTorch 0.0215 ms.
- Conclusion: runtime correctness and lightweight benchmarks are available; AOT remains blocked until `nvcc` is installed.

## 9. Baseline Comparison

Compare the same AI agent without this skill and with this skill:

| Task | Without Skill | With Skill | Improvement |
| --- | --- | --- | --- |
| Elementwise | TODO | TODO | TODO |
| Reduction | TODO | TODO | TODO |
| Layout | TODO | TODO | TODO |
| Diagnosis | TODO | TODO | TODO |

## 10. Safety and Compliance

No secrets, no hidden answers, no test bypasses, and no online-only dependencies are included.

## 11. Dependencies

Remote validation environment:

- Cloud GPU: NVIDIA GeForce RTX 4090 24 GB
- Driver/CUDA reported by `nvidia-smi`: driver 570.169, CUDA 12.8
- OS image: Ubuntu 22.04, CUDA 12.8, PyTorch image
- Python: 3.12.11 from `/usr/local/miniconda3/envs/py312/bin/python`
- PyTorch: 2.9.1+cu128
- Triton: 3.5.1
- NineToothed: editable install from public repository snapshot
- Known blocker: AOT tests require `nvcc`, which was not present in the rented image PATH

## 12. References

See `REFERENCE.md`.

## 13. Maintenance Plan

Update references when NineToothed APIs or benchmark harnesses change. Add new self-test tasks when new operator families are introduced.
