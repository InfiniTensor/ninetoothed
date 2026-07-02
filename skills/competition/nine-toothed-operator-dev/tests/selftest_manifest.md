# Self-Test Manifest

Seven self-test tasks covering the four required categories. Tasks 05 and 06 are full from-scratch operator developments; tasks 01-04 exercise the skill against existing repository code and diagnostic scenarios; task 07 is a controlled with/without-skill A/B comparison on a fresh operator.

| Task | Category | Correctness Result | Benchmark Result | Status |
| --- | --- | --- | --- | --- |
| 01-elementwise-add | Elementwise/broadcast | `1 passed in 5.58s` | `1 passed in 9.80s`; custom 0.0512 ms vs torch 0.0205 ms | Complete |
| 02-softmax-reduction | Reduction/block | Core `1 passed in 1.86s`; examples Triton exact-match comparison failed | Public benchmark blocked by Triton exact-match precheck; custom 0.0543 ms vs torch 0.0215 ms | Complete with diagnosis |
| 03-layout-stride-offset | Non-contiguous layout | Root cause traced (rank mismatch → silent 1-column processing); fixes (a) flatten guard and (b) rank-2 kernel both `allclose=True` on GPU | Not applicable | Complete, closed loop |
| 04-performance-diagnosis | Performance/diagnosis | Core add/softmax passed | AOT re-verified with nvcc 12.8 (see 04 log); custom timings collected | Complete, closed loop |
| 05-from-scratch-gelu | Elementwise (from scratch) | 4/4 allclose vs torch GELU tanh, incl. non-power-of-two and boundary sizes | 0.99x–1.05x vs PyTorch across 2^16–2^24; no regression | Complete, incl. NameError failure-diagnosis loop |
| 06-from-scratch-l2norm | Reduction/block (from scratch) | 4/4 allclose vs PyTorch reference, fp16/fp32, odd sizes | 0.72x–0.80x of PyTorch time (20-28% faster, fused single kernel) | Complete |
| 07-ab-comparison | Meta: skill A/B on softmax-temperature | A: 8/8 allclose; B: 9/9 (incl. non-contiguous view) | Both 0.50x–0.75x vs unfused torch baseline; checklist A 8.5/10 vs B 10/10 | Complete, contamination caveat disclosed |

Benchmark coverage: tasks 01, 02, 05, 06, 07 (requirement: at least 2).
Failure-diagnosis coverage: tasks 02 (harness tolerance), 03 (silent rank mismatch), 04 (missing nvcc → re-verified), 05 (application globals NameError).
