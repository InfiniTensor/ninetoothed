# ninetoothed-operator-skill

A `.skill` for guiding AI agents to write, verify, optimize, and debug
NineToothed GPU operators.

## Scope

- Elementwise / broadcast operators (relu, gelu, add, mul, …)
- Reduction / block operators (softmax, rms_norm)
- Layout-sensitive tasks where non-contiguous inputs use an explicit
  contiguous-copy fallback unless native stride support is proven
- Benchmark and diagnosis tasks (timing, generated-source dump trigger,
  BLOCK_SIZE analysis, failing tests)

## Validated scope

- ReLU elementwise correctness for fp32/fp16 and benchmark timing for fp16
- 2D last-dimension softmax correctness and benchmark comparison
- RMSNorm with fixed `eps=1e-6`
- Non-contiguous input compatibility through explicit `.contiguous()` fallback
- Generated-source dump trigger with post-dump correctness validation

## Documented but not self-test validated

- AOT build workflow
- Detailed generated-source load/store/mask analysis
- fp16 reduction kernels beyond the covered self-tests

## Out of scope

- Matrix multiplication / GEMM implementation guidance or self-test claims
- NineToothed compiler internals
- Raw Triton kernels without the NineToothed DSL
- Native arbitrary-stride 2D reductions
- Dynamic RMSNorm `eps` values

## Package structure

| Path | Purpose |
|------|---------|
| `README.md` | Entry point and self-test summary for human readers |
| `SKILL.md` | Main workflow for AI agents (Step 0-8 SOP and evidence boundaries) |
| `references/dsl-quick-reference.md` | NineToothed DSL cheat sheet |
| `references/repo-structure-index.md` | Where to find patterns in ninetoothed-examples |
| `references/operator-patterns.md` | Operator arrangement/application templates |
| `references/v0.26-known-failures.md` | Known NineToothed 0.26.0 failure modes |
| `references/benchmarking-and-diagnostics.md` | Benchmark, generated-source dump, and AOT workflow notes |
| `ops/` | Reference operator implementations |
| `tests/` | 44 correctness, benchmark, generated-source, and diagnosis tests |
| `examples/` | 4 self-test task records with results |
| `scripts/` | Reproducible benchmark and correctness scripts |
| `evidence/` | Validation commands, environment notes, and Colab result summaries |
| `reports/` | Final competition report PDF |

## How to use

1. Open the NineToothed repository in your AI coding tool (Trae, Cursor, Codex).
2. Ask the agent to read `SKILL.md` before implementing any operator task.
3. Follow the Step 0-8 workflow in SKILL.md.
4. Run correctness tests: `python scripts/run_correctness.py`
5. Record results and any failure diagnosis.

## Self-test tasks

| ID | Type | Operator | Correctness | Benchmark |
|----|------|----------|-------------|-----------|
| T1 | Elementwise / broadcast | ReLU | 12/12 PASSED | 145.0 -> 243.4 GB/s (1M->16M fp16) |
| T2 | Reduction / block | Softmax (online two-pass) | 9/9 PASSED | NT 227.8 GB/s vs PT 242.3 GB/s at (4096,4096); see T4 for shape-wise diagnosis |
| T3 | Layout-sensitive | RMS Norm (non-contiguous fallback) | 7/7 PASSED | contiguous 0.041ms / non-contig 0.067ms (+65.6%) |
| T4 | Benchmark / diagnosis | Softmax perf + generated-source dump | 10/10 PASSED | NT 235.5 GB/s vs PT 145.5 GB/s at (2048,4096) |

Softmax performance varies across shapes in the recorded T4 runs: (4096,4096)
is close to the PyTorch baseline, while the separated throughput benchmark shows
NT ahead at (2048,4096). See `evidence/pytest-benchmark.txt` for raw logs.

Validation on Google Colab T4 GPU with NineToothed 0.26.0:

- Correctness: 28/28 passed (`python scripts/run_correctness.py`)
- Benchmark / generated source / performance diagnosis: 16/16 passed
- Overall: 44/44 selected validation targets passed across the two runs

## Requirements

```
ninetoothed==0.26.0
torch>=2.0
triton>=2.0
pytest
```

## Run correctness tests

```bash
pip install -r requirements.txt
python scripts/run_correctness.py
```

## Run benchmarks

```bash
python -m pytest tests/ -m benchmark -v -s --tb=short
```

Useful filters:

```bash
python scripts/run_correctness.py --op relu
python scripts/run_correctness.py --op softmax
python scripts/run_correctness.py --op rms_norm
python scripts/run_benchmark.py --op softmax
```
