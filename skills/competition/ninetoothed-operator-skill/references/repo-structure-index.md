# NineToothed Repository Structure Index

> Read this before writing any code. Understanding the repo layout lets you
> find existing patterns instead of re-inventing them.

## Repository: InfiniTensor/ninetoothed

```
ninetoothed/
├── README.md               ← Start here: DSL overview, installation, quick examples
├── CONTRIBUTING.md         ← PR rules, branch naming, commit style, ruff/pytest
├── docs/                   ← Concept explanations (arrangement, application, make)
├── src/ninetoothed/
│   ├── __init__.py         ← Public API: Tensor, Symbol, make, block_size, jit
│   └── language.py         ← ntl.* ops: exp, log, sqrt, rsqrt, sum, max, cast, dot
└── tests/                  ← Unit tests for the compiler (not operator tests)
```

## Repository: InfiniTensor/ninetoothed-examples

```
ninetoothed-examples/
├── ops/ninetoothed/kernels/
│   ├── add.py              ← Pattern A: elementwise add (simplest example)
│   ├── softmax.py          ← Pattern B: row-wise reduction, online two-pass
│   ├── rms_norm.py         ← Pattern B: reduction + normalization
│   └── mm.py               ← Pattern C: matrix multiplication with autotuning
├── ops/ninetoothed/kernels/
│   ├── element_wise.py     ← Generic elementwise arrangement factory
│   └── reduction.py        ← Generic reduction arrangement factory (dim, ndim)
├── tests/
│   └── test_*.py           ← Reference correctness tests (pytest -v)
└── bench/
    └── bench_*.py          ← Benchmark scripts using triton.testing.perf_report
```

## What to search before writing a new operator

1. **Arrangement pattern** → look in `ninetoothed-examples/ops/ninetoothed/kernels/element_wise.py`
   and `reduction.py` first. Do not write a new arrangement from scratch if a
   generic factory already covers your operator type.

2. **Similar operator** → search `ninetoothed-examples/ops/ninetoothed/kernels/` for an
   operator with the same parallelism axis (elementwise → `add.py`,
   row-reduction → `softmax.py` or `rms_norm.py`).

3. **Test structure** → look in `ninetoothed-examples/tests/` for `test_<similar_op>.py`
   to see the expected correctness test format, shape parametrization, and
   tolerance values.

4. **Benchmark structure** → look in `ninetoothed-examples/bench/bench_<similar_op>.py`
   for the `triton.testing.perf_report` template.

5. **CONTRIBUTING.md** → always read before creating files or writing commit messages.
   Branch name format: `spring-2026-<githubid>-t3-1-1`.
   Commit style: imperative, capitalized, no trailing punctuation.

## Key files in this skill package

```
ninetoothed-operator-skill/
├── ops/relu.py             ← Pattern A reference: 1D elementwise
├── ops/softmax.py          ← Pattern B reference: row-wise reduction (online two-pass)
├── ops/rms_norm.py         ← Pattern B reference: normalization, non-contiguous inputs
├── tests/test_relu.py                      ← 12 correctness + 2 benchmark tests
├── tests/test_softmax.py                   ← 9 correctness + 3 benchmark tests
├── tests/test_rms_norm_noncontiguous.py    ← 7 correctness + 1 copy-overhead benchmark test
├── tests/test_softmax_perf.py              ← 10 benchmark + generated-source dump tests
└── examples/self_test_{1,2,3,4}.md        ← Self-test records with results
```
