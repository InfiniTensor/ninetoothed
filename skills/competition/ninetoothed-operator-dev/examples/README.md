# Examples

One complete worked example per operator family, demonstrating how an agent uses
this `.skill` end to end. Each example directory will contain:

```
<example>/
  task.md          # the input task statement
  kernel.py        # produced arrangement + application + make(...)
  wrapper.py       # torch wrapper (pre-allocate output, call kernel, return)
  test_*.py        # correctness scaffold (from gen_pytorch_oracle.py)
  matrix.csv       # run_correctness_matrix.py output
  bench.py + bench.csv   # for perf-sensitive examples
  trace.md         # agent execution summary + (for diagnosis) fix writeup
```

Planned examples (mirror the four hidden-task families):

| Dir | Family | Demonstrates |
|-----|--------|--------------|
| `01_elementwise_where_mask/` | elementwise / broadcast | broadcast + bool mask + `other=` fill |
| `02_reduction_parameterized/` | reduction / blocking | `reduction in {none,mean,sum}` + fp32 accumulate + benchmark |
| `03_layout_pixel_unshuffle/` | layout-sensitive | space-to-depth via ravel/flatten + non-contiguous test |
| `04_perf_diag_generated_source/` | perf / diagnosis | generated-source inspection + before/after benchmark + minimal fix |

> v0 status: kernels and scaffolds are authored from the verified API; the
> `matrix.csv` / `bench.csv` / `trace.md` artifacts are produced on a CUDA host
> during the self-test run (see `../tests/self_test_tasks.md`).
