# Operator Taxonomy — classify before you write

Pick the family by the **computation shape**, not the op name. Many torch ops
decompose: `mse_loss` = elementwise `(a-b)**2` then reduction `mean`; `fliplr`
= layout `flip(dim=-1)`. Classify each stage.

## Decision rule

```
Does the output element depend only on the same-index input element(s)?
├── yes, 1:1 (maybe with broadcast / mask)        → ELEMENTWISE
└── no
    ├── output folds many inputs along an axis
    │   (sum/mean/max/softmax/norm/argmax)         → REDUCTION
    ├── output is a re-indexing / re-view of input
    │   (flip/narrow/permute/space-to-depth,
    │    non-contiguous / stride / offset inputs)  → LAYOUT
    └── task is "benchmark / inspect / AOT /
        find the regression / fix failing test"    → PERF-DIAG
```

Composite ops: handle the elementwise stage with `elementwise.md`, the fold
with `reduction.md`. A reduction with `reduction='none'` is pure elementwise.

**Expressibility screen (before writing anything).** None of the four
families covers a task whose semantics the DSL cannot express: read/write
addresses driven by tensor *values* (gather/scatter by index tensor),
many-to-one scatter needing atomic RMW, value-dependent control flow,
dynamic output shapes (nonzero/unique), or cross-tile communication beyond
the built-in fold. If the spec has any of these, do not classify it into a
family — take the declared-fallback path (SKILL.md §3 `dsl_limit`;
definitions in `scripts/failure_classifier.py::KNOWN_INEXPRESSIBLE`).

## Family → reference

| Family | Reference | Core idiom |
|--------|-----------|------------|
| Elementwise / broadcast | `elementwise.md` | `tile((BLOCK_SIZE,))` flat; `tile((1, BLOCK_SIZE))` row-wise; `other=` for mask |
| Reduction / blocking | `reduction.md` | row tile + `ntl.sum/max`; fp32 accumulate; `reduction=` modes |
| Layout-sensitive | `layout.md` | `ravel`/`flatten`/`permute`; `offsets()`; contiguous-fallback decision |
| Perf / diagnosis | `perf-diag.md` | `simulate_arrangement`, generated source, benchmark, Roofline, AOT |

When a failure appears at any step, jump to `common-errors.md`.

## Before writing: find prior art

Grep the repo's `tests/test_*.py` files for the nearest existing arrangement
(e.g. `tests/test_add.py` for elementwise, `tests/test_softmax.py` for
reduction/layout-sensitive ops). Reusing an existing arrangement skeleton is
faster and matches repo style — a scoring dimension in the rubric.
