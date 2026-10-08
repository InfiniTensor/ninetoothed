# NineToothed

![NineToothed Logo](https://github.com/InfiniTensor/ninetoothed/raw/master/docs/source/_static/ninetoothed-logo.png)

[![Document](https://img.shields.io/badge/Document-ready-blue)](https://ninetoothed.org/)
[![PyPI Version](https://img.shields.io/pypi/v/ninetoothed?color=cyan)](https://pypi.org/project/ninetoothed/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
![star](https://atomgit.com/InfiniTensor/ninetoothed/star/badge.svg)

NineToothed is a Triton-based domain-specific language (DSL). By introducing **tensor-oriented meta-programming (TOM)**, it makes writing high-performance GPU kernels easier.

## Installation

We can use `pip` to install `ninetoothed`.

```shell
pip install ninetoothed
```

After successfully running the above command, `ninetoothed` will be installed. However, to fully utilize its capabilities, you also need to install a deep learning framework supported by `ninetoothed`. For trial purposes, we recommend installing `torch`.

## Usage

Thanks to tensor-oriented meta-programming, NineToothed can be written using the **arrange-and-apply** paradigm, which involves separately defining `arrangement`, `application`, and `tensors`, and then integrating them using `ninetoothed.make` to generate the kernel.

### Matrix Multiplication

Here is the code we need for matrix multiplication:

```python
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size

BLOCK_SIZE_M = block_size()
BLOCK_SIZE_N = block_size()
BLOCK_SIZE_K = block_size()


def arrangement(input, other, output):
    output_arranged = output.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))

    input_arranged = input.tile((BLOCK_SIZE_M, BLOCK_SIZE_K))
    input_arranged = input_arranged.tile((1, -1))
    input_arranged = input_arranged.expand((-1, output_arranged.shape[1]))
    input_arranged.dtype = input_arranged.dtype.squeeze(0)

    other_arranged = other.tile((BLOCK_SIZE_K, BLOCK_SIZE_N))
    other_arranged = other_arranged.tile((-1, 1))
    other_arranged = other_arranged.expand((output_arranged.shape[0], -1))
    other_arranged.dtype = other_arranged.dtype.squeeze(1)

    return input_arranged, other_arranged, output_arranged


def application(input, other, output):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

    for k in range(input.shape[0]):
        accumulator += ntl.dot(input[k], other[k])

    output = accumulator


tensors = (Tensor(2), Tensor(2), Tensor(2))

kernel = ninetoothed.make(arrangement, application, tensors)
```

## Debugging Without a GPU

`ninetoothed.interpret` runs a lowered program with NumPy instead of a GPU backend. It reuses the existing lowering chain: the arrangement produces the layout, the Python frontend produces the `ssa.Program`, and the interpreter walks that program once per program instance. Because no CUDA device is involved, applications can be developed, traced and diff-tested on any machine.

```python
import numpy as np

import ninetoothed.language as ntl
from ninetoothed import Tensor, block_size
from ninetoothed.interpret import Tracer, interpret

BLOCK_SIZE = block_size()


def arrangement(x, out, BLOCK_SIZE=BLOCK_SIZE):
    return x.tile((1, BLOCK_SIZE)), out.tile((1, BLOCK_SIZE))


def application(x, out):
    shifted = x - ntl.max(x, axis=1)[:, None]
    numerator = ntl.exp(shifted)
    out = numerator / ntl.sum(numerator, axis=1)[:, None]


x = np.random.default_rng(0).random((3, 11), dtype=np.float32)
out = np.zeros_like(x)

result = interpret(
    arrangement,
    application,
    tensors=(Tensor(2, other=float("-inf")), Tensor(2)),
    inputs=(x, out),
    symbols={"BLOCK_SIZE": 16},
    trace=Tracer(opcodes={"mem.store"}),
)

print(result.launch_shape)  # (3, 1)
print(result.output("out"))
print(result.render_trace())
```

The interpreter enforces the same masking contract as the generated code: a masked-out access never touches the backing buffer, while an unmasked access that falls outside the buffer is an error, not a silent wrong read. Integer and boolean results are bit-exact, and `float32` is never silently widened to `float64`.

It also doubles as a differential debugger. `compare_pipeline` runs the same program with and without a pass pipeline; a pipeline has to preserve semantics, so any output difference is a compiler bug. `compare_interpretations` compares any two runs:

```python
from ninetoothed.interpret import compare_pipeline

diff = compare_pipeline(
    arrangement,
    application,
    inputs=(x, out),
    symbols={"BLOCK_SIZE": 16},
    pipeline=["ssa.canonicalize", "ssa.analyze_effects"],
    trace=True,
)

print(diff.render())

if not diff.matches:
    print(diff.minimal_reproduction())
```

When a pipeline breaks semantics, `compare_passes` applies it one pass at a time and names the first pass at fault, then pins the difference on a program instance and the `mem.store` that produced the wrong value:

```python
from ninetoothed.interpret import compare_passes

diff = compare_passes(
    arrangement,
    application,
    inputs=(x, out),
    symbols={"BLOCK_SIZE": 16},
)

print(diff.render())  # one line per cumulative pass prefix
print(diff.localize())  # the pass, the program instance, the store
```

See the [CPU Reference Interpreter](https://ninetoothed.org/python_api/interpret.html) documentation for the supported operation set, the tracing filters, per-pass bisection, minimal reproductions and the known limitations.

## Useful Links

- [NineToothed Documentation](https://ninetoothed.org/)
- [NineToothed Operators](https://github.com/InfiniTensor/ntops)
- [NineToothed Examples](https://github.com/InfiniTensor/ninetoothed-examples)

## License

This project is distributed under the Apache-2.0 license. See the included [LICENSE](LICENSE) file for details.
