# NineToothed

![NineToothed Logo](https://github.com/InfiniTensor/ninetoothed/raw/master/docs/source/_static/ninetoothed-logo.png)

[![Document](https://img.shields.io/badge/Document-ready-blue)](https://ninetoothed.org/)
[![PyPI Version](https://img.shields.io/pypi/v/ninetoothed?color=cyan)](https://pypi.org/project/ninetoothed/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

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

### First-stage serial-to-kernel demo

We can also prototype a much more ergonomic style with `@ninetoothed.parallelize`.
For now this is intentionally a small demo: it recognizes a couple of canonical
serial loop nests and lowers them to existing NineToothed kernels.

```python
import torch
import ninetoothed


@ninetoothed.parallelize
def add(input, other, output):
    for i in range(output.shape[0]):
        output[i] = input[i] + other[i]


input = torch.rand(8192, device="cuda")
other = torch.rand(8192, device="cuda")
output = torch.empty_like(input)

add(input, other, output)
```

The goal of this first step is to show the direction: users can start from
plain serial Python while NineToothed handles tiling and parallelization under
the hood. Today the demo supports canonical `add` and `mm` loop nests and
raises `NotImplementedError` for anything else.

## Useful Links

- [NineToothed Documentation](https://ninetoothed.org/)
- [NineToothed Operators](https://github.com/InfiniTensor/ntops)
- [NineToothed Examples](https://github.com/InfiniTensor/ninetoothed-examples)

## License

This project is distributed under the Apache-2.0 license. See the included [LICENSE](LICENSE) file for details.
