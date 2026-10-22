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

### CUDA Accumulation Policy

For eligible FP16 dot/add loops with a zero-initialized FP32 accumulator, the CUDA backend defaults to `cuda_wmma_accumulation="fused"`. It keeps one FP32 WMMA accumulator across the source K groups, preserving the existing CUDA accumulation behavior. This mode does not promise a separate FP32 rounding boundary after every source dot/add group.

Pass `cuda_wmma_accumulation="ordered"` to `ninetoothed.make(..., backend="cuda")` when those group boundaries matter. This mode resets each dot partial and adds it to the carry in source order with a round-to-nearest FP32 addition. Both modes keep FP32 accumulation; their results, including a final FP16 conversion, can differ numerically. The selected mode, accumulator dtype, and rounding policy are recorded in artifact metadata. The policy applies only to the validated pure dot/add loop; other types and regions retain ordinary lowering.

## Useful Links

- [NineToothed Documentation](https://ninetoothed.org/)
- [NineToothed Operators](https://github.com/InfiniTensor/ntops)
- [NineToothed Examples](https://github.com/InfiniTensor/ninetoothed-examples)

## License

This project is distributed under the Apache-2.0 license. See the included [LICENSE](LICENSE) file for details.
