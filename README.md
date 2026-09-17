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

For the NumPy CPU interpreter and step debugger without Torch or Triton, follow
the [CPU wheel installation instructions](docs/source/installation.rst#cpu-interpreter-without-gpu-packages).
This explicit dependency override installs a normal wheel from this checkout;
the default GPU installation above is unchanged. See the
[CPU interpreter guide](docs/source/cpu_interpreter.rst) for supported operations.
The [validation summary](docs/cpu_interpreter_acceptance.md) records the tested
scope, reproduction commands, implementation tradeoffs and fixed evidence links.

[恒等布局内存优化](docs/cpu_interpreter_identity_memory.md) 在大张量测试中降低约 36%–42% 的执行分配峰值，并修复零维张量掩码写入；577 项 CPU 回归与 15 项真实 GPU 差分通过。

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

## Useful Links

- [NineToothed Documentation](https://ninetoothed.org/)
- [NineToothed Operators](https://github.com/InfiniTensor/ntops)
- [NineToothed Examples](https://github.com/InfiniTensor/ninetoothed-examples)

## License

This project is distributed under the Apache-2.0 license. See the included [LICENSE](LICENSE) file for details.

### 恒等布局读取快照优化

[读取快照优化](docs/cpu_interpreter_identity_read.md) 在未安排布局、以读取为主的公开 SSA 测试中实现三轮 7.15–7.61× 的几何平均加速；32 项新边界测试、609 项纯 CPU 回归和 15 项真实 GPU 差分通过。数值只是指定 CPU 解释器测试的收益，完整轨迹及带输出写入的对照均保留。

### 调试轨迹的活动地址记录

[活动地址记录优化](docs/cpu_interpreter_active_addresses.md) 减少稀疏掩码下的临时分配；记录层微基准三轮几何平均约快 24%，实际前端程序另作完整轨迹对照。662 项 CPU 回归、15 项真实 GPU 差分通过，存储依赖和重叠字节语义保持一致。

### 布局调用的边界语义

[布局调用修复](docs/cpu_interpreter_layout_calls.md) 明确拒绝 IR 无法表示的关键字参数，并使零长度 padded shape 保持为零。新增 67 项回归，729 项 CPU 检查通过；此轮属于语义修复，没有新 GPU 性能结论。

[表达式语义回归与负实验](docs/cpu_interpreter_expression_semantics.md) 新增 41 项测试，完整 CPU 选择范围为 770 passed / 15 GPU deselected。两项求值优化未达到预设性能门槛，生产求值器保持原版，候选源码与完整原始数据已归档。
