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

[近期修订的 RTX4090 实机核对](docs/cpu_interpreter_layout_gpu.md) 已对精确版本 0706213 完成 15 项实际 GPU 差分和 108 项布局/表达式语义回归；196 份提交文件与远端输入逐一匹配。该记录用于正确性，不是新的 GPU 性能成绩。

## 无跟踪矩阵乘的地址图复用

[执行内几何缓存](docs/cpu_interpreter_geometry_performance.md) 在三个小矩阵的 CPU 解释执行中，三轮目标几何平均约快 3.85 倍，代价是约 141–183 KB 的额外执行分配峰值。该轮源码通过 793 项 CPU 回归、15 项原有真实 GPU 差分和 3 项目标矩阵差分；该轮保留跟踪与回调的原路径；普通跟踪的后续优化见下节，数组数值始终实时读取。

## 几何缓存命中路径的增量优化

[相关符号键与证明复用](docs/cpu_interpreter_geometry_symbols.md) 相对c2c35ec，在同一组CPU矩阵乘目标上三轮再获得1.128–1.133倍加速；峰值增加约0.04%–0.37%。796项CPU回归、15+3项实际GPU差分通过。初稿未达门槛的结果一并保留，不把不同阶段的加速比相乘。

## 普通跟踪的地址几何复用

[跟踪执行优化](docs/cpu_interpreter_traced_geometry.md) 在三个CPU矩阵乘目标上，相对a1ef123三轮约快2.38倍，完整轨迹一致；用户回调、watch、handlers和事件过滤器保留原路径。799项CPU、15项常规及3项带跟踪目标实机差分通过，峰值增加约168–175KB。协议的一处继承文案错误已明确记录，数值配置和原始结果未改。

## dtype快照格式化实验

[三版尝试与未采用原因](docs/cpu_interpreter_dtype_display.md) 保留字符串共享差异和控制点退化结果。候选虽在部分跟踪用例提速，但未满足完整门槛；生产格式化路径保持原样。

## 最终版本 A100 验收

[完整验收报告](docs/cpu_interpreter_a100_final.md)：最终功能源码a86bea9在A100-SXM4-40GB通过全库1206项、仅2项多GPU条件跳过；正式GPU差分15/15及额外跟踪目标3/3通过，所有源码指纹与测试集合核对完成。原镜像链接问题及第一台GPU的ECC硬件故障分别归档，迁移后完整重跑通过。

## 2026-09-20 上游兼容性同步

[同步说明与回归](docs/upstream_sync_20260920.md)：已合入官方整数dtype别名规范#218，三方合并保留解释器追踪逻辑。NumPy-only回归799通过，Torch CPU相关回归96通过（含新增40项别名测试）；GPU相关排除项和原A100证据的版本边界在报告中明确列出。
