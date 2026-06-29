# NineToothed / ntops 速查（v0.3）

## ntops 标准内核（优先用这个）

```python
import functools
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ntops.kernels.element_wise import arrangement

def application(input, output):
    output = ...  # noqa: F841

def premake(ndim, dtype=None, block_size=None):
    arrangement_ = functools.partial(arrangement, block_size=block_size)
    tensors = (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype))
    return arrangement_, application, tensors
```

二元（含 alpha）见 `add.py`：`Tensor(0, dtype=ninetoothed.float64)` 作为 alpha。

## torch 封装

```python
from ntops.torch.utils import _cached_make
kernel = _cached_make(ntops.kernels.<op>.premake, input.ndim)
kernel(...)
```

## 与 examples 仓库差异

| | ntops（生产） | ninetoothed-examples（教学） |
|--|---------------|------------------------------|
| 入口 | `premake` + `_cached_make` | 顶层 `ninetoothed.make` |
| arrangement | `element_wise.arrangement` | 手写 `.tile` |
| scaffold | `--style ntops` | `--style examples` |

## 目录

```
ntops/
  src/ntops/kernels/<op>.py
  src/ntops/kernels/__init__.py
  src/ntops/kernels/element_wise.py
  src/ntops/torch/<op>.py
  tests/test_<op>.py
```

## 链接

- ntops: https://github.com/InfiniTensor/ntops
- examples: https://github.com/InfiniTensor/ninetoothed-examples
- 文档: https://ninetoothed.org/

## GPU 环境提示

- SeetaCloud/AutoDL：常需 `source /root/miniconda3/bin/activate base`
- 无 `nvidia-smi` 时 pytest cuda 用例会 skip
