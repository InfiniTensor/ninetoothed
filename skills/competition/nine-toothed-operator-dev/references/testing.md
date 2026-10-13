# Testing and Self-Test Design

Use this reference before writing correctness tests or contest self-test logs.

## Correctness Test Checklist

For every operator task, identify:

- semantic reference: PyTorch function, existing implementation, or hand formula
- dtype: at least the requested dtype; include fp32 for diagnosis when fp16 is unstable
- shape cases: normal, small, odd size, boundary tile size, and non-power-of-two size when relevant
- layout cases: contiguous and non-contiguous when the task mentions layout
- tolerance: use tighter tolerance for exact elementwise fp32, looser justified tolerance for fp16 reductions

## Pytest Style

Follow local tests. In the core `ninetoothed` repository, `device` is NOT a fixture; it is a parametrized argument fed by `tests/utils.py::get_available_devices()`. When no CUDA/MLU device exists, that function returns an empty tuple, so the tests are skipped automatically. Do not add `pytest.mark.skipif(not torch.cuda.is_available())` in the core repository; that is not the local style.

```python
import pytest
import torch

from tests.utils import get_available_devices


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float32,))
@pytest.mark.parametrize("shape", ((128,), (98432,)))
def test_my_operator(shape, dtype, device):
    input = torch.randn(shape, dtype=dtype, device=device)
    output = my_operator(input)
    expected = torch_reference(input)
    assert torch.allclose(output, expected, atol=1e-5, rtol=1e-5)
```

Also note: `tests/conftest.py` only seeds RNG per test; it does not provide device or tensor fixtures. Other repositories, such as `ninetoothed-examples`, may use different conventions; always read the nearest existing test first.

## Non-Contiguous Test Patterns

Choose the smallest pattern that proves the contract:

```python
base = torch.randn((64, 128), device=device, dtype=dtype)
input = base[:, ::2]
assert not input.is_contiguous()
```

Other useful views:

```python
input = base.t()
input = base.permute(0, 2, 1)
input = base.narrow(-1, 1, 63)
```

Only use `as_strided` when the target semantics are very explicit.

## Self-Test Task Log Requirements

Each contest self-test task must include:

- input task statement
- AI agent execution summary
- files changed or patch summary
- correctness command and result
- benchmark command and result for at least two tasks
- failure symptom, root cause, fix, and rerun result when relevant

Recommended four tasks:

1. Elementwise/broadcast: add, relu, gelu, silu, or swiglu.
2. Reduction/block: softmax, max_pool2d, rms_norm, or block statistics.
3. Layout-sensitive: non-contiguous stride/offset input.
4. Performance/diagnosis: benchmark regression, generated source inspection, AOT build config, or failing-test repair.

## Result Wording

Use concrete lines:

```text
Command: pytest tests/test_add.py -q
Result: 1 passed in 3.42s
Conclusion: correctness matches torch.add for fp32 size 98432.
```

If blocked:

```text
Command: pytest tests/test_add.py -q
Result: blocked - CUDA unavailable on this machine.
Fallback: import check passed with python -m py_compile ...
Risk: runtime kernel behavior still needs CUDA verification.
```
