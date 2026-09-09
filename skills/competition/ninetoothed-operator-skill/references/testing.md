# Testing Guide

## Test File Structure

```python
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

ninetoothed = pytest.importorskip("ninetoothed")

if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

from <operator> import <func>  # noqa: E402
```

Key points:
- `pytest.importorskip("ninetoothed")` must come **before** importing operator code.
- CUDA skip uses `allow_module_level=True` so the entire file skips, not just one test.
- `sys.path` insert is needed so test files can import sibling operator modules.

## Test Coverage Checklist

- Same-shape correctness vs PyTorch reference
- Broadcast case (if applicable)
- Non-contiguous / stride / offset input (layout-sensitive tasks)
- Edge sizes (non-power-of-2, small, large)
- `torch.allclose` with appropriate tolerance (atol=1e-6, rtol=1e-5 for float32)
- Output shape assertion
- Error handling (invalid input raises expected exception)
