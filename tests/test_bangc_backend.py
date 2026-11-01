import pytest

import ninetoothed as nt
import ninetoothed.language as ntl
from tests._native_backend_cases import (
    _application,
    _arrangement,
    _conditional,
    _entropy,
    _loop_arrangement,
    _loop_matmul,
    _row_arrangement,
    check_native_baseline,
)


def _atomic_arrangement(x, out):
    return x.tile((128,)), out.tile((1,))


def _atomic_sum(x, out):
    ntl.atomic_add(out.data_ptr(), ntl.sum(x))


def test_bangc_scalar_source_contract():
    from ninetoothed.compiler import lower

    artifact = lower(
        _arrangement,
        _application,
        tuple(nt.Tensor(1) for _ in range(3)),
        backend="bangc",
        kernel_name="bangc_baseline",
    )
    source = artifact.primary_source
    assert "__mlu_entry__" in source
    assert "cnrtGetLastError" in source
    assert "launch_bangc_baseline" in source
    assert artifact.backend.value == "bangc"
    artifact = lower(
        _loop_arrangement,
        _loop_matmul,
        tuple(nt.Tensor(shape=(8, 8), dtype=nt.float32) for _ in range(4)),
        backend="bangc",
        kernel_name="bangc_bounded_loop",
    )
    assert "__nram__" in artifact.primary_source
    artifact = lower(
        _arrangement,
        _conditional,
        tuple(nt.Tensor(shape=(8,), dtype=nt.float32) for _ in range(2)),
        backend="bangc",
    )
    assert "__nram__" not in artifact.primary_source
    artifact = lower(
        _row_arrangement,
        _entropy,
        (
            nt.Tensor(shape=(8, 32), dtype=nt.float32),
            nt.Tensor(shape=(8,), dtype=nt.float32),
        ),
        backend="bangc",
        kernel_name="bangc_bounded_reductions",
    )
    assert "__nram__" in artifact.primary_source


def test_bangc_native_arch_uses_current_device(monkeypatch):
    from types import SimpleNamespace

    import torch

    from ninetoothed.backends.toolchain import resolve_bangc_arch

    monkeypatch.delenv("NINETOOTHED_BANGC_ARCH", raising=False)
    monkeypatch.setattr(
        torch,
        "mlu",
        SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: 1,
            get_device_capability=lambda device: ((5, 0), (3, 0))[device],
        ),
        raising=False,
    )
    assert resolve_bangc_arch("native") == "compute_30"
    assert resolve_bangc_arch("compute_50") == "compute_50"


def test_bangc_jit_aot_reload_and_numerics(tmp_path):
    import torch

    pytest.importorskip("torch_mlu")

    if not torch.mlu.is_available():
        pytest.skip("MLU device required")

    check_native_baseline(tmp_path, backend="bangc", device="mlu")

    kernel = nt.make(
        _atomic_arrangement,
        _atomic_sum,
        (nt.Tensor(1), nt.Tensor(1)),
        backend="bangc",
        max_num_configs=1,
    )
    out = torch.tensor([2**55 + 1], dtype=torch.int64).to("mlu")
    kernel(torch.ones(128, dtype=torch.int64).to("mlu"), out)
    torch.mlu.synchronize()
    assert out.cpu().item() == 2**55 + 129
