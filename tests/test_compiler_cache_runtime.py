from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from ninetoothed.compiler.cache import stable_digest, write_source
from ninetoothed.compiler.runtime import _public_values
from ninetoothed.ir import LaunchABI, TensorSpec


class _Tensor:
    def __init__(self, shape, dtype="float32", device_type="cuda", device_index=0):
        self.shape = shape
        self.dtype = dtype
        self.device = SimpleNamespace(type=device_type, index=device_index)


def _abi():
    return LaunchABI(public_args=("x", "out"), outputs=("out",))


def _specs():
    return (
        TensorSpec(name="x", ndim=2, shape=("m", "n"), dtype="float32"),
        TensorSpec(name="out", ndim=2, shape=("m", "n"), dtype="float32"),
    )


def test_runtime_binding_rejects_unknown_duplicate_and_missing_arguments():
    x = _Tensor((2, 3))
    out = _Tensor((2, 3))

    with pytest.raises(TypeError, match="Unknown kernel arguments"):
        _public_values(_abi(), (x, out), {"extra": 1}, specs=_specs())

    with pytest.raises(TypeError, match="passed twice"):
        _public_values(_abi(), (x,), {"x": x, "out": out}, specs=_specs())

    with pytest.raises(TypeError, match="Missing kernel arguments"):
        _public_values(_abi(), (x,), {}, specs=_specs())


@pytest.mark.parametrize(
    "value, message",
    (
        (_Tensor((6,)), "rank 1; expected 2"),
        (_Tensor((2, 3), dtype="float16"), "dtype float16; expected float32"),
        (_Tensor((2, 3), device_type="cpu"), "must be on a CUDA device"),
    ),
)
def test_runtime_binding_validates_tensor_contract(value, message):
    with pytest.raises(TypeError, match=message):
        _public_values(
            _abi(),
            (value, _Tensor((2, 3))),
            {},
            specs=_specs(),
        )


def test_content_digest_is_stable_for_a_b_a_sources():
    a1 = stable_digest({"name": "same", "source": "A"})
    b = stable_digest({"name": "same", "source": "B"})
    a2 = stable_digest({"name": "same", "source": "A"})
    assert a1 == a2
    assert a1 != b


def test_concurrent_source_writes_are_atomic(tmp_path, monkeypatch):
    import ninetoothed.compiler.cache as cache

    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    source = "def kernel():\n    return 1\n"

    with ThreadPoolExecutor(max_workers=8) as executor:
        paths = tuple(
            executor.map(
                lambda _: write_source("kernel", source, "py"),
                range(32),
            )
        )

    assert len(set(paths)) == 1
    assert paths[0].read_text(encoding="utf-8") == source
