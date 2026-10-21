import collections
import contextlib
from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks.ascend_report.e2e_attention import (
    Adapter,
    MaskData,
    UnsupportedAttention,
    classify_mask,
    plan_attention,
)

DEVICE = SimpleNamespace(type="npu")


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.ndim = self.values.ndim
        self.dtype = self.values.dtype
        self.device = DEVICE
        self._version = 0

    def repeat_interleave(self, repeats, dim):
        return Tensor(np.repeat(self.values, repeats, axis=dim))

    def stride(self):
        return tuple(value // self.values.itemsize for value in self.values.strides)

    def contiguous(self):
        return Tensor(np.ascontiguousarray(self.values))

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values.tolist()


def torch_stub():
    return SimpleNamespace(
        float16=np.dtype("float16"),
        bfloat16="bfloat16",
        bool=np.dtype("bool"),
        _C=SimpleNamespace(_DisableTorchDispatch=contextlib.nullcontext),
        nn=SimpleNamespace(
            functional=SimpleNamespace(scaled_dot_product_attention=object())
        ),
    )


def reference(query, key, value, *, scale, causal=False, mask=None):
    scores = (
        query.astype(np.float64) @ np.swapaxes(key.astype(np.float64), -1, -2)
    ) * scale

    if causal:
        scores = np.where(
            np.arange(key.shape[-2])[None, :] <= np.arange(query.shape[-2])[:, None],
            scores,
            -np.inf,
        )

    if mask is not None:
        scores = np.where(mask, scores, -np.inf)

    probability = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    probability /= probability.sum(axis=-1, keepdims=True)

    return (probability @ value.astype(np.float64)).astype(np.float16)


def arrays(query_length=19, key_length=19, query_heads=4, key_heads=2):
    rng = np.random.default_rng(19)

    return tuple(
        Tensor(rng.normal(size=shape).astype(np.float16))
        for shape in (
            (1, query_heads, query_length, 64),
            (1, key_heads, key_length, 64),
            (1, key_heads, key_length, 64),
        )
    )


@pytest.mark.parametrize("backend", ("triton", "ntops"))
@pytest.mark.parametrize(
    "query_length,key_length,causal", ((19, 19, True), (1, 19, False), (1, 146, False))
)
def test_prefill_and_cached_decode_preserve_gqa_and_scale(
    backend, query_length, key_length, causal
):
    query, key, value = arrays(query_length, key_length)
    calls = []

    def implementation(q, k, v, **kwargs):
        calls.append((q, k, v, kwargs))

        return Tensor(
            reference(
                q.values,
                k.values,
                v.values,
                scale=kwargs["scale"],
                causal=kwargs["is_causal"],
            )
        )

    recorder = SimpleNamespace(
        collect=True,
        calls=collections.Counter(),
        shapes=collections.Counter(),
        source_imports={},
    )
    adapter = Adapter(backend, torch_stub(), recorder, implementation=implementation)
    output = adapter(query, key, value, is_causal=causal, enable_gqa=True, scale=0.1)
    expanded_key = np.repeat(key.values, 2, axis=1)
    expanded_value = np.repeat(value.values, 2, axis=1)
    expected = reference(
        query.values, expanded_key, expanded_value, scale=0.1, causal=causal
    )
    np.testing.assert_array_equal(output.values, expected)
    assert len(calls) == 1
    assert calls[0][0] is query
    np.testing.assert_array_equal(calls[0][1].values, expanded_key)
    assert calls[0][3] == {"scale": 0.1, "is_causal": causal, "enable_gqa": False}
    assert recorder.calls == {"sdpa": 1}
    assert recorder.active_operation == "sdpa"
    assert adapter.provenance()["numeric_fallback"] is False


def test_decode_does_not_apply_upper_left_causal_mask():
    query, key, value = arrays(1, 7, query_heads=2, key_heads=2)
    query.values.fill(0)
    key.values.fill(0)
    value.values[:] = np.arange(7)[None, None, :, None]

    def implementation(q, k, v, **kwargs):
        return Tensor(
            reference(
                q.values,
                k.values,
                v.values,
                scale=kwargs["scale"],
                causal=kwargs["is_causal"],
            )
        )

    adapter = Adapter("ntops", torch_stub(), implementation=implementation)
    output = adapter(query, key, value, is_causal=False)
    np.testing.assert_array_equal(
        output.values, np.full(query.shape, 3, dtype=np.float16)
    )

    with pytest.raises(UnsupportedAttention, match="Non-square causal"):
        adapter(query, key, value, is_causal=True)


@pytest.mark.parametrize("boolean", (False, True))
def test_square_causal_mask_matches_explicit_mask_reference(boolean):
    query, key, value = arrays(3, 3, query_heads=2, key_heads=2)
    allowed = np.tril(np.ones((3, 3), dtype=bool))
    mask = Tensor(
        allowed if boolean else np.where(allowed, 0.0, -np.inf).astype(np.float32)
    )
    seen = []

    def implementation(q, k, v, **kwargs):
        seen.append(kwargs)

        return Tensor(
            reference(
                q.values,
                k.values,
                v.values,
                scale=kwargs["scale"],
                causal=kwargs["is_causal"],
            )
        )

    adapter = Adapter("triton", torch_stub(), implementation=implementation)
    output = adapter(query, key, value, attn_mask=mask)
    expected = reference(
        query.values, key.values, value.values, scale=64**-0.5, mask=allowed
    )
    np.testing.assert_array_equal(output.values, expected)
    assert seen[0]["is_causal"] is True
    assert seen[0]["scale"] == 64**-0.5


def test_mask_cache_revalidates_mutation_and_query_shape():
    query, key, value = arrays(1, 7, query_heads=2, key_heads=2)
    mask = Tensor(np.ones((1, 1, 1, 7), dtype=bool))
    adapter = Adapter("ntops", torch_stub(), implementation=lambda q, k, v, **kw: q)
    adapter(query, key, value, attn_mask=mask)
    adapter(query, key, value, attn_mask=mask)
    assert adapter.provenance()["mask_inspections"] == 1
    assert adapter.provenance()["mask_cache_hits"] == 1
    mask.values[..., 3] = False
    mask._version += 1

    with pytest.raises(UnsupportedAttention, match="general mask"):
        adapter(query, key, value, attn_mask=mask)

    single_row = Tensor(np.array([[True, False, False]], dtype=bool))
    assert adapter._mask_kind(single_row, (1, 2, 1, 64), 3) == "lower_triangular"

    with pytest.raises(UnsupportedAttention, match="general mask"):
        adapter._mask_kind(single_row, (1, 2, 2, 64), 3)


@pytest.mark.parametrize("backend", ("triton", "ntops"))
def test_different_kv_batch_head_strides_are_normalized_for_both_kernels(backend):
    query, key, value = arrays(3, 3, query_heads=2, key_heads=2)
    value = Tensor(np.repeat(value.values, 2, axis=-1)[..., ::2])
    assert key.stride()[:2] != value.stride()[:2]
    observed = []
    adapter = Adapter(
        backend,
        torch_stub(),
        implementation=lambda q, k, v, **kw: observed.append((k, v)) or q,
    )
    adapter(query, key, value)
    prepared_key, prepared_value = observed[0]
    assert prepared_key.stride()[:2] == prepared_value.stride()[:2]
    np.testing.assert_array_equal(prepared_key.values, key.values)
    np.testing.assert_array_equal(prepared_value.values, value.values)
    assert adapter.provenance()["kv_layout_normalizations"] == 1


@pytest.mark.parametrize(
    "mask",
    (
        MaskData((1, 3), [[True, False, True]], True),
        MaskData((1, 3), [[0.0, -0.1, 0.0]], False),
        MaskData((1, 3), [[0.0, float("nan"), 0.0]], False),
        MaskData((1, 3), [[False, False, False]], True),
    ),
)
def test_nonrepresentable_masks_are_rejected(mask):
    with pytest.raises(UnsupportedAttention):
        classify_mask(mask, 1, 2, 1, 3)


def test_cached_offset_causal_mask_is_not_confused_with_square_prefill():
    mask = MaskData((2, 5), [[True, True, True, True, False], [True] * 5], True)

    with pytest.raises(UnsupportedAttention, match="offset-causal"):
        plan_attention((1, 4, 2, 64), (1, 4, 5, 64), (1, 4, 5, 64), mask=mask)


@pytest.mark.parametrize(
    "options",
    (
        {"dropout_p": 0.1},
        {"scale": float("inf")},
        {"scale": float("nan")},
        {"is_causal": True},
    ),
)
def test_unsupported_contract_never_reaches_a_backend(options):
    query, key, value = arrays(1, 7, query_heads=2, key_heads=2)
    called = []
    adapter = Adapter(
        "ntops", torch_stub(), implementation=lambda *a, **kw: called.append(True)
    )

    with pytest.raises(UnsupportedAttention):
        adapter(query, key, value, **options)

    assert not called


def test_same_head_hf_expansion_does_not_repeat_kv_again():
    query, key, value = arrays(query_heads=4, key_heads=4)
    observed = []
    adapter = Adapter(
        "ntops",
        torch_stub(),
        implementation=lambda q, k, v, **kw: observed.append((q, k, v)) or q,
    )
    adapter(query, key, value, is_causal=True)
    assert observed == [(query, key, value)]


def test_patch_restores_the_original_sdpa_on_failure():
    torch = torch_stub()
    original = torch.nn.functional.scaled_dot_product_attention
    adapter = Adapter("ntops", torch, implementation=lambda *a, **kw: None)

    with pytest.raises(RuntimeError, match="test failure"):
        with adapter.patched():
            assert torch.nn.functional.scaled_dot_product_attention is adapter
            raise RuntimeError("test failure")

    assert torch.nn.functional.scaled_dot_product_attention is original
