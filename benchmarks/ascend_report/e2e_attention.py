"""Replace the HF SDPA call with pinned kernels while preserving attention semantics."""

import collections
import contextlib
import importlib
import itertools
import math
import numbers
import weakref
from dataclasses import asdict, dataclass


class UnsupportedAttention(ValueError):
    """The pinned kernels cannot express the requested attention contract."""


@dataclass(frozen=True)
class MaskData:
    shape: tuple
    values: object
    boolean: bool


@dataclass(frozen=True)
class AttentionPlan:
    query_shape: tuple
    key_shape: tuple
    value_shape: tuple
    input_is_causal: bool
    input_enable_gqa: bool
    is_causal: bool
    scale: float
    repeat_kv: int
    mask_kind: str


def _flatten(values):
    if isinstance(values, (list, tuple)):
        for value in values:
            yield from _flatten(value)
    else:
        yield values


def classify_mask(mask, batch, heads, query_length, key_length):
    """Recognize only exact all-visible or ordinary lower-triangular masks."""
    if mask is None:
        return "none"

    if len(mask.shape) > 4:
        raise UnsupportedAttention("Attention masks may have at most four dimensions.")

    shape = (1,) * (4 - len(mask.shape)) + tuple(mask.shape)
    target = (batch, heads, query_length, key_length)

    if any(size not in (1, extent) for size, extent in zip(shape, target)):
        raise UnsupportedAttention(
            "The attention mask does not broadcast to [B,H,Q,K]."
        )

    values = list(_flatten(mask.values))

    if len(values) != math.prod(shape):
        raise UnsupportedAttention("Attention mask shape and values disagree.")

    allowed = []

    for value in values:
        if mask.boolean:
            if not isinstance(value, bool):
                raise UnsupportedAttention(
                    "A Boolean attention mask must contain booleans."
                )

            allowed.append(value)
        elif isinstance(value, numbers.Real) and not isinstance(value, bool):
            if value == 0:
                allowed.append(True)
            elif math.isinf(value) and value < 0:
                allowed.append(False)
            else:
                raise UnsupportedAttention(
                    "Only zero/-infinity additive masks can be reduced exactly; finite biases are unsupported."
                )
        else:
            raise UnsupportedAttention(
                "Attention masks must be Boolean or floating point."
            )

    if all(allowed):
        return "all_visible"

    # Broadcast batch/head copies have identical masks and need only one inspection.
    inspected_shape = (shape[0], shape[1], query_length, key_length)

    for indices in itertools.product(*(range(size) for size in inspected_shape)):
        offset = 0

        for index, size in zip(indices, shape):
            offset = offset * size + (0 if size == 1 else index)

        visible = allowed[offset]

        if visible != (indices[3] <= indices[2]):
            raise UnsupportedAttention(
                "The pinned SDPA kernels cannot represent this padding, offset-causal or general mask."
            )

    return "lower_triangular"


def plan_attention(
    query_shape,
    key_shape,
    value_shape,
    *,
    mask=None,
    mask_kind=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
):
    """Validate a public SDPA request before choosing lossless host adaptations."""
    query_shape, key_shape, value_shape = map(
        tuple, (query_shape, key_shape, value_shape)
    )

    if any(
        len(shape) != 4 or any(type(size) is not int or size <= 0 for size in shape)
        for shape in (query_shape, key_shape, value_shape)
    ):
        raise UnsupportedAttention("SDPA requires nonempty rank-four [B,H,S,D] inputs.")

    batch, query_heads, query_length, width = query_shape
    key_batch, key_heads, key_length, key_width = key_shape

    if batch != key_batch or width != key_width or key_shape != value_shape:
        raise UnsupportedAttention(
            "SDPA requires matching batches, head widths and K/V shapes."
        )

    if width not in (64, 128):
        raise UnsupportedAttention(
            "The pinned SDPA kernels support head widths 64 and 128."
        )

    if type(is_causal) is not bool or type(enable_gqa) is not bool:
        raise UnsupportedAttention("is_causal and enable_gqa must be Boolean values.")

    if dropout_p != 0:
        raise UnsupportedAttention("The pinned SDPA kernels do not implement dropout.")

    if query_heads != key_heads and not enable_gqa:
        raise UnsupportedAttention(
            "Different query and KV head counts require enable_gqa=True."
        )

    if query_heads % key_heads:
        raise UnsupportedAttention(
            "The query head count must be divisible by the KV head count."
        )

    if scale is None:
        scale = width**-0.5
    elif not isinstance(scale, numbers.Real) or not math.isfinite(scale):
        raise UnsupportedAttention("SDPA scale must be a finite Python number or None.")

    kind = (
        mask_kind
        if mask_kind is not None
        else classify_mask(mask, batch, query_heads, query_length, key_length)
    )

    if kind not in ("none", "all_visible", "lower_triangular"):
        raise UnsupportedAttention("Unknown attention mask classification.")

    if kind != "none" and is_causal:
        raise UnsupportedAttention(
            "Explicit attention masks and is_causal=True cannot be combined."
        )

    causal = is_causal or kind == "lower_triangular"

    if causal and query_length != key_length:
        raise UnsupportedAttention(
            "Non-square causal attention is unsupported; cached single-token decode must retain is_causal=False."
        )

    return AttentionPlan(
        query_shape,
        key_shape,
        value_shape,
        is_causal,
        enable_gqa,
        causal,
        float(scale),
        query_heads // key_heads,
        kind,
    )


class Adapter:
    def __init__(self, backend, torch, recorder=None, *, implementation=None):
        if backend not in ("triton", "ntops"):
            raise ValueError("Attention backend must be 'triton' or 'ntops'.")

        self.backend = backend
        self.torch = torch
        self.recorder = recorder
        self.source_imports = {}
        self._mask_cache = {}
        self._mask_inspections = 0
        self._mask_cache_hits = 0
        self._kv_layout_normalizations = 0
        self._observations = collections.Counter()

        if implementation is not None:
            self._implementation = implementation
        elif backend == "ntops":
            module = importlib.import_module(
                "ntops_lab.kernels.fused.general.scaled_dot_product_attention"
            )
            self._implementation = module.run
            self.source_imports[module.__name__] = module.__file__
            kernel_module = importlib.import_module(
                "ntops_lab.kernels.fused.general._flash_attention"
            )
            self.source_imports[kernel_module.__name__] = kernel_module.__file__
        else:
            package = importlib.import_module("triton_ascend_ops")
            module = package.load_operator("scaled_dot_product_attention")
            self.source_imports[module.__name__] = module.__file__
            self._baseline_kernel = module._attn_fwd
            self._implementation = self._run_baseline_kernel

        if recorder is not None:
            recorder.source_imports.update(self.source_imports)

    def _run_baseline_kernel(self, query, key, value, *, is_causal, scale, enable_gqa):
        if enable_gqa or query.shape[1] != key.shape[1]:
            raise UnsupportedAttention(
                "The shared adapter must expand grouped KV heads before launch."
            )

        torch = self.torch
        batch, heads, query_length, width = query.shape
        key_length = key.shape[2]
        output = torch.empty_like(query)
        logsumexp = torch.empty(
            (batch, heads, query_length), dtype=torch.float32, device=query.device
        )

        with torch.npu.device(query.device):
            self._baseline_kernel[((query_length + 15) // 16, batch * heads, 1)](
                query,
                key,
                value,
                None,
                scale,
                logsumexp,
                output,
                *query.stride(),
                *key.stride(),
                *value.stride(),
                1,
                1,
                1,
                1,
                *output.stride(),
                batch,
                heads,
                heads,
                1,
                query_length,
                key_length,
                width,
                16,
                16,
                3 if is_causal else 1,
                False,
                False,
                sync_solver=True,
            )

        return output

    def _mask_kind(self, mask, query_shape, key_length):
        if mask is None:
            return "none"

        try:
            version = mask._version
        except (AttributeError, RuntimeError):
            version = None

        key = (
            version,
            tuple(mask.shape),
            str(mask.dtype),
            str(mask.device),
            tuple(query_shape),
            key_length,
        )
        entry = self._mask_cache.get(id(mask))

        if (
            version is not None
            and entry is not None
            and entry[0]() is mask
            and entry[1] == key
        ):
            self._mask_cache_hits += 1

            return entry[2]

        boolean = mask.dtype == self.torch.bool

        if (
            not boolean
            and not getattr(mask.dtype, "is_floating_point", False)
            and getattr(mask.dtype, "kind", None) != "f"
        ):
            raise UnsupportedAttention(
                "Attention mask dtype must be Boolean or floating point."
            )

        data = MaskData(tuple(mask.shape), mask.detach().cpu().tolist(), boolean)
        kind = classify_mask(
            data, query_shape[0], query_shape[1], query_shape[2], key_length
        )
        self._mask_inspections += 1

        if version is not None:
            identifier = id(mask)
            reference = weakref.ref(
                mask, lambda _: self._mask_cache.pop(identifier, None)
            )
            self._mask_cache[identifier] = (reference, key, kind)

        return kind

    def __call__(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        *,
        scale=None,
        enable_gqa=False,
    ):
        torch = self.torch

        if query.dtype not in (torch.float16, torch.bfloat16) or any(
            tensor.dtype != query.dtype or tensor.device != query.device
            for tensor in (key, value)
        ):
            raise UnsupportedAttention(
                "SDPA inputs must share a device and FP16/BF16 dtype."
            )

        if query.device.type != "npu":
            raise UnsupportedAttention("This benchmark adapter requires NPU tensors.")

        with torch._C._DisableTorchDispatch():
            if any(tensor.ndim != 4 for tensor in (query, key, value)):
                raise UnsupportedAttention("SDPA requires rank-four [B,H,S,D] inputs.")

            kind = self._mask_kind(attn_mask, tuple(query.shape), key.shape[2])
            plan = plan_attention(
                query.shape,
                key.shape,
                value.shape,
                mask_kind=kind,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )

            if self.recorder is not None and self.recorder.collect:
                self.recorder.active_operation = "sdpa"
                self.recorder.calls["sdpa"] += 1
                descriptor = str(asdict(plan))
                self.recorder.shapes["sdpa:" + descriptor] += 1
                self._observations[descriptor] += 1

            if plan.repeat_kv != 1:
                repeat = getattr(self, "_generated_gqa", None)

                if repeat is None:
                    key = key.repeat_interleave(plan.repeat_kv, dim=1)
                    value = value.repeat_interleave(plan.repeat_kv, dim=1)
                else:
                    if self.recorder is not None and self.recorder.collect:
                        self.recorder.active_operation = "gqa_head_expansion"
                        self.recorder.calls["gqa_head_expansion"] += 2
                    key = repeat(key, plan.repeat_kv)
                    value = repeat(value, plan.repeat_kv)
                    if self.recorder is not None and self.recorder.collect:
                        self.recorder.active_operation = "sdpa"

            # The pinned baseline uses the key's batch/head offset for both KV pointers.
            if key.stride()[:2] != value.stride()[:2]:
                key = key.contiguous()
                value = value.contiguous()
                self._kv_layout_normalizations += 1

            return self._implementation(
                query,
                key,
                value,
                is_causal=plan.is_causal,
                scale=plan.scale,
                enable_gqa=False,
            )

    @contextlib.contextmanager
    def patched(self):
        functional = self.torch.nn.functional
        original = functional.scaled_dot_product_attention
        functional.scaled_dot_product_attention = self

        try:
            yield
        finally:
            functional.scaled_dot_product_attention = original

    def provenance(self):
        return {
            "backend": self.backend,
            "patch_point": "torch.nn.functional.scaled_dot_product_attention",
            "source_imports": self.source_imports,
            "layout": "Preserve Q/K/V strides when KV batch/head strides match; otherwise make both KV tensors contiguous symmetrically.",
            "gqa": "Use the generated stage-seven repeat kernel when installed; otherwise repeat_interleave KV heads identically.",
            "mask": "Only exact all-visible or square lower-triangular masks are reduced; other masks raise UnsupportedAttention.",
            "causal": "Square prefill preserves causal=True; cached query-length-one decode preserves causal=False and all KV entries.",
            "scale": "Forward explicit finite scale; None resolves to head_dim**-0.5.",
            "baseline_host_adapter": "Call the pinned _attn_fwd kernel with real strides, without the public adapter's contiguous copies.",
            "mask_cache": "Weak references plus tensor version and shape; unversioned inference masks are checked on every call.",
            "mask_inspections": self._mask_inspections,
            "mask_cache_hits": self._mask_cache_hits,
            "kv_layout_normalizations": self._kv_layout_normalizations,
            "observed_calls": dict(self._observations),
            "numeric_fallback": False,
            "kernel_arithmetic_modified": False,
        }
