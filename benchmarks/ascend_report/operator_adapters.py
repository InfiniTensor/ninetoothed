"""Explicit host adapters for the fixed ntops.lab / NineToothed source pair.

No numerical kernels are implemented here. Each compute dispatch calls an
existing NineToothed-generated kernel from ntops.lab. Repository NPU matmul
fallbacks are deliberately bypassed. Compiler shims are the runner's concern.

The callable returned by build_call performs layout work on every call, so
that public API timing includes its copies. A prepared-kernel measurement may
capture the resulting Triton launches after this work, but must separately
identify that scope. Returned tensors normally alias actual kernel outputs.
RoPE split-half packs its pair into the required full tensor inside the public
call. Its private metadata callbacks let the runner freeze the actual pair,
poison/replay those buffers, and assemble a full tensor outside device timing.
"""

from __future__ import annotations

import importlib
import math
from typing import Any, Callable


class UnsupportedCase(RuntimeError):
    """The requested contract cannot be supplied by an existing source kernel."""

    def __init__(
        self, reason: str, *, category: str = "unsupported_contract", metadata=None
    ):
        super().__init__(reason)
        self.reason = reason
        self.category = category
        self.metadata = dict(metadata or {})
        self.metadata.update(unsupported_reason=reason, unsupported_category=category)


_POINTWISE = frozenset(
    "abs add bitwise_and bitwise_not bitwise_or cos div eq exp ge gelu gt "
    "isinf isnan le lt mul ne neg pow relu rsqrt sigmoid silu sin sub tanh".split()
)
_BINARY = frozenset(
    "add sub mul div pow eq ne ge gt le lt bitwise_and bitwise_or".split()
)
_BOOLEAN = frozenset("eq ne ge gt le lt isinf isnan".split())
_REDUCTION_MODULE = {"sum": "sum", "mean": "mean", "amin": "min", "amax": "max"}
_ROOT = "ntops_lab.kernels."


def _module(path: str):
    return importlib.import_module(_ROOT + path)


def _metadata(case, mode):
    return {
        "operator": case["operator"],
        "case_id": case.get("id"),
        "mode": mode,
        "implementation_type": "ninetoothed_generated",
        "native_contract_match": False,
        "adaptations": [],
        "numeric_notes": [],
        "source_modules": [],
        "output_contract": "tensor",
        "reference_transform": "identity",
        "expected_triton_launches": 1,
        "output_replay_safe": True,
        "requires_compiler_shims": ["explicit_kernel_name_npu_launcher_alias"],
        "measurement_scope": {
            "public_api": "Includes allocation and all adapter layout copies on each call.",
            "prepared_device": "Existing generated Triton kernels after layout preparation; excludes adapter copies.",
        },
    }


def _adapt(meta, kind, detail):
    meta["adaptations"].append({"kind": kind, "detail": detail})


def _source(meta, path):
    meta["source_modules"].append(_ROOT + path)
    return _module(path)


def _reject(meta, reason, category="unsupported_contract"):
    raise UnsupportedCase(reason, category=category, metadata=meta)


def _require(meta, condition, reason, category="shape_contract"):
    if not condition:
        _reject(meta, reason, category)


def _check_keywords(meta, params, allowed):
    extra = set(params) - set(allowed)
    _require(
        meta,
        not extra,
        f"Unsupported parameters: {sorted(extra)}",
        "parameter_contract",
    )


def _pair(value):
    if isinstance(value, int):
        return (value, value)
    return tuple(value)


def _shape(x):
    return tuple(x.shape)


def _matches_fixed_pair(value, expected):
    return (type(value) is int and value == expected) or (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(type(item) is int and item == expected for item in value)
    )


def build_call(
    case: dict, prepared: dict, *, mode: str = "adapted"
) -> tuple[Callable, dict[str, Any]]:
    """Return (zero-argument callable, metadata).

    `prepared` follows baseline benchmarks/cases.py: args, parameters and
    reference. This function does not read or alter the reference. Validation
    and timing belong to the runner. Only explicit adapted mode is provided;
    pristine source API runs should remain a separate runner path. Remove the
    optional _capture_graph_buffers and _assemble_graph_result callbacks before
    serializing metadata; all other fields are JSON-serializable.
    """
    if mode != "adapted":
        raise ValueError("operator_adapters implements mode='adapted' only")

    meta = _metadata(case, mode)
    op = case["operator"]
    args = tuple(prepared["args"])
    params = dict(prepared.get("parameters", {}))
    common_contract = case.get("case_profile") == "common_contracts_v1"

    if common_contract:
        meta.update(
            case_profile="common_contracts_v1",
            original_case_id=case.get("original_id"),
        )

    # These reject before importing numerical dependencies or kernel modules.
    if op == "clamp" and not common_contract:
        _reject(
            meta,
            "Existing clamp is fixed to [-1,1]; CSV requires [-0.25,0.75]. No parameter-only adapter preserves the source algorithm.",
            "parameter_contract",
        )
    if op == "dropout" and not common_contract:
        _reject(
            meta,
            "Existing dropout is a rank-2 identity only, lacks RNG/seed/offset/returned mask and rejects stochastic training. It cannot supply the full CSV dropout contract.",
        )
    if op == "max_pool2d" and not common_contract:
        _reject(
            meta,
            "Existing max_pool2d returns values only and fixes a different window. Its pooling helper can match values, but no existing kernel provides the required indices; full CSV contract is unsupported.",
        )

    import torch

    if op == "clamp":
        _check_keywords(meta, params, {"min", "max"})
        _require(
            meta,
            type(params.get("min")) in (int, float)
            and type(params.get("max")) in (int, float)
            and params["min"] == -1
            and params["max"] == 1,
            "The common clamp contract requires min=-1 and max=1.",
            "parameter_contract",
        )
        _require(meta, len(args) == 1, "Clamp expects one tensor input.")
        x = args[0]
        mod = _source(meta, "pointwise.clamp")
        meta["native_contract_match"] = x.ndim == 1
        meta["numeric_scope"] = "clamp_fixed_minus_one_to_one"
        _adapt(
            meta,
            "parameter",
            "Validate min=-1 and max=1, matching the unchanged source constants.",
        )
        _adapt(
            meta,
            "layout",
            "Reshape input to rank one inside each public call, including any required materialization; restore the output shape with a view.",
        )

        def clamp_call():
            flat = x.reshape(-1)
            out = torch.empty(flat.shape, device=x.device, dtype=x.dtype)
            mod.kernel(flat, out)

            return out.reshape(_shape(x))

        return clamp_call, meta

    if op == "dropout":
        _check_keywords(meta, params, {"p", "training", "return_mask"})
        probability = params.get("p", 0.5)
        training = params.get("training", False)
        _require(
            meta,
            type(probability) in (int, float)
            and math.isfinite(probability)
            and 0 <= probability <= 1
            and type(training) is bool
            and params.get("return_mask", False) is False,
            "The common dropout contract requires p in [0,1], Boolean training, and return_mask=False.",
            "parameter_contract",
        )
        _require(
            meta,
            not training or probability == 0,
            "Existing dropout supports inference or p=0 identity only; stochastic training is unsupported.",
            "parameter_contract",
        )
        _require(
            meta,
            len(args) == 1 and args[0].ndim in (1, 2),
            "Dropout requires one rank-one or rank-two tensor.",
        )
        x = args[0]
        mod = _source(meta, "pointwise.dropout")
        meta["native_contract_match"] = x.ndim == 2
        meta["numeric_scope"] = (
            "deterministic_identity_p0" if training else "inference_identity"
        )
        meta["numeric_notes"].append(
            "Only the original deterministic identity kernel is measured; stochastic dropout, RNG state and returned masks are outside this profile."
        )
        _adapt(
            meta,
            "parameter",
            "Validate the values-only identity contract and call the existing dropout.run with inplace=False.",
        )
        _adapt(
            meta,
            "layout",
            "Reshape rank-one inputs to one row inside each public call and restore the original output shape; retain rank-two input strides.",
        )

        def dropout_call():
            rows = x.reshape(1, -1) if x.ndim == 1 else x
            out = mod.run(rows, p=probability, training=training, inplace=False)

            return out.reshape(_shape(x))

        return dropout_call, meta

    if op == "max_pool2d":
        _check_keywords(
            meta, params, {"kernel_size", "stride", "padding", "return_indices"}
        )
        _require(
            meta,
            _matches_fixed_pair(params.get("kernel_size"), 3)
            and _matches_fixed_pair(params.get("stride"), 2)
            and _matches_fixed_pair(params.get("padding"), 1)
            and params.get("return_indices", False) is False,
            "The common max_pool2d contract requires kernel_size=3, stride=2, padding=1, and return_indices=False.",
            "parameter_contract",
        )
        _require(
            meta,
            len(args) == 1 and args[0].ndim == 4,
            "MaxPool2d requires one rank-four NCHW tensor.",
        )
        x = args[0]
        mod = _source(meta, "layout.avg_pool2d")
        meta["numeric_scope"] = "max_pool2d_values_only"
        meta["numeric_notes"].append(
            "Retain the existing max-reduction application and negative-infinity padding; indices are neither produced nor fabricated."
        )
        _adapt(
            meta,
            "dispatch",
            "Call the existing pooling helper with kind='max', kernel_size=(3,3), stride=(2,2), and padding=(1,1).",
        )

        return (
            lambda: mod._pool2d(
                x, "max", kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            ),
            meta,
        )

    if op in _POINTWISE:
        _check_keywords(meta, params, ())
        count = 2 if op in _BINARY else 1
        _require(meta, len(args) == count, f"{op} expects {count} tensor inputs")
        mod = _source(meta, "pointwise." + op)
        result_dtype = torch.bool if op in _BOOLEAN else args[0].dtype
        meta["native_contract_match"] = all(x.ndim == 1 for x in args) and (
            count == 1 or _shape(args[0]) == _shape(args[1])
        )
        _adapt(
            meta,
            "layout",
            "Broadcast binary inputs, materialize contiguous rank-1 operands, and return an output view with the original/broadcast shape.",
        )
        _adapt(
            meta,
            "host_output_contract",
            "Allocate bool output for comparisons/isinf/isnan; otherwise preserve input dtype. Call the original kernel directly.",
        )
        if op == "gelu":
            meta["numeric_notes"].append(
                "Original tanh-approximate GELU retained; baseline reference is exact erf GELU."
            )
        if op == "pow":
            meta["numeric_notes"].append(
                "Original exp(y*log(x)) retained; CSV positive-base domain is required."
            )

        def pointwise_call():
            values = torch.broadcast_tensors(*args) if count == 2 else args
            shape = _shape(values[0])
            flat = tuple(x.contiguous().view(-1) for x in values)
            out = torch.empty(shape, dtype=result_dtype, device=args[0].device)
            mod.kernel(*flat, out.view(-1))
            return out

        return pointwise_call, meta

    if op in {"rms_norm", "layer_norm"}:
        _check_keywords(meta, params, {"eps"})
        _require(
            meta,
            params.get("eps", 1e-5) == 1e-5,
            "Source epsilon is fixed to 1e-5",
            "parameter_contract",
        )
        if op == "rms_norm":
            _require(meta, len(args) == 2, "RMSNorm requires x,weight")
            x, weight = args
            native_args = args
        else:
            _require(
                meta,
                len(args) == 4,
                "Baseline LayerNorm arguments must be x,normalized_shape,weight,bias",
            )
            x, normalized_shape, weight, bias = args
            _require(
                meta,
                tuple(normalized_shape) == (_shape(x)[-1],),
                "Only normalization over the final dimension is implemented",
            )
            _require(
                meta,
                bias is not None and _shape(bias) == (_shape(x)[-1],),
                "LayerNorm requires a bias vector matching hidden width",
            )
            native_args = (x, weight, bias)
            _adapt(
                meta,
                "parameter",
                "Validate normalized_shape and omit it from the existing run(x,weight,bias) signature.",
            )
        _require(
            meta,
            x.ndim == 2 and weight is not None and _shape(weight) == (x.shape[-1],),
            "Normalization requires rank-2 x and matching rank-1 weight",
        )
        mod = _source(meta, "normalization." + op)
        _adapt(
            meta,
            "parameter",
            "Validate eps=1e-5 and omit keyword because source hardcodes that value.",
        )
        meta["requires_compiler_shims"].append(
            "parameter_tensor_to_masked_load_lowering"
        )
        meta["numeric_notes"].append(
            "Retain original FP32 normalization and weight arithmetic; no new intermediate dtype rounding."
        )
        if op == "rms_norm":
            meta["numeric_notes"].append(
                "Baseline rounds normalized values to input dtype before weight multiplication; this source does not."
            )
        return lambda: mod.run(*native_args), meta

    if op == "softmax":
        _check_keywords(meta, params, {"dim"})
        _require(
            meta,
            len(args) == 1 and args[0].ndim == 2,
            "Softmax source requires rank-2 input",
        )
        x = args[0]
        dim = params.get("dim", -1)
        _require(
            meta,
            dim in (-2, -1, 0, 1),
            "Only dimensions of rank-2 input are supported",
            "parameter_contract",
        )
        axis = dim % 2
        mod = _source(meta, "reduction.softmax")
        _adapt(
            meta,
            "parameter",
            "Validate dim and dispatch the existing row-softmax without a dim keyword.",
        )
        if axis == 0:
            _adapt(
                meta,
                "layout",
                "Transpose input/output as views to reduce original axis zero.",
            )
        meta["requires_compiler_shims"].append(
            "parameter_tensor_to_masked_load_lowering"
        )

        def softmax_call():
            if axis == 0:
                return mod.run(x.transpose(0, 1)).transpose(0, 1)
            return mod.run(x)

        return softmax_call, meta

    if op == "reduction":
        _check_keywords(meta, params, {"kind", "dim", "keepdim"})
        kind = params.get("kind")
        if kind not in _REDUCTION_MODULE:
            _reject(
                meta,
                f"No existing row-reduction kernel for kind={kind!r}; prod is absent from this source.",
                "missing_implementation",
            )
        _require(
            meta,
            len(args) == 1 and args[0].ndim in (1, 2),
            "Reduction adapter supports rank-1 or rank-2 inputs",
        )
        x = args[0]
        dim, keepdim = params.get("dim", -1), params.get("keepdim", False)
        if dim is None:
            axis = None
            output_shape = (1,) * x.ndim if keepdim else ()
        else:
            _require(
                meta,
                isinstance(dim, int) and -x.ndim <= dim < x.ndim,
                "Reduction dim is outside the input rank",
                "parameter_contract",
            )
            axis = dim % x.ndim
            output_shape = tuple(
                1 if keepdim and i == axis else size
                for i, size in enumerate(x.shape)
                if keepdim or i != axis
            )
        mod = _source(meta, "reduction." + _REDUCTION_MODULE[kind])
        _adapt(
            meta,
            "dispatch",
            f"Map baseline kind={kind} to existing row kernel {_REDUCTION_MODULE[kind]}; do not use tiled amax.py.",
        )
        _adapt(
            meta,
            "layout",
            "Use transpose/reshape input views and reshape output for dim/keepdim; any necessary reshape materialization occurs inside the public callable.",
        )
        if kind in ("amin", "amax"):
            meta["numeric_notes"].append(
                "Original min/max masked tail fill is retained; Ascend other=None may become non-neutral zero. Reference validation is mandatory."
            )
        if dim is None:
            meta["numeric_notes"].append(
                "Single whole-row reduction retained, including its resource limits; no multi-stage reduction is introduced."
            )

        def reduction_call():
            if axis is None or x.ndim == 1:
                rows = x.reshape(1, -1)
            elif axis == 0:
                rows = x.transpose(0, 1)
            else:
                rows = x
            return mod.run(rows).reshape(output_shape)

        return reduction_call, meta

    if op in {"mm", "addmm", "bmm"}:
        _check_keywords(meta, params, {"alpha", "beta"} if op == "addmm" else ())
        if op == "addmm":
            _require(
                meta,
                params.get("alpha", 1) == 1 and params.get("beta", 1) == 1,
                "Existing addmm implements alpha=beta=1 only; non-default scaling would change the computation.",
                "parameter_contract",
            )
            _require(meta, len(args) == 3, "addmm expects bias,a,b")
            bias, a, b = args
        else:
            _require(meta, len(args) == 2, f"{op} expects a,b")
            a, b = args
        rank = 3 if op == "bmm" else 2
        _require(
            meta,
            a.ndim == b.ndim == rank and a.shape[-1] == b.shape[-2],
            "Matrix ranks/inner dimensions do not match",
        )
        if op == "bmm":
            _require(meta, a.shape[0] == b.shape[0], "bmm batch counts must match")
            output_shape = (a.shape[0], a.shape[1], b.shape[2])
            meta["expected_triton_launches"] = int(a.shape[0])
        else:
            output_shape = (a.shape[0], b.shape[1])
        if op == "addmm":
            _require(
                meta,
                _shape(bias) in {(b.shape[1],), output_shape},
                "addmm adapter supports a bias vector or full bias matrix only",
            )
            _adapt(
                meta,
                "layout",
                "Expand rank-1 bias to rank-2 with a zero-copy stride-zero view.",
            )
        mod = _source(meta, "linear." + op)
        _adapt(
            meta,
            "dispatch",
            "Call the original NineToothed module.kernel, bypassing public NPU handwritten Triton fallback.",
        )
        meta["numeric_notes"].append(
            "Original FP32 accumulation followed by explicit FP16 intermediate output cast is unchanged for every dtype."
        )
        if op == "bmm":
            _adapt(
                meta,
                "dispatch",
                "Repeat the existing rank-2 kernel over batches, exactly as source non-NPU path; each out[i] is a view of final output.",
            )

        def matmul_call():
            out = torch.empty(output_shape, device=a.device, dtype=a.dtype)
            if op == "bmm":
                for i in range(a.shape[0]):
                    mod.kernel(a[i], b[i], out[i])
            elif op == "addmm":
                mod.kernel(bias.expand(output_shape), a, b, out)
            else:
                mod.kernel(a, b, out)
            return out

        return matmul_call, meta

    if op == "conv2d":
        _check_keywords(meta, params, {"stride", "padding", "dilation", "groups"})
        _require(
            meta,
            _pair(params.get("stride", 1)) == (1, 1)
            and _pair(params.get("padding", 0)) == (1, 1)
            and _pair(params.get("dilation", 1)) == (1, 1)
            and params.get("groups", 1) == 1,
            "Existing conv2d public contract is stride1,padding1,dilation1,groups1; grouped/dilated case is unsupported.",
            "parameter_contract",
        )
        _require(meta, len(args) == 3, "conv2d expects x,weight,bias_or_None")
        x, weight, bias = args
        _require(
            meta,
            x.ndim == weight.ndim == 4 and x.shape[1] == weight.shape[1],
            "conv2d requires NCHW and OIHW with matching input channels",
        )
        _require(
            meta,
            bias is None or _shape(bias) == (weight.shape[0],),
            "conv2d bias must match output channels",
        )
        mod = _source(meta, "layout.conv2d")
        _adapt(
            meta,
            "parameter",
            "Validate source's fixed stride/padding/dilation/groups and call existing run(x,weight,bias) without kwargs.",
        )
        if bias is None:
            meta["numeric_notes"].append(
                "Original source allocates a zero bias for None; allocation/fill remains in public API timing."
            )
        return lambda: mod.run(*args), meta

    if op == "scaled_dot_product_attention":
        _check_keywords(
            meta, params, {"attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa"}
        )
        _require(
            meta,
            params.get("attn_mask") is None and params.get("dropout_p", 0) == 0,
            "Existing SDPA has no attention-mask or dropout implementation",
            "parameter_contract",
        )
        _require(meta, len(args) == 3, "SDPA expects q,k,v")
        q, k, v = args
        _require(meta, q.ndim == k.ndim == v.ndim == 4, "SDPA requires rank-4 tensors")
        _require(
            meta,
            q.shape[0] == k.shape[0] == v.shape[0]
            and k.shape[1:3] == v.shape[1:3]
            and q.shape[3] == k.shape[3] == v.shape[3]
            and q.shape[3] <= 128,
            "SDPA batch/KV/head-width contract is incompatible with source",
        )
        gqa = bool(params.get("enable_gqa", False))
        _require(
            meta,
            k.shape[1] > 0
            and (q.shape[1] % k.shape[1] == 0 if gqa else q.shape[1] == k.shape[1]),
            "SDPA requires equal heads or an integral GQA repetition factor",
        )
        mod = _source(meta, "fused.general.scaled_dot_product_attention")
        _adapt(
            meta,
            "parameter",
            "Always pass scale=None when absent so source applies D**-0.5 instead of its Python default1.0.",
        )
        if gqa:
            _adapt(
                meta,
                "layout",
                "Repeat-interleave KV heads inside each public call and use original equal-head kernel with enable_gqa=False.",
            )
        meta["numeric_notes"].append(
            "Original Ascend flash-attention tiles and numerical path remain unchanged."
        )

        def attention_call():
            kk, vv = k, v
            if gqa:
                repeats = q.shape[1] // k.shape[1]
                kk = k.repeat_interleave(repeats, dim=1)
                vv = v.repeat_interleave(repeats, dim=1)
            return mod.run(
                q,
                kk,
                vv,
                is_causal=params.get("is_causal", False),
                scale=params.get("scale"),
                enable_gqa=False,
            )

        return attention_call, meta

    if op == "swiglu":
        _check_keywords(meta, params, ())
        _require(
            meta,
            len(args) == 1 and args[0].ndim == 2 and args[0].shape[-1] % 2 == 0,
            "SwiGLU adapter requires rank-2 packed gate/up with even last dimension",
        )
        x = args[0]
        output_shape = (x.shape[0], x.shape[1] // 2)
        mod = _source(meta, "fused.general.swiglu")
        _adapt(
            meta,
            "layout",
            "Split last dimension into gate/up; materialize flattened operands inside public call; write original kernel into flattened view of final output.",
        )

        def swiglu_call():
            gate, up = x.chunk(2, dim=-1)
            gate = gate.contiguous().view(-1)
            up = up.contiguous().view(-1)
            out = torch.empty(output_shape, device=x.device, dtype=x.dtype)
            mod.kernel(gate, up, out.view(-1))
            return out

        return swiglu_call, meta

    if op == "rotary_position_embedding":
        _check_keywords(meta, params, {"interleaved"})
        _require(meta, len(args) == 3, "RoPE adapter requires x,cos,sin")
        x, cosine, sine = args
        _require(
            meta,
            x.ndim == 3 and x.shape[-1] % 2 == 0,
            "RoPE requires [tokens,heads,even_dim]",
        )
        tokens, heads, dim = x.shape
        half = dim // 2
        _require(
            meta,
            _shape(cosine) == _shape(sine) == (tokens, half),
            "RoPE cos/sin must have shape(tokens,dim//2)",
        )
        interleaved = bool(params.get("interleaved", False))
        mod = _source(meta, "fused.general.rotary_embedding")
        _adapt(
            meta,
            "dispatch",
            "Use existing rotary_embedding pair kernel; CSV operator name is rotary_position_embedding.",
        )
        _adapt(
            meta,
            "layout",
            "Split x into the requested component pairs, expand cos/sin over heads, and materialize rank-1 input operands inside public call.",
        )
        if interleaved:
            _adapt(
                meta,
                "output_contract",
                "Original rank-1 kernel writes directly into final output flat even/odd views with stride2; no packing copy.",
            )
        else:
            meta["output_replay_safe"] = False
            meta["graph_validation_protocol"] = (
                "freeze_kernel_pair_then_assemble_full_tensor_outside_device_timing"
            )
            _adapt(
                meta,
                "output_contract",
                "Concatenate the original pair outputs into the full required tensor inside public API timing.",
            )
            _adapt(
                meta,
                "validation",
                "Freeze the actual kernel pair immediately after captured call; poison/replay those buffers and concatenate outside device timing for unchanged full-tensor reference verification.",
            )
            meta["measurement_scope"]["public_api"] += (
                " Includes concatenating the two RoPE halves into the full output."
            )

        last_kernel_buffers = None

        def capture_graph_buffers():
            if last_kernel_buffers is None:
                raise RuntimeError(
                    "Call the split RoPE public callable before capturing its buffers"
                )
            return tuple(last_kernel_buffers)

        def assemble_graph_result(buffers):
            if len(buffers) != 2:
                raise ValueError(
                    "Split RoPE graph validation requires its two kernel output buffers"
                )
            return torch.cat(tuple(buffers), dim=-1)

        if not interleaved:
            meta["_capture_graph_buffers"] = capture_graph_buffers
            meta["_assemble_graph_result"] = assemble_graph_result

        def rotary_call():
            nonlocal last_kernel_buffers
            if interleaved:
                first, second = x[..., 0::2], x[..., 1::2]
            else:
                first, second = x.chunk(2, dim=-1)
            first = first.contiguous().view(-1)
            second = second.contiguous().view(-1)
            c = cosine[:, None, :].expand(tokens, heads, half).contiguous().view(-1)
            s = sine[:, None, :].expand(tokens, heads, half).contiguous().view(-1)
            if interleaved:
                out = torch.empty(_shape(x), device=x.device, dtype=x.dtype)
                flat = out.view(-1)
                mod.kernel(first, second, c, s, flat[0::2], flat[1::2])
                return out
            left = torch.empty((tokens, heads, half), device=x.device, dtype=x.dtype)
            right = torch.empty_like(left)
            mod.kernel(first, second, c, s, left.view(-1), right.view(-1))
            last_kernel_buffers = (left, right)
            return assemble_graph_result(last_kernel_buffers)

        return rotary_call, meta

    _reject(meta, f"No audited adapter for operator {op!r}", "missing_adapter")
