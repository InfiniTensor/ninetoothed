"""Inference-only Linear adapters using existing generated or baseline MM kernels.

Both implementations consume a transposed weight view directly. This avoids
copying model weights during each token's Linear calls. The ntops public NPU
wrapper is not used because it selects a handwritten matrix fallback.
"""

from __future__ import annotations

import collections
import contextlib
import importlib
import math
import types

_PROJECTIONS = frozenset(
    {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    }
)
_MISSING = object()


def projection_kind(module_name):
    """Classify a model projection without depending on a fixed layer count."""
    leaf = module_name.rsplit(".", 1)[-1]

    return leaf if leaf in _PROJECTIONS else "other_linear"


class Adapter:
    """Replace this model's bias-free Linear calls with the selected MM kernel.

    Create after the runner has configured torch_npu, source paths and any
    compiler compatibility shims. `recorder` may be the existing runner's
    Adapters object: its collect/calls/shapes/active_operation fields are used
    so the runner's real JIT-launch observer attributes every Linear dispatch.
    Without a recorder, set this instance's collect flag for local counters.
    """

    def __init__(self, backend, torch, *, recorder=None):
        if backend not in {"ntops", "triton"}:
            raise ValueError("Linear backend must be 'ntops' or 'triton'.")

        self.backend = backend
        self.torch = torch
        self.recorder = recorder
        self.collect = False
        self.calls = collections.Counter()
        self.module_calls = collections.Counter()
        self.shape_calls = collections.Counter()
        self.source_imports = {}
        self._patched = False
        self._module_inventory = []

        if backend == "ntops":
            module = importlib.import_module("ntops_lab.kernels.linear.mm")
            self.kernel = module.kernel
            self.source_imports[module.__name__] = module.__file__
        else:
            package = importlib.import_module("triton_ascend_ops")
            module = package.load_operator("mm")
            self.kernel = module.mm_kernel
            self.language = importlib.import_module("triton.language")
            self.source_imports[module.__name__] = module.__file__

        if recorder is not None:
            recorder.source_imports.update(self.source_imports)

    def provenance(self):
        return {
            "implementation_type": "ninetoothed_generated_mm"
            if self.backend == "ntops"
            else "handwritten_triton_ascend_mm_baseline",
            "source_imports": dict(self.source_imports),
            "dispatch": "Existing kernel called directly; public ntops NPU fallback is bypassed.",
            "weight_layout": "weight.transpose(0,1) view; no per-call weight contiguous/copy/cache.",
            "activation_layout": "Reshape leading input dimensions into M; materialization only if reshape requires it, inside timed public call.",
            "output_layout": "Restore original leading dimensions and output_features with a view.",
            "bias_contract": "Bias must be None; unsupported inputs raise instead of falling back.",
            "ntops_numeric_contract": "Original FP32 accumulator is explicitly cast to FP16 in the pinned ntops MM kernel, then stored in input dtype (including BF16). This rounding is retained.",
            "baseline_numeric_contract": "Original MM kernel uses FP32 accumulation with allow_tf32=False, then casts directly to output dtype.",
            "arithmetic_rewritten": False,
            "native_matmul_fallback": False,
            "patched_modules": list(self._module_inventory),
            "projection_calls": dict(self.calls),
            "module_calls": dict(self.module_calls),
            "shape_calls": dict(self.shape_calls),
        }

    def _record(self, name, input, weight):
        recorder = self.recorder
        collect = recorder.collect if recorder is not None else self.collect

        if not collect:
            return

        kind = projection_kind(name)
        rows = math.prod(input.shape[:-1])
        signature = (
            kind,
            rows,
            weight.shape[1],
            weight.shape[0],
            tuple(input.shape),
            str(input.dtype),
            tuple(input.stride()),
            tuple(weight.stride()),
        )
        self.calls[kind] += 1
        self.module_calls[name] += 1
        self.shape_calls[str(signature)] += 1

        if recorder is not None:
            recorder.calls["linear:" + kind] += 1
            recorder.shapes["linear_mm:" + str(signature)] += 1
            recorder.active_operation = "linear_mm"

    def linear(self, input, weight, bias=None, *, module_name="linear"):
        """Apply a bias-free Linear while preserving the pinned kernel arithmetic."""
        torch = self.torch

        if bias is not None:
            raise ValueError("The matched Linear adapter supports bias=None only.")

        if input.ndim < 1 or weight.ndim != 2 or input.shape[-1] != weight.shape[1]:
            raise ValueError("Linear requires input[...,K] and weight[N,K].")

        if (
            input.device.type != "npu"
            or weight.device != input.device
            or weight.dtype != input.dtype
            or input.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        ):
            raise ValueError(
                "Linear operands must share an NPU device and floating dtype."
            )

        rows = math.prod(input.shape[:-1])
        reduction, columns = weight.shape[1], weight.shape[0]

        if min(rows, reduction, columns) < 1:
            raise ValueError("The matched Linear adapter requires nonempty dimensions.")

        self._record(module_name, input, weight)

        with torch._C._DisableTorchDispatch():
            left = input.reshape(rows, reduction)
            right = weight.transpose(0, 1)
            output = torch.empty(
                (rows, columns), device=input.device, dtype=input.dtype
            )

            if self.backend == "ntops":
                self.kernel(left, right, output)
            else:
                grid = (((rows + 31) // 32) * ((columns + 31) // 32), 1)

                with torch.npu.device(input.device):
                    self.kernel[grid](
                        left,
                        right,
                        output,
                        rows,
                        columns,
                        reduction,
                        *left.stride(),
                        *right.stride(),
                        *output.stride(),
                        self.language.float32,
                        32,
                        32,
                        32,
                        1,
                        1,
                        reduction % 32 == 0,
                    )

            return output.view((*input.shape[:-1], columns))

    @contextlib.contextmanager
    def patched(self, model):
        """Patch model Linear instances and restore every original forward on exit."""
        if self._patched:
            raise RuntimeError("This Linear adapter is already patching a model.")

        modules = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, self.torch.nn.Linear)
        ]

        if not modules:
            raise ValueError("The model contains no torch.nn.Linear modules.")

        for _, module in modules:
            if module.bias is not None:
                raise ValueError(
                    "The matched Linear adapter requires a bias-free model."
                )

            if type(module).forward is not self.torch.nn.Linear.forward:
                raise ValueError(
                    "Custom Linear subclass forwards require an explicit adapter."
                )

            override = module.__dict__.get("forward", _MISSING)

            if (
                override is not _MISSING
                and getattr(override, "__func__", None)
                is not self.torch.nn.Linear.forward
            ):
                raise ValueError(
                    "Custom Linear instance forwards require an explicit adapter."
                )

        self._module_inventory = [
            {
                "name": name,
                "kind": projection_kind(name),
                "in_features": int(module.weight.shape[1]),
                "out_features": int(module.weight.shape[0]),
            }
            for name, module in modules
        ]
        originals = []
        self._patched = True

        def forward_for(name):
            def forward(module, input):
                return self.linear(input, module.weight, module.bias, module_name=name)

            return forward

        try:
            for name, module in modules:
                originals.append((module, module.__dict__.get("forward", _MISSING)))
                module.forward = types.MethodType(forward_for(name), module)

            yield self
        finally:
            for module, original in reversed(originals):
                if original is _MISSING:
                    del module.forward
                else:
                    module.forward = original

            self._patched = False
