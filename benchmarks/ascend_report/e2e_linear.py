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
import time
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

    def __init__(
        self,
        backend,
        torch,
        *,
        recorder=None,
        tuning="auto",
        block_sizes=(32, 32, 32),
        weight_layout="view",
    ):
        if backend not in {"ntops", "triton"}:
            raise ValueError("Linear backend must be 'ntops' or 'triton'.")

        if tuning not in {"auto", "fixed"}:
            raise ValueError("Linear tuning must be 'auto' or 'fixed'.")

        if weight_layout not in {"view", "packed"}:
            raise ValueError("Linear weight layout must be 'view' or 'packed'.")

        block_sizes = tuple(block_sizes)

        if len(block_sizes) != 3 or any(
            type(size) is not int or size < 16 or size & (size - 1)
            for size in block_sizes
        ):
            raise ValueError(
                "Linear block sizes must be three powers of two of at least16."
            )

        self.backend = backend
        self.tuning = tuning
        self.block_sizes = block_sizes
        self.weight_layout = weight_layout
        self.torch = torch
        self.recorder = recorder
        self.collect = False
        self.calls = collections.Counter()
        self.module_calls = collections.Counter()
        self.shape_calls = collections.Counter()
        self.source_imports = {}
        self._patched = False
        self._module_inventory = []
        self._layout_statistics = {
            "mode": weight_layout,
            "packing": None,
            "restoration": None,
            "old_tensor_or_storage_snapshots_retained": False,
            "parameter_identity_preserved": True,
            "model_weight_files_modified": False,
        }

        if backend == "ntops":
            module = importlib.import_module("ntops_lab.kernels.linear.mm")
            if tuning == "fixed":
                ninetoothed = importlib.import_module("ninetoothed")
                original = module.arrangement
                globals_ = dict(original.__globals__)
                globals_.update(zip(("BM", "BN", "BK"), block_sizes))
                arrangement = types.FunctionType(
                    original.__code__,
                    globals_,
                    name=original.__name__,
                    argdefs=original.__defaults__,
                    closure=original.__closure__,
                )
                self.kernel_name = "ntops_lab_mm_fixed_" + "_".join(
                    map(str, block_sizes)
                )
                self.kernel = ninetoothed.make(
                    arrangement,
                    module.application,
                    tuple(ninetoothed.Tensor(2) for _ in range(3)),
                    kernel_name=self.kernel_name,
                    num_warps=8,
                    num_stages=1,
                )
            else:
                self.kernel_name = "ntops_lab_mm"
                self.kernel = module.kernel
            self.source_imports[module.__name__] = module.__file__
        else:
            package = importlib.import_module("triton_ascend_ops")
            module = package.load_operator("mm")
            self.kernel = module.mm_kernel
            self.language = importlib.import_module("triton.language")
            self.kernel_name = "mm_kernel"
            self.source_imports[module.__name__] = module.__file__

        if recorder is not None:
            recorder.source_imports.update(self.source_imports)

    def provenance(self):
        return {
            "implementation_type": "ninetoothed_generated_mm"
            if self.backend == "ntops"
            else "handwritten_triton_ascend_mm_baseline",
            "source_imports": dict(self.source_imports),
            "dispatch": "Original ntops arrangement/application generated with fixed blocks; public NPU fallback is bypassed."
            if self.backend == "ntops" and self.tuning == "fixed"
            else "Existing kernel called directly; public ntops NPU fallback is bypassed.",
            "weight_layout": (
                "Before timing, each unique Parameter is packed once as W.T.contiguous().T; calls use a contiguous right operand view; original row-major layout is restored on context exit."
                if self.weight_layout == "packed"
                else "weight.transpose(0,1) view; no per-call weight contiguous/copy/cache."
            ),
            "weight_preparation": self._layout_statistics,
            "activation_layout": "Reshape leading input dimensions into M; materialization only if reshape requires it, inside timed public call.",
            "output_layout": "Restore original leading dimensions and output_features with a view.",
            "bias_contract": "Bias must be None; unsupported inputs raise instead of falling back.",
            "ntops_numeric_contract": "Original FP32 accumulator is explicitly cast to FP16 in the pinned ntops MM kernel, then stored in input dtype (including BF16). This rounding is retained.",
            "baseline_numeric_contract": "Original MM kernel uses FP32 accumulation with allow_tf32=False, then casts directly to output dtype.",
            "arithmetic_rewritten": False,
            "native_matmul_fallback": False,
            "tuning": {
                "requested_mode": self.tuning,
                "autotune_enabled": self.backend == "ntops" and self.tuning == "auto",
                "block_sizes": list(self.block_sizes)
                if self.backend == "ntops" and self.tuning == "fixed"
                else [32, 32, 32]
                if self.backend == "triton"
                else None,
                "num_warps": 8
                if self.backend == "ntops" and self.tuning == "fixed"
                else None,
                "num_stages": 1
                if self.backend == "ntops" and self.tuning == "fixed"
                else None,
                "kernel_name": self.kernel_name + "_npu"
                if self.backend == "ntops"
                else self.kernel_name,
                "application": "Unmodified ntops.lab MM application, including FP16 intermediate cast.",
            },
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

    @staticmethod
    def _storage_key(tensor):
        return str(tensor.device), tensor.untyped_storage().data_ptr()

    def _packing_plan(self, model, modules):
        """Inspect registered aliases without retaining any old storage objects."""
        parameters = {}
        selected_names = set()
        storage_owners = {}
        device = None

        for name, module in modules:
            parameter = module.weight
            qualified = name + ".weight" if name else "weight"
            selected_names.add(qualified)

            if not isinstance(parameter, self.torch.nn.Parameter):
                raise ValueError("Packed Linear weights must be Parameter objects.")

            if parameter.ndim != 2 or min(parameter.shape) < 1:
                raise ValueError(
                    "Packed Linear weights must be nonempty rank-two matrices."
                )

            if parameter.device.type != "npu" or parameter.grad is not None:
                raise ValueError(
                    "Weight packing requires NPU inference parameters without gradients."
                )

            if device is None:
                device = parameter.device
            elif parameter.device != device:
                raise ValueError("Pack all Linear weights on one NPU device.")

            shape = tuple(parameter.shape)
            stride = tuple(parameter.stride())
            size = parameter.numel() * parameter.element_size()

            if (
                not parameter.is_contiguous()
                or stride != (shape[1], 1)
                or parameter.storage_offset() != 0
                or parameter.untyped_storage().nbytes() != size
            ):
                raise ValueError(
                    "Packing requires independent canonical contiguous weights, not noncontiguous or partial-storage views."
                )

            identity = id(parameter)
            storage = self._storage_key(parameter)

            if storage in storage_owners and storage_owners[storage] != identity:
                raise ValueError(
                    "Distinct Parameters sharing weight storage cannot be packed safely."
                )

            storage_owners[storage] = identity

            if identity not in parameters:
                parameters[identity] = {
                    "parameter": parameter,
                    "shape": shape,
                    "stride": stride,
                    "dtype": parameter.dtype,
                    "requires_grad": parameter.requires_grad,
                    "bytes": size,
                    "names": [],
                    "needs_copy": not parameter.transpose(0, 1).is_contiguous(),
                }

            parameters[identity]["names"].append(qualified)

        for name, parameter in model.named_parameters(remove_duplicate=False):
            if parameter.numel() == 0:
                continue

            owner = storage_owners.get(self._storage_key(parameter))

            if owner is not None and (
                id(parameter) != owner or name not in selected_names
            ):
                raise ValueError(
                    "A Linear weight aliases another registered parameter or non-Linear module."
                )

        for _, buffer in model.named_buffers(remove_duplicate=False):
            if buffer.numel() and self._storage_key(buffer) in storage_owners:
                raise ValueError("A Linear weight aliases a registered buffer.")

        self._layout_statistics.update(
            unique_parameter_count=len(parameters),
            selected_module_count=len(modules),
            shared_parameter_module_count=len(modules) - len(parameters),
            copied_parameter_count=sum(
                item["needs_copy"] for item in parameters.values()
            ),
            registered_aliases_checked=True,
            external_alias_contract="The caller must not retain detached weight/storage views outside the model during packing.",
        )
        return list(parameters.values()), device

    def _memory_statistics(self, device):
        values = {}

        for name in ("memory_allocated", "max_memory_allocated", "memory_reserved"):
            function = getattr(self.torch.npu, name, None)

            try:
                values[name] = int(function(device)) if function is not None else None
            except (RuntimeError, TypeError, AttributeError):
                values[name] = None

        return values

    def _change_layout(self, entries, device, changed, *, restore=False):
        """Copy one matrix at a time; keep only Parameter references and metadata."""
        torch = self.torch
        torch.npu.synchronize(device)
        before = self._memory_statistics(device)
        started = time.perf_counter()
        statistics_ = {
            "completed": False,
            "parameters_processed": 0,
            "copied_bytes": 0,
            "minimum_read_plus_write_bytes": 0,
            "maximum_single_matrix_temporary_bytes": 0,
            "device_copy_ms_sum": 0.0,
            "wall_ms": None,
            "memory_before": before,
            "memory_after": None,
            "maximum_observed_allocated_bytes": before["memory_allocated"],
            "peak_memory_scope": "Allocator process high-water mark; this adapter does not reset it.",
            "numeric_transformation": "None; transpose/contiguous/transpose or contiguous only.",
        }
        self._layout_statistics["restoration" if restore else "packing"] = statistics_
        begin = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        try:
            with (
                torch.inference_mode(False),
                torch.no_grad(),
                torch._C._DisableTorchDispatch(),
            ):
                for item in entries:
                    if not item["needs_copy"]:
                        continue

                    parameter = item["parameter"]

                    if (
                        tuple(parameter.shape) != item["shape"]
                        or parameter.dtype != item["dtype"]
                    ):
                        raise RuntimeError(
                            "A packed parameter changed shape or dtype before restoration."
                        )

                    begin.record()
                    replacement = (
                        parameter.contiguous()
                        if restore
                        else parameter.transpose(0, 1).contiguous().transpose(0, 1)
                    )
                    allocated = self._memory_statistics(device)["memory_allocated"]

                    if allocated is not None:
                        statistics_["maximum_observed_allocated_bytes"] = max(
                            statistics_["maximum_observed_allocated_bytes"] or 0,
                            allocated,
                        )

                    parameter.data = replacement

                    if not restore:
                        changed.append(item)

                    del replacement
                    end.record()
                    end.synchronize()
                    statistics_["device_copy_ms_sum"] += float(begin.elapsed_time(end))
                    statistics_["parameters_processed"] += 1
                    statistics_["copied_bytes"] += item["bytes"]
                    statistics_["minimum_read_plus_write_bytes"] += 2 * item["bytes"]
                    statistics_["maximum_single_matrix_temporary_bytes"] = max(
                        statistics_["maximum_single_matrix_temporary_bytes"],
                        item["bytes"],
                    )

                    if parameter.requires_grad != item["requires_grad"]:
                        raise RuntimeError(
                            "Weight packing changed Parameter gradient metadata."
                        )

                    if restore and tuple(parameter.stride()) != item["stride"]:
                        raise RuntimeError(
                            "Weight restoration did not recover its original row-major strides."
                        )

                    if not restore and not parameter.transpose(0, 1).is_contiguous():
                        raise RuntimeError(
                            "Weight packing did not create a contiguous right operand."
                        )

            statistics_["completed"] = True
        finally:
            statistics_["wall_ms"] = (time.perf_counter() - started) * 1000
            statistics_["memory_after"] = self._memory_statistics(device)

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
        changed = []
        packing_entries, packing_device = (
            self._packing_plan(model, modules)
            if self.weight_layout == "packed"
            else ([], None)
        )
        self._patched = True

        def forward_for(name):
            def forward(module, input):
                return self.linear(input, module.weight, module.bias, module_name=name)

            return forward

        try:
            if self.weight_layout == "packed":
                self._change_layout(packing_entries, packing_device, changed)

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

            try:
                if changed:
                    self._change_layout(
                        list(reversed(changed)), packing_device, [], restore=True
                    )
            finally:
                self._patched = False
