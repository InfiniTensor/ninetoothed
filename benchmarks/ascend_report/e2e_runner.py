"""Run one isolated Hugging Face generation benchmark from a JSON configuration."""

import argparse
import ast
import collections
import contextlib
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
import traceback
import unicodedata
from pathlib import Path

if __package__:
    from .affinity import apply_cpu_affinity
else:
    from affinity import apply_cpu_affinity

DEFAULT_PROMPT = (
    "Explain why the sky appears blue during the day and why sunsets often look red."
)
SCOPES = {
    1: "SiLU + RMSNorm",
    2: "SiLU + RMSNorm + MLP gating Mul",
    3: "SiLU + RMSNorm + MLP gating Mul + TensorAdd (RoPE and residual paths)",
    4: "SiLU + RMSNorm + MLP gating Mul + TensorAdd + Neg + Sin + Cos",
    5: "Stage 4 + scaled dot product attention (prefill and cached decode)",
    6: "Stage 5 + all model Linear projections and language model head",
}


def write_json(path, value):
    """Atomically save a JSON artifact without nonfinite number literals."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def read_config(path):
    """Resolve paths relative to the config file and validate generation settings."""
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    values = dict(raw.get("e2e", {}))
    paths = raw.get("paths", {})

    def resolve(value):
        candidate = Path(value).expanduser()

        return str(
            (path.parent / candidate).resolve()
            if not candidate.is_absolute()
            else candidate.resolve()
        )

    required = {
        "ntops": paths.get("ntops", raw.get("ntops_path")),
        "baseline": paths.get("baseline", raw.get("baseline_path")),
        "model": paths.get("model", values.get("model_path", raw.get("model_path"))),
    }

    for name, value in required.items():
        if not value:
            raise ValueError(f"Missing configured {name} path.")

    configured_paths = {name: resolve(value) for name, value in required.items()}
    configured_paths["ninetoothed"] = resolve(
        paths.get("ninetoothed", raw.get("ninetoothed_path", Path(__file__).parents[2]))
    )

    for name in ("output_dir", "cache_dir", "tmp_dir"):
        if not raw.get(name):
            raise ValueError(f"Missing configured {name}.")

    settings = {
        "prompt": DEFAULT_PROMPT,
        "input_tokens": 19,
        "output_tokens": 128,
        "dtype": "bfloat16",
        "batch": 1,
        "warmup": 2,
        "samples": 10,
        "logits_diagnostics": False,
        "worker_timeout_seconds": 3600,
        "seed": 20261020,
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "model_revision": None,
        "cpu_threads": 1,
        "stages": list(SCOPES),
        "linear_tuning": "fixed",
        "linear_block_sizes": [32, 32, 32],
        "linear_weight_layout": "packed",
    }
    settings.update(values)
    if settings["linear_tuning"] not in ("auto", "fixed"):
        raise ValueError("linear_tuning must be auto or fixed.")
    if settings["linear_weight_layout"] not in ("view", "packed"):
        raise ValueError("linear_weight_layout must be view or packed.")
    blocks = settings["linear_block_sizes"]
    if (
        not isinstance(blocks, (list, tuple))
        or len(blocks) != 3
        or any(
            type(size) is not int or size < 16 or size & (size - 1) for size in blocks
        )
    ):
        raise ValueError(
            "linear_block_sizes must be three powers of two of at least16."
        )

    if (
        not isinstance(settings["stages"], list)
        or not settings["stages"]
        or any(
            type(stage) is not int or stage not in SCOPES
            for stage in settings["stages"]
        )
        or len(set(settings["stages"])) != len(settings["stages"])
    ):
        raise ValueError("e2e.stages must contain distinct supported stage numbers.")

    for name, minimum in (
        ("input_tokens", 1),
        ("output_tokens", 1),
        ("warmup", 0),
        ("samples", 1),
        ("worker_timeout_seconds", 1),
        ("cpu_threads", 1),
    ):
        if not isinstance(settings[name], int) or settings[name] < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}.")

    if settings["dtype"] != "bfloat16" or settings["batch"] != 1:
        raise ValueError("This matched generation protocol requires BF16 and batch 1.")

    if not isinstance(settings["prompt"], str) or not settings["prompt"].strip():
        raise ValueError("A nonempty prompt is required.")

    fallback_device = values.get("device", raw.get("device", 0))
    global_devices = raw.get("devices", [raw.get("device", 0)])
    devices = values.get("devices", raw.get("devices", [fallback_device]))

    for name, selected in (("devices", global_devices), ("e2e.devices", devices)):
        if (
            not isinstance(selected, list)
            or not selected
            or any(type(device) is not int or device < 0 for device in selected)
            or len(set(selected)) != len(selected)
        ):
            raise ValueError(
                f"{name} must contain distinct nonnegative device integers."
            )

    settings["devices"] = devices
    settings["cpu_affinity"] = values.get(
        "cpu_affinity", raw.get("cpu_affinity", "auto")
    )

    configured_parallel = raw.get("scheduling", {}).get(
        "TRITON_ALL_BLOCKS_PARALLEL", "0"
    )

    if str(configured_parallel) != "0":
        raise ValueError("This report protocol requires TRITON_ALL_BLOCKS_PARALLEL=0.")

    result = {
        "config_file": str(path),
        "paths": configured_paths,
        "output_dir": resolve(raw["output_dir"]),
        "cache_dir": resolve(raw["cache_dir"]),
        "tmp_dir": resolve(raw["tmp_dir"]),
        "device": devices[0],
        "devices": global_devices,
        "cpu_affinity": raw.get("cpu_affinity", "auto"),
        "e2e": settings,
        "scheduling": {"TRITON_ALL_BLOCKS_PARALLEL": "0"},
    }
    result["e2e_output_dir"] = str(Path(result["output_dir"]) / "e2e")

    return result


def reference_path(config, device=None):
    device = config["device"] if device is None else device

    return Path(config["e2e_output_dir"]) / "references" / f"device-{device}.json"


def text_sanity(output_ids, text, expected_tokens):
    """Check mechanical output validity without pretending to judge its meaning."""
    checks = {
        "token_count_matches": len(output_ids) == expected_tokens,
        "nonempty_text": bool(text.strip()),
        "no_replacement_characters": "\ufffd" not in text,
        "no_disallowed_controls": not any(
            unicodedata.category(character) == "Cc" and character not in "\n\r\t"
            for character in text
        ),
    }

    return {"passed": all(checks.values()), **checks}


def validate_core_launches(backend, stage, calls, launches, linear_kernel_name=None):
    """Match logical core calls to actual launches of the pinned numeric kernels."""
    result = {}
    expected = []
    if stage >= 5:
        expected.append(
            (
                "sdpa",
                calls.get("sdpa", 0),
                "ntops_lab_flash_attention_ascend_npu"
                if backend == "ntops"
                else "_attn_fwd",
            )
        )
    if stage >= 6:
        expected.append(
            (
                "linear_mm",
                sum(
                    count for name, count in calls.items() if name.startswith("linear:")
                ),
                linear_kernel_name
                or ("ntops_lab_mm_npu" if backend == "ntops" else "mm_kernel"),
            )
        )
    for operation, logical_count, kernel_name in expected:
        selected = {
            name: count
            for name, count in launches.items()
            if name.split(":", 1)[0] == operation
        }
        if logical_count <= 0 or sum(selected.values()) != logical_count:
            raise RuntimeError(
                f"Core operator {operation} calls do not match its real JIT launches."
            )
        if any(name.rsplit(":", 1)[-1] != kernel_name for name in selected):
            raise RuntimeError(
                f"Core operator {operation} executed an unexpected kernel."
            )
        result[operation] = {
            "verified": True,
            "logical_calls": logical_count,
            "kernel_launches": sum(selected.values()),
            "kernel_name": kernel_name,
        }
    return result


def configure_runtime(config):
    """Set process-local paths and scheduling before importing accelerator packages."""
    device_cache = (
        Path(config["cache_dir"]) / f"device-{config['device']}" / "parallel0"
    )
    process_tmp = Path(config["tmp_dir"]) / f"device-{config['device']}"

    for directory in (device_cache, process_tmp, Path(config["e2e_output_dir"])):
        directory.mkdir(parents=True, exist_ok=True)

    config["actual_cpu_affinity"] = apply_cpu_affinity(
        config.get("assigned_cpu_affinity")
    )

    os.environ.update(
        ASCEND_RT_VISIBLE_DEVICES=str(config["device"]),
        TRITON_ALL_BLOCKS_PARALLEL="0",
        TRITON_CACHE_DIR=str(device_cache / "triton"),
        TMPDIR=str(process_tmp),
        TMP=str(process_tmp),
        TEMP=str(process_tmp),
        OMP_NUM_THREADS=str(config["e2e"]["cpu_threads"]),
        MKL_NUM_THREADS=str(config["e2e"]["cpu_threads"]),
        TOKENIZERS_PARALLELISM="false",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    sys.path[:0] = [
        str(Path(config["paths"]["ninetoothed"]) / "src"),
        str(Path(config["paths"]["ntops"]) / "src"),
        config["paths"]["baseline"],
    ]
    torch = importlib.import_module("torch")
    importlib.import_module("torch_npu")
    torch.set_num_threads(config["e2e"]["cpu_threads"])
    torch.set_num_interop_threads(config["e2e"]["cpu_threads"])
    torch.npu.set_device(0)
    torch.manual_seed(config["e2e"]["seed"])

    return torch, device_cache


def runtime_metadata(torch, config):
    versions = {}

    for name in (
        "torch",
        "torch_npu",
        "triton-ascend",
        "transformers",
        "numpy",
        "sympy",
    ):
        versions[name] = importlib.metadata.version(name)

    try:
        npu_smi = subprocess.check_output(["npu-smi", "info"], text=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as error:
        npu_smi = f"Unavailable: {error}"

    torch_npu = importlib.import_module("torch_npu")
    properties = torch.npu.get_device_properties(0)
    device_uuid = getattr(properties, "uuid", None)

    if device_uuid is None:
        match = re.search(r"\buuid=([^,\s)]+)", str(properties))
        device_uuid = match.group(1) if match else None

    if not device_uuid:
        raise RuntimeError("The NPU runtime did not expose a UUID for device pairing.")

    return {
        "python": sys.version,
        "pid": os.getpid(),
        "versions": versions,
        "physical_device": config["device"],
        "device_uuid": str(device_uuid),
        "device_name": torch.npu.get_device_name(0),
        "device_properties": str(properties),
        "cann": torch_npu.utils.get_cann_version(),
        "cpu_threads": torch.get_num_threads(),
        "cpu_interop_threads": torch.get_num_interop_threads(),
        "cpu_affinity": apply_cpu_affinity(None),
        "requested_cpu_affinity": config.get("assigned_cpu_affinity"),
        "runner_sha256": sha256_file(__file__),
        "npu_smi": npu_smi,
        "compiler_environment": {
            key: os.environ[key]
            for key in ("TRITON_ALL_BLOCKS_PARALLEL", "TRITON_CACHE_DIR", "TMPDIR")
        },
    }


class Adapters:
    def __init__(self, backend, torch, device_cache):
        self.backend = backend
        self.torch = torch
        self.collect = False
        self.active_operation = "unclassified"
        self.calls = collections.Counter()
        self.shapes = collections.Counter()
        self.native_matrix_calls = collections.Counter()
        self.aliases = []
        self.parameter_to_hits = 0
        self.source_imports = {}
        self.core_operator_adapters = {}
        operators = ("silu", "mul", "add", "neg", "sin", "cos")

        if backend == "ntops":
            generation = importlib.import_module("ninetoothed.generation")
            self.source_imports["ninetoothed.generation"] = generation.__file__
            generation.CACHE_DIR = device_cache / "generated"
            generation.CACHE_DIR.mkdir(parents=True, exist_ok=True)
            jit = importlib.import_module("ninetoothed.jit")
            tensor_type = importlib.import_module("ninetoothed.tensor").Tensor
            original_import = jit.import_from_path
            original_attribute = generation.CodeGenerator.visit_Attribute

            def import_alias(name, path):
                module = original_import(name, path)

                if getattr(module, "_IS_NPU", False):
                    for key, value in list(vars(module).items()):
                        if (
                            key.startswith("launch_")
                            and not key.endswith("_npu")
                            and hasattr(module, key[7:] + "_npu")
                            and not hasattr(module, key + "_npu")
                        ):
                            setattr(module, key + "_npu", value)
                            self.aliases.append(key + "_npu -> " + key)

                return module

            def visit_attribute(generator, node):
                if (
                    node.attr == "to"
                    and generator._in_context(node.value)
                    and isinstance(generator._context[node.value.id], tensor_type)
                ):
                    self.parameter_to_hits += 1

                    return ast.copy_location(
                        ast.Attribute(
                            value=generator._generate_load(
                                generator._context[node.value.id]
                            ),
                            attr="to",
                            ctx=node.ctx,
                        ),
                        node,
                    )

                return original_attribute(generator, node)

            jit.import_from_path = import_alias
            generation.CodeGenerator.visit_Attribute = visit_attribute
            self.modules = {
                name: importlib.import_module("ntops_lab.kernels.pointwise." + name)
                for name in operators
            }
            self.modules["rms_norm"] = importlib.import_module(
                "ntops_lab.kernels.normalization.rms_norm"
            )
            self.source_imports.update(
                {module.__name__: module.__file__ for module in self.modules.values()}
            )
        else:
            baseline = importlib.import_module("triton_ascend_ops")
            self.modules = {
                name: baseline.get_operator(name) for name in (*operators, "rms_norm")
            }
            self.source_imports["triton_ascend_ops"] = baseline.__file__
            self.source_imports["triton_ascend_ops.api"] = importlib.import_module(
                "triton_ascend_ops.api"
            ).__file__

    def provenance(self):
        return {
            "launcher_aliases": self.aliases,
            "parameter_to_lowering_enabled": self.backend == "ntops",
            "parameter_to_hits": self.parameter_to_hits,
            "source_imports": self.source_imports,
            "shape_adapters": "Broadcast, contiguous, flatten or reshape without numerical fallback.",
            "arithmetic_rewritten": False,
        }

    def call(self, name, *args, scope=None, eps=1e-5):
        torch = self.torch

        if self.collect:
            self.active_operation = name
            self.calls[scope or name] += 1
            shapes = [(tuple(x.shape), str(x.dtype), tuple(x.stride())) for x in args]
            self.shapes[name + ":" + str(shapes)] += 1

        with torch._C._DisableTorchDispatch():
            x = args[0]
            shape = tuple(x.shape)

            if name == "rms_norm":
                if eps != 1e-5:
                    raise ValueError(
                        "The existing ntops RMSNorm requires epsilon 1e-5."
                    )

                operands = (x.contiguous().view(-1, shape[-1]), args[1].contiguous())
                output = (
                    self.modules[name].run(*operands)
                    if self.backend == "ntops"
                    else self.modules[name](*operands, eps=eps)
                )
            else:
                if name in ("mul", "add"):
                    y = args[1]

                    if not isinstance(y, torch.Tensor) or x.dtype != y.dtype:
                        raise ValueError(
                            "Binary adapters require tensors of one dtype."
                        )

                    x, y = torch.broadcast_tensors(x, y)
                    shape = tuple(x.shape)
                    operands = (x.contiguous().view(-1), y.contiguous().view(-1))
                else:
                    operands = (x.contiguous().view(-1),)

                output = (
                    self.modules[name].run(*operands)
                    if self.backend == "ntops"
                    else self.modules[name](*operands)
                )

            return output.view(shape)


def stage_dispatch(adapter, stage):
    from torch.utils._python_dispatch import TorchDispatchMode

    torch = adapter.torch

    class StageDispatch(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}

            if (
                args
                and isinstance(args[0], torch.Tensor)
                and args[0].device.type == "npu"
                and args[0].dtype.is_floating_point
            ):
                if (
                    stage >= 3
                    and func is torch.ops.aten.add.Tensor
                    and len(args) > 1
                    and isinstance(args[1], torch.Tensor)
                ):
                    if kwargs.get("alpha", 1) != 1:
                        raise ValueError("TensorAdd coverage requires alpha=1.")

                    scope = "tensor_add"

                    if adapter.collect:
                        frame = inspect.currentframe().f_back
                        names = []

                        for _ in range(16):
                            if frame is None:
                                break

                            names.append(frame.f_code.co_name)
                            frame = frame.f_back

                        scope += (
                            ":rope" if "apply_rotary_pos_emb" in names else ":other"
                        )

                    return adapter.call("add", args[0], args[1], scope=scope)

                unary = {
                    torch.ops.aten.neg.default: "neg",
                    torch.ops.aten.sin.default: "sin",
                    torch.ops.aten.cos.default: "cos",
                }

                if stage >= 4 and func in unary:
                    return adapter.call(unary[func], args[0])

            if adapter.collect and func in (
                torch.ops.aten.mm.default,
                torch.ops.aten.bmm.default,
                torch.ops.aten.matmul.default,
                torch.ops.aten.addmm.default,
            ):
                signature = [
                    (tuple(value.shape), str(value.dtype))
                    for value in args
                    if isinstance(value, torch.Tensor)
                ]
                adapter.native_matrix_calls[f"{func}:{signature}"] += 1

            return func(*args, **kwargs)

    return StageDispatch()


@contextlib.contextmanager
def patched(
    adapter,
    stage,
    model=None,
    *,
    linear_tuning="auto",
    linear_block_sizes=(32, 32, 32),
    linear_weight_layout="view",
):
    from transformers.activations import SiLUActivation
    from transformers.models.llama.modeling_llama import LlamaMLP, LlamaRMSNorm

    torch = adapter.torch
    originals = (
        torch.nn.SiLU.forward,
        SiLUActivation.forward,
        LlamaRMSNorm.forward,
        LlamaMLP.forward,
    )

    def silu(module, x):
        return adapter.call("silu", x)

    def rms(module, x):
        return adapter.call("rms_norm", x, module.weight, eps=module.variance_epsilon)

    def mlp(module, x):
        return module.down_proj(
            adapter.call(
                "mul",
                module.act_fn(module.gate_proj(x)),
                module.up_proj(x),
                scope="mlp_gating_mul",
            )
        )

    torch.nn.SiLU.forward = silu
    SiLUActivation.forward = silu
    LlamaRMSNorm.forward = rms

    if stage >= 2:
        LlamaMLP.forward = mlp

    try:
        with contextlib.ExitStack() as patches:
            patches.enter_context(stage_dispatch(adapter, stage))
            if stage >= 5:
                if __package__:
                    from .e2e_attention import Adapter as AttentionAdapter
                else:
                    from e2e_attention import Adapter as AttentionAdapter

                attention = AttentionAdapter(adapter.backend, torch, recorder=adapter)
                adapter.core_operator_adapters["sdpa"] = attention
                patches.enter_context(attention.patched())
            if stage >= 6:
                if model is None:
                    raise ValueError("Linear replacement requires the loaded model.")
                if __package__:
                    from .e2e_linear import Adapter as LinearAdapter
                else:
                    from e2e_linear import Adapter as LinearAdapter

                linear = LinearAdapter(
                    adapter.backend,
                    torch,
                    recorder=adapter,
                    tuning=linear_tuning,
                    block_sizes=linear_block_sizes,
                    weight_layout=linear_weight_layout,
                )
                adapter.core_operator_adapters["linear"] = linear
                patches.enter_context(linear.patched(model))
            yield
    finally:
        (
            torch.nn.SiLU.forward,
            SiLUActivation.forward,
            LlamaRMSNorm.forward,
            LlamaMLP.forward,
        ) = originals


def load_model(config, torch):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = config["paths"]["model"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .eval()
        .to("npu:0")
    )

    if model.config.model_type != "llama" or model.config.rms_norm_eps != 1e-5:
        raise ValueError("This adapter requires a Llama model with RMS epsilon 1e-5.")

    return tokenizer, model


def generate(model, inputs, tokenizer, settings):
    return model.generate(
        **inputs,
        do_sample=False,
        use_cache=True,
        min_new_tokens=settings["output_tokens"],
        max_new_tokens=settings["output_tokens"],
        pad_token_id=tokenizer.eos_token_id,
    )


def decode_output(output, tokenizer, settings):
    output_ids = output[0, settings["input_tokens"] :].detach().cpu().tolist()
    text = tokenizer.decode(output_ids, skip_special_tokens=True)

    return {
        "output_ids": output_ids,
        "output_text": text,
        "sanity": text_sanity(output_ids, text, settings["output_tokens"]),
    }


def save_logits(model, inputs, torch, path, decode_input_id):
    """Optionally save compact float32 last-position diagnostics outside timing."""
    import numpy as np

    first = model(**inputs, use_cache=True)
    prefill = first.logits[:, -1, :].detach().float().cpu().contiguous().clone()
    token = torch.tensor([[decode_input_id]], dtype=torch.long, device="npu")
    mask = torch.ones(
        (1, inputs["input_ids"].shape[1] + 1), dtype=torch.long, device="npu"
    )
    second = model(
        input_ids=token,
        attention_mask=mask,
        past_key_values=first.past_key_values,
        use_cache=True,
    )
    decode = second.logits[:, -1, :].detach().float().cpu().contiguous().clone()
    torch.npu.synchronize()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, prefill=prefill.numpy(), decode=decode.numpy())

    return {
        "comparison_scope": "last_position_logits",
        "tensor_shape": list(prefill.shape),
        "path": str(path),
        "sha256": sha256_file(path),
        "dtype": "float32",
        "acceptance_gate": False,
    }


def measure_worker(config, backend, stage, order, identity):
    torch, device_cache = configure_runtime(config)
    settings = config["e2e"]
    output_root = Path(config["e2e_output_dir"])
    environment = runtime_metadata(torch, config)

    if backend != "reference":
        reference = json.loads(reference_path(config).read_text(encoding="utf-8"))
        stable_fields = (
            "python",
            "versions",
            "physical_device",
            "device_uuid",
            "device_name",
            "device_properties",
            "cann",
            "cpu_threads",
            "cpu_interop_threads",
            "cpu_affinity",
        )

        for field in stable_fields:
            if environment[field] != reference["environment"][field]:
                raise ValueError(f"Worker runtime differs from the reference: {field}")

    started = time.perf_counter()
    tokenizer, model = load_model(config, torch)
    input_ids = tokenizer(settings["prompt"], add_special_tokens=True)["input_ids"]

    if len(input_ids) != settings["input_tokens"]:
        raise ValueError(
            f"Prompt tokenized to {len(input_ids)} tokens; expected {settings['input_tokens']}."
        )

    inputs = {
        "input_ids": torch.tensor([input_ids], dtype=torch.long, device="npu"),
        "attention_mask": torch.ones(
            (1, len(input_ids)), dtype=torch.long, device="npu"
        ),
    }
    result = {
        "status": "PASS",
        "run_identity": identity,
        "backend": backend,
        "stage": stage,
        "order": order,
        "physical_device": config["device"],
        "reference_artifact": str(reference_path(config)),
        "environment": environment,
        "model": {
            "id": settings["model_id"],
            "revision": settings["model_revision"],
            "path": config["paths"]["model"],
            "dtype": settings["dtype"],
            "batch": 1,
            "prompt": settings["prompt"],
            "input_ids": input_ids,
            "input_tokens": len(input_ids),
            "output_tokens": settings["output_tokens"],
            "transformers_version": environment["versions"]["transformers"],
        },
        "generation_settings": {
            **settings,
            "do_sample": False,
            "use_cache": True,
            "attn_implementation": "sdpa",
        },
        "load_and_tokenize_seconds": time.perf_counter() - started,
        "semantic_review": {
            "status": "pending",
            "criterion": "The response is normal, coherent text relevant to the prompt.",
            "automatic_sanity_is_not_semantic_review": True,
        },
        "token_equality_is_acceptance_gate": False,
        "logits_are_acceptance_gate": False,
    }

    with torch.inference_mode():
        if backend == "reference":
            output = generate(model, inputs, tokenizer, settings)
            torch.npu.synchronize()
            result.update(decode_output(output, tokenizer, settings))

            if settings["logits_diagnostics"]:
                result["logits_diagnostic"] = save_logits(
                    model,
                    inputs,
                    torch,
                    output_root / "logits" / f"reference_device-{config['device']}.npz",
                    result["output_ids"][0],
                )

            if not result["sanity"]["passed"]:
                result["status"] = "SANITY_FAIL"

            return result

        adapter = Adapters(backend, torch, device_cache)
        reference = json.loads(reference_path(config).read_text(encoding="utf-8"))

        with patched(
            adapter,
            stage,
            model,
            linear_tuning=settings["linear_tuning"],
            linear_block_sizes=settings["linear_block_sizes"],
            linear_weight_layout=settings["linear_weight_layout"],
        ):
            if settings["logits_diagnostics"]:
                result["logits_diagnostic"] = save_logits(
                    model,
                    inputs,
                    torch,
                    output_root / "logits" / f"{order}_stage{stage}_{backend}.npz",
                    reference["output_ids"][0],
                )

            for index in range(settings["warmup"]):
                generate(model, inputs, tokenizer, settings)
                torch.npu.synchronize()
                print(f"WARMUP {index + 1}", flush=True)

            samples = []
            outputs = []

            for index in range(settings["samples"]):
                torch.npu.synchronize()
                begin = time.perf_counter_ns()
                output = generate(model, inputs, tokenizer, settings)
                torch.npu.synchronize()
                samples.append((time.perf_counter_ns() - begin) / 1e9)
                outputs.append(decode_output(output, tokenizer, settings))
                print(f"SAMPLE {index + 1} {samples[-1]:.6f}s", flush=True)

            from triton.runtime.jit import JITFunction

            original_run = JITFunction.run
            launches = collections.Counter()

            def observe(jit, *args, **kwargs):
                compiled = original_run(jit, *args, **kwargs)
                module = getattr(getattr(jit, "fn", None), "__module__", "")
                launches[f"{adapter.active_operation}:{module}:{jit.__name__}"] += 1

                return compiled

            adapter.collect = True
            JITFunction.run = observe

            try:
                counted = generate(model, inputs, tokenizer, settings)
            finally:
                JITFunction.run = original_run
                adapter.collect = False

            torch.npu.synchronize()
            counted_output = decode_output(counted, tokenizer, settings)
            required = ["rms_norm", "silu"]

            if stage >= 2:
                required.append("mlp_gating_mul")

            if stage >= 4:
                required += ["neg", "sin", "cos"]

            if stage >= 5:
                required.append("sdpa")
            if stage >= 6:
                required.extend(
                    "linear:" + name
                    for name in (
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                        "lm_head",
                    )
                )

            if not all(adapter.calls[name] > 0 for name in required):
                raise RuntimeError(
                    f"Requested patches did not execute: {adapter.calls}"
                )

            if stage >= 3 and not any(
                key.startswith("tensor_add") for key in adapter.calls
            ):
                raise RuntimeError("Requested TensorAdd patches did not execute.")

            if backend == "ntops" and any(
                name.rsplit(":", 1)[-1] == "_matmul_kernel" for name in launches
            ):
                raise RuntimeError(
                    "A handwritten matmul fallback executed in the NineToothed path."
                )

            core_coverage = validate_core_launches(
                backend,
                stage,
                adapter.calls,
                launches,
                adapter.core_operator_adapters["linear"].provenance()["tuning"][
                    "kernel_name"
                ]
                if stage >= 6
                else None,
            )
            if stage >= 6:
                linear_provenance = adapter.core_operator_adapters[
                    "linear"
                ].provenance()
                missing = [
                    item["name"]
                    for item in linear_provenance["patched_modules"]
                    if linear_provenance["module_calls"].get(item["name"], 0) <= 0
                ]
                if missing:
                    raise RuntimeError(
                        f"Requested Linear modules did not execute: {missing}"
                    )
                core_coverage["linear_mm"]["all_patched_modules_executed"] = True
                core_coverage["linear_mm"]["module_count"] = len(
                    linear_provenance["patched_modules"]
                )

        if not all(math.isfinite(sample) and sample > 0 for sample in samples):
            raise RuntimeError("Generation timing produced an invalid sample.")

        median = statistics.median(samples)
        result.update(
            scope=SCOPES[stage],
            samples_seconds=samples,
            median_seconds=median,
            tokens_per_second=settings["output_tokens"] / median,
            sample_outputs=outputs,
            output_ids=outputs[0]["output_ids"],
            output_text=outputs[0]["output_text"],
            sanity={
                "passed": all(
                    value["sanity"]["passed"] for value in [*outputs, counted_output]
                )
            },
            tokens_stable_across_samples=all(
                value["output_ids"] == outputs[0]["output_ids"] for value in outputs
            ),
            tokens_match_reference=all(
                value["output_ids"] == reference["output_ids"] for value in outputs
            ),
            text_matches_reference=all(
                value["output_text"] == reference["output_text"] for value in outputs
            ),
            counted_output=counted_output,
            counted_tokens_match_samples=counted_output["output_ids"]
            == outputs[0]["output_ids"],
            operator_calls=dict(adapter.calls),
            native_matrix_calls=dict(adapter.native_matrix_calls),
            core_kernel_coverage=core_coverage,
            shape_calls=dict(adapter.shapes),
            jit_launches=dict(launches),
            generated_sources={
                key.rsplit(":", 2)[1]: sha256_file(key.rsplit(":", 2)[1])
                for key in launches
                if Path(key.rsplit(":", 2)[1]).is_file()
            },
            compatibility_shim=adapter.provenance(),
            core_operator_adapters={
                name: implementation.provenance()
                for name, implementation in adapter.core_operator_adapters.items()
            },
            measurement_role="text_output_review_pending",
            acceptance_eligible=None,
        )

        if not result["sanity"]["passed"]:
            result["status"] = "SANITY_FAIL"
            result["acceptance_eligible"] = False

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--backend", choices=("reference", "triton", "ntops"), required=True
    )
    parser.add_argument("--stage", type=int, choices=[0, *SCOPES], default=0)
    parser.add_argument("--order", default="reference")
    parser.add_argument("--identity", required=True)
    parser.add_argument("--device", type=int)
    parser.add_argument(
        "--cpu-affinity", help="Comma-separated logical CPU identifiers."
    )
    args = parser.parse_args()
    config = read_config(args.config)

    if args.device is not None:
        if args.device not in config["e2e"]["devices"]:
            parser.error("Worker device is not listed in the E2E device configuration.")

        config["device"] = args.device

    if args.cpu_affinity is not None:
        try:
            affinity = [int(value) for value in args.cpu_affinity.split(",")]
        except ValueError:
            parser.error("CPU affinity must be a comma-separated integer list.")

        if (
            not affinity
            or any(value < 0 for value in affinity)
            or len(set(affinity)) != len(affinity)
        ):
            parser.error("CPU affinity must contain distinct nonnegative integers.")

        config["assigned_cpu_affinity"] = sorted(affinity)

    if (args.backend == "reference") != (args.stage == 0):
        parser.error("Stage 0 is reserved for the unmodified Hugging Face reference.")

    output_root = Path(config["e2e_output_dir"])
    destination = (
        reference_path(config)
        if args.backend == "reference"
        else output_root
        / "results"
        / f"{args.order}_stage{args.stage}_{args.backend}.json"
    )

    try:
        result = measure_worker(
            config, args.backend, args.stage, args.order, args.identity
        )
    except Exception as error:
        result = {
            "status": "ERROR",
            "run_identity": args.identity,
            "backend": args.backend,
            "stage": args.stage,
            "order": args.order,
            "physical_device": config["device"],
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        traceback.print_exc()

    write_json(destination, result)
    print(f"{result['status']} {destination}", flush=True)

    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
