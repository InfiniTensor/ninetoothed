"""Build and load Ascend device binaries with precompiled Python launchers."""

import ast
import hashlib
import importlib.util
import json
import math
import shutil
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

from ninetoothed.compiler.cache import compilation_cache_key, write_manifest


def _extension(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _launch_source(source, entrypoint, block):
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == entrypoint
    )
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "block"
            for target in node.targets
        ):
            node.value = ast.Constant(block)
    return ast.unparse(ast.fix_missing_locations(function)) + "\n"


def build(compilation, *, output_dir):
    import torch
    import triton
    from triton.runtime import driver

    from ninetoothed.backends.materializers.triton import (
        _compile_block,
        _compile_schedule,
        _compile_signature,
    )
    from ninetoothed.compiler.runtime import Handle, import_python_module

    if any(spec.dtype is None for spec in compilation.kernel.tensors):
        raise ValueError("Ascend AOT requires explicit tensor dtypes at build time.")
    artifact = compilation.artifact
    key = compilation_cache_key(compilation)
    output = Path(output_dir).resolve() / key
    output.mkdir(parents=True, exist_ok=True)
    source_path = output / f"{artifact.kernel_name}.ascend_triton.py"
    source_path.write_text(artifact.primary_source, encoding="utf-8")
    module = import_python_module(source_path)
    kernel = getattr(module, f"{artifact.kernel_name}_kernel")
    entries = _compile_signature(compilation).split(",")
    if len(entries) != len(kernel.arg_names):
        raise ValueError("Ascend AOT kernel signature does not match the launch ABI.")
    signature, constants = {}, {}
    for name, value in zip(kernel.arg_names, entries):
        try:
            constants[name] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            signature[name] = value
    warps, stages = _compile_schedule(compilation)
    compiled = triton.compile(
        triton.compiler.ASTSource(kernel, signature=signature, constants=constants),
        options={"num_warps": warps, "num_stages": stages},
    )
    # Materialize both native host extensions now, never on artifact reload.
    compiled._init_handles()
    binary = output / "kernel.bin"
    binary.write_bytes(compiled.kernel)
    shutil.copy2(compiled.run.launch.__self__.__file__, output / "launcher.so")
    shutil.copy2(driver.active.utils.npu_utils_mod.__file__, output / "npu_utils.so")
    (output / "launch.py").write_text(
        _launch_source(
            artifact.primary_source, artifact.entrypoint, _compile_block(compilation)
        ),
        encoding="utf-8",
    )
    info = {
        "schema": 1,
        "python": list(sys.version_info[:2]),
        "device_name": torch.npu.get_device_name(),
        "name": compiled.name,
        "shared": compiled.metadata.shared,
        "packed_metadata": compiled.packed_metadata,
        "arg_names": kernel.arg_names,
        "signature_names": list(signature),
        "constants": constants,
        "files": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ("kernel.bin", "launcher.so", "npu_utils.so", "launch.py")
        },
    }
    write_manifest(output / "bundle.json", info)
    built = SimpleNamespace(
        source=artifact,
        binary_path=str(binary),
        abi=artifact.metadata["launch_abi"],
    )
    launch = load(built)
    return Handle(compilation, compiled, launch, source_path, binary)


def load(built):
    import torch
    import torch_npu  # noqa: F401

    from ninetoothed.compiler.runtime import (
        _launch_abi_from_dict,
        _runtime_specs,
        _runtime_wrapper,
        _verified_runtime_launch,
    )
    from ninetoothed.targets import runtime_device_types

    if built.binary_path is None:
        raise ValueError("Ascend AOT artifact has no device binary.")
    output = Path(built.binary_path).parent
    info = json.loads((output / "bundle.json").read_text(encoding="utf-8"))
    if info["schema"] != 1 or info["python"] != list(sys.version_info[:2]):
        raise ValueError("Incompatible Ascend AOT bundle or Python ABI.")
    for name in ("kernel.bin", "launcher.so", "npu_utils.so", "launch.py"):
        if (
            hashlib.sha256((output / name).read_bytes()).hexdigest()
            != info["files"][name]
        ):
            raise ValueError(f"Ascend AOT bundle checksum mismatch: {name}.")
    utils = _extension(output / "npu_utils.so", "npu_utils")
    launcher = _extension(output / "launcher.so", "__triton_launcher")
    binary = (output / "kernel.bin").read_bytes()
    handles = {}
    lock = threading.Lock()

    class Kernel:
        def __getitem__(self, grid):
            def invoke(*args, **kwargs):
                values = dict(zip(info["arg_names"], args)) | kwargs
                for name, value in info["constants"].items():
                    if values.get(name) != value:
                        raise ValueError(
                            f"Ascend AOT constant `{name}` must equal {value}."
                        )
                device = torch.npu.current_device()
                if torch.npu.get_device_name(device) != info["device_name"]:
                    raise ValueError(
                        "Ascend AOT device does not match the build device."
                    )
                with lock:
                    if device not in handles:
                        name, mix_mode = info["name"].split()
                        handles[device] = utils.load_kernel_binary(
                            name, binary, info["shared"], device, mix_mode
                        )
                    function = handles[device][1]
                dimensions = tuple(grid) + (1,) * (3 - len(grid))
                stream = torch.npu.current_stream(device).npu_stream
                launcher.launch(
                    *dimensions,
                    stream,
                    function,
                    dict(info["packed_metadata"]),
                    None,
                    None,
                    None,
                    *(values[name] for name in info["signature_names"]),
                )

            return invoke

    namespace = {
        "triton": SimpleNamespace(cdiv=lambda x, y: (x + y - 1) // y),
        "floor": math.floor,
        f"{built.source.kernel_name}_kernel": Kernel(),
    }
    exec(
        compile((output / "launch.py").read_text(), str(output / "launch.py"), "exec"),
        namespace,
    )
    return _verified_runtime_launch(
        _runtime_wrapper(
            namespace[built.source.entrypoint],
            _launch_abi_from_dict(built.abi),
            specs=_runtime_specs(built.source),
            device_types=runtime_device_types(built.source),
        )
    )
