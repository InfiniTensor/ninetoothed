import functools
import importlib.util
import json
import math
import os
import pathlib
import sys
import time
import uuid


class AscendAotBackend:
    """NPU runtime backend for AOT entrypoints.

    This backend intentionally avoids CUDA-only `triton.tools.compile`
    and `nvcc` stages. It reuses generated Python launch wrappers and
    resolves the NPU-specific launch symbol when present.
    """

    def __call__(self, func, *, kernel_name, num_warps, num_stages):
        num_warps, num_stages = _normalize_compile_options(num_warps, num_stages)
        artifact = build_kernel_artifact(
            func, kernel_name=kernel_name, num_warps=num_warps, num_stages=num_stages
        )
        return load_kernel_artifact(artifact)


_UNKNOWN_SIZE_BUCKET = -1
_MULTI_AXIS_MAX_CANDIDATES_DEFAULT = 16
_MULTI_AXIS_MAX_TUNE_MS_DEFAULT = 400.0


# -----------------------------------------------------------------------------
# Shared Utilities
# -----------------------------------------------------------------------------


def _first_tensor_like_arg(args):
    return next((arg for arg in args if hasattr(arg, "numel")), None)


def _cdiv(x, y):
    return (int(x) + int(y) - 1) // int(y)


# -----------------------------------------------------------------------------
# Policy / Routing
# -----------------------------------------------------------------------------


def should_use_ascend_aot_dispatch(caller):
    """Return whether Ascend AOT dispatch should be used for current runtime."""
    if caller not in ("torch", "ascend"):
        return False

    try:
        import torch

        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


def _normalize_compile_options(num_warps, num_stages):
    """Fill missing compile options using project defaults."""
    from ninetoothed.utils import calculate_default_configs

    default_num_warps, default_num_stages = calculate_default_configs()

    if num_warps is None:
        num_warps = default_num_warps

    if num_stages is None:
        num_stages = default_num_stages

    return int(num_warps), int(num_stages)


# -----------------------------------------------------------------------------
# Observability
# -----------------------------------------------------------------------------


def _debug_enabled():
    return os.getenv("NINETOOTHED_ASCEND_DEBUG_META", "0") == "1"


def _debug_meta_selection(kernel_name, size_bucket, config_values, meta_values, source):
    if not _debug_enabled():
        return

    print(
        f"[ninetoothed][ascend][meta] kernel={kernel_name} "
        f"bucket={size_bucket} config={config_values} meta={meta_values} source={source}",
        flush=True,
    )


def _multi_axis_max_candidates():
    raw = os.getenv("NINETOOTHED_MULTI_AXIS_MAX_CANDIDATES", "")
    if not raw:
        return _MULTI_AXIS_MAX_CANDIDATES_DEFAULT

    try:
        value = int(raw)
    except ValueError:
        return _MULTI_AXIS_MAX_CANDIDATES_DEFAULT

    return max(1, value)


def _multi_axis_max_tune_ms():
    raw = os.getenv("NINETOOTHED_MULTI_AXIS_MAX_TUNE_MS", "")
    if not raw:
        return _MULTI_AXIS_MAX_TUNE_MS_DEFAULT

    try:
        value = float(raw)
    except ValueError:
        return _MULTI_AXIS_MAX_TUNE_MS_DEFAULT

    return max(1.0, value)


# -----------------------------------------------------------------------------
# Artifact Build / Load
# -----------------------------------------------------------------------------


def build_kernel_artifact(func, *, kernel_name, num_warps, num_stages):
    """Build a Python-launch-wrapper artifact manifest for Ascend runtime."""
    # Lazy import prevents cache-dir resolution at module import time
    # inside process-pool workers.
    from ninetoothed.generation import CodeGenerator

    code_generator = CodeGenerator()
    source_file = code_generator(
        func,
        caller="torch",
        kernel_name=kernel_name,
        num_warps=num_warps,
        num_stages=num_stages,
        max_num_configs=None,
        prettify=False,
    )

    source_path = pathlib.Path(source_file).resolve()
    launch_name = code_generator.launch_func_name

    artifact = {
        "schema_version": 1,
        "backend": "ascend",
        "kind": "python_launch_wrapper",
        "kernel_name": kernel_name,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "source_file": str(source_path),
        "launch_name": launch_name,
        "preferred_launch_name": f"{launch_name}_npu",
    }
    artifact["artifact_id"] = _make_artifact_id(artifact)

    manifest_path = source_path.with_suffix(
        f".{kernel_name}.ascend-aot.manifest.json"
    )
    _write_manifest(manifest_path, artifact)
    artifact["manifest_path"] = str(manifest_path)

    return artifact


def load_kernel_artifact(artifact):
    """Load a launch callable from an artifact dict or manifest path."""
    artifact = _normalize_artifact(artifact)

    source_path = pathlib.Path(artifact["source_file"])
    module_name = f"{source_path.stem}_{artifact.get('artifact_id', uuid.uuid4().hex)}"
    module = _import_from_path(module_name, str(source_path))
    module_vars = vars(module)

    launch_name = artifact["launch_name"]
    preferred_launch_name = artifact["preferred_launch_name"]

    if preferred_launch_name in module_vars:
        launch_name = preferred_launch_name

    if launch_name not in module_vars:
        raise KeyError(f"Launch symbol `{launch_name}` not found in `{source_path}`.")

    return module_vars[launch_name]


def _import_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def _normalize_artifact(artifact):
    if isinstance(artifact, (str, pathlib.Path)):
        return _read_manifest(pathlib.Path(artifact))

    if isinstance(artifact, dict):
        manifest_path = artifact.get("manifest_path")

        if manifest_path:
            merged = _read_manifest(pathlib.Path(manifest_path))
            merged.update(artifact)
            return merged

        return _backfill_legacy_artifact(artifact)

    raise TypeError("Ascend artifact must be a dict or manifest path.")


def _backfill_legacy_artifact(artifact):
    source_file = artifact["source_file"]
    launch_name = artifact["launch_name"]
    normalized = dict(artifact)

    normalized.setdefault("schema_version", 1)
    normalized.setdefault("backend", "ascend")
    normalized.setdefault("kind", "python_launch_wrapper")
    normalized.setdefault("preferred_launch_name", f"{launch_name}_npu")
    normalized.setdefault(
        "artifact_id", uuid.uuid5(uuid.NAMESPACE_URL, f"{source_file}:{launch_name}").hex
    )

    return normalized


def _write_manifest(path, artifact):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True))


def _read_manifest(path):
    return _backfill_legacy_artifact(json.loads(path.read_text()))


def _make_artifact_id(artifact):
    payload = json.dumps(artifact, sort_keys=True, separators=(",", ":"))
    return uuid.uuid5(uuid.NAMESPACE_URL, payload).hex


# -----------------------------------------------------------------------------
# Record Build / Assemble
# -----------------------------------------------------------------------------


def build_record(
    premake,
    config,
    *,
    kernel_name,
    arg_to_int,
    generate_suffix,
    annotate_application,
):
    """Build one precompiled record for a single config tuple."""
    args, kwargs, compilation_configs = config

    arrangement, application, tensors = premake(*args, **kwargs)

    import inspect

    premake_signature = inspect.signature(premake)
    bound_arguments = premake_signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()
    combination = bound_arguments.arguments
    combination = {f"{name}_": value for name, value in combination.items()}
    combination |= compilation_configs

    for name, value in combination.items():
        combination[name] = arg_to_int(value)

    kernel_name_ = f"{kernel_name}_{generate_suffix(combination.values())}"

    annotate_application(arrangement, application, tensors)

    built_kernel = build_kernel_artifact(
        application,
        kernel_name=kernel_name_,
        num_warps=compilation_configs.get("num_warps"),
        num_stages=compilation_configs.get("num_stages"),
    )

    application_signature = inspect.signature(application)
    param_names = tuple(application_signature.parameters.keys())

    return kernel_name_, param_names, combination, config, tensors, built_kernel


def build_from_records(
    records,
    *,
    meta_parameters,
    caller,
    kernel_name,
    output_dir,
    arg_to_int,
    kernel_launch_error_cls,
    auto_tune_fn,
    auto_tuned_kernel_cls,
):
    """Assemble dispatcher and auto-tuned wrapper from prebuilt records."""
    configs = tuple(record[3] for record in records)
    all_tensors = tuple(record[4] for record in records)
    all_param_names = tuple(record[1] for record in records)
    combinations = tuple(record[2] for record in records)
    built_kernels = tuple(load_kernel_artifact(record[5]) for record in records)

    tensor_param_names = tuple(
        functools.reduce(
            lambda x, y: dict.fromkeys(x) | dict.fromkeys(y),
            sorted(all_param_names, key=len, reverse=True),
            {},
        )
    )
    non_tensor_param_names = tuple(
        functools.reduce(lambda x, y: x | y, combinations, {})
    )

    kernel_before_auto_tuning = build_dispatch_kernel(
        tensor_param_names=tensor_param_names,
        non_tensor_param_names=non_tensor_param_names,
        all_param_names=all_param_names,
        combinations=combinations,
        built_kernels=built_kernels,
        arg_to_int=arg_to_int,
        kernel_launch_error_cls=kernel_launch_error_cls,
    )

    if meta_parameters is None:
        return kernel_before_auto_tuning

    config_to_best_meta_arguments = auto_tune_fn(
        kernel_before_auto_tuning,
        configs,
        all_tensors,
        meta_parameters,
        caller=caller,
        kernel_name=kernel_name,
        output_dir=output_dir,
    )

    meta_value_set = set()
    for _, kwargs, compilation_configs in configs:
        meta = {
            param: kwargs[param] for param in meta_parameters if param in kwargs
        } | compilation_configs
        meta_value_set.add(tuple(arg_to_int(value) for value in meta.values()))

    kernel_after_auto_tuning = build_kernel_with_auto_tuning(
        config_to_best_meta_arguments,
        non_tensor_param_names,
        kernel_name=kernel_name,
        tensor_param_names=tensor_param_names,
        kernel_before_auto_tuning=kernel_before_auto_tuning,
        all_meta_values=tuple(meta_value_set),
        arg_to_int=arg_to_int,
        kernel_launch_error_cls=kernel_launch_error_cls,
    )

    return auto_tuned_kernel_cls(
        kernel_after_auto_tuning=kernel_after_auto_tuning,
        kernel_before_auto_tuning=kernel_before_auto_tuning,
        configs=configs,
        meta_parameters=meta_parameters,
        config_to_best_meta_arguments=config_to_best_meta_arguments,
        kernel_name=kernel_name,
        output_dir=output_dir,
        runtime_fallback_handler=try_runtime_meta_fallback,
    )


# -----------------------------------------------------------------------------
# Dispatch / Matching
# -----------------------------------------------------------------------------


def _match_combination(non_tensor_args, combination, arg_to_int):
    """Return True when runtime non-tensor args match one prebuilt config."""
    for key, expected in combination.items():
        if key not in non_tensor_args:
            return False

        if arg_to_int(non_tensor_args[key]) != expected:
            return False

    return True


def build_dispatch_kernel(
    *,
    tensor_param_names,
    non_tensor_param_names,
    all_param_names,
    combinations,
    built_kernels,
    arg_to_int,
    kernel_launch_error_cls,
):
    """Build runtime dispatcher that selects a concrete kernel by config."""

    def dispatch_kernel(*args, **kwargs):
        if kwargs:
            raise TypeError("Ascend AOT dispatcher only supports positional arguments.")

        expected_num_args = len(tensor_param_names) + len(non_tensor_param_names)

        if len(args) != expected_num_args:
            raise TypeError(
                f"Expected {expected_num_args} arguments, got {len(args)}."
            )

        tensor_args = dict(zip(tensor_param_names, args[: len(tensor_param_names)]))
        non_tensor_args = dict(
            zip(non_tensor_param_names, args[len(tensor_param_names) :])
        )

        for param_names, combination, kernel in zip(
            all_param_names, combinations, built_kernels
        ):
            if not _match_combination(non_tensor_args, combination, arg_to_int):
                continue

            call_args = tuple(tensor_args[name] for name in param_names)
            return kernel(*call_args)

        raise kernel_launch_error_cls("No matching Ascend AOT kernel configuration found.")

    return dispatch_kernel


# -----------------------------------------------------------------------------
# Auto-tuning Adaptation
# -----------------------------------------------------------------------------


def _has_multi_axis_block_sizes(meta_param_to_index):
    """Return True when meta defines two or more block_size_* axes."""
    axes = tuple(
        name
        for name in meta_param_to_index
        if name.startswith("block_size_") and len(name) > len("block_size_")
    )
    return len(axes) >= 2


def _estimate_core_dim(*, tensor_args, meta_values, meta_param_to_index):
    """Estimate launch coreDim for simple 1D block-size layouts."""

    if _has_multi_axis_block_sizes(meta_param_to_index):
        # Avoid false negatives from inaccurate static estimation for multi-axis kernels.
        return None

    block_size_index = None
    for name, index in meta_param_to_index.items():
        if name.startswith("block_size"):
            block_size_index = index
            break

    if block_size_index is None:
        return None

    first_tensor = _first_tensor_like_arg(tensor_args)
    if first_tensor is None:
        return None

    block_size = int(meta_values[block_size_index])
    if block_size <= 0:
        return -1

    return _cdiv(int(first_tensor.numel()), block_size)


def _is_meta_safe_for_tensor_args(args, meta_values, meta_param_to_index, core_dim_limit):
    """Check whether static coreDim estimate is within hardware limit."""
    core_dim = _estimate_core_dim(
        tensor_args=args,
        meta_values=meta_values,
        meta_param_to_index=meta_param_to_index,
    )
    if core_dim is None:
        return True
    return core_dim <= core_dim_limit


def _pick_safe_meta_values(
    tensor_args, preferred_meta_values, all_meta_values, meta_param_to_index, core_dim_limit
):
    """Pick a statically safe fallback meta when possible."""
    if _has_multi_axis_block_sizes(meta_param_to_index):
        # For multi-axis kernels (e.g. GEMM-like), launch dimensionality cannot be
        # inferred safely from names alone. Keep current choice and rely on runtime
        # execution/bench filtering to reject invalid meta configs.
        return preferred_meta_values

    def is_safe(meta_values):
        core_dim = _estimate_core_dim(
            tensor_args=tensor_args,
            meta_values=meta_values,
            meta_param_to_index=meta_param_to_index,
        )
        if core_dim is None:
            return True
        if core_dim <= 0:
            return False
        return core_dim <= core_dim_limit

    if is_safe(preferred_meta_values):
        return preferred_meta_values

    candidates = sorted(all_meta_values, reverse=True)
    for meta_values in candidates:
        if is_safe(meta_values):
            return meta_values

    return preferred_meta_values


def _size_bucket_from_tensor_args(tensor_args):
    """Return floor(log2(numel)) for first tensor-like argument."""
    first_tensor = _first_tensor_like_arg(tensor_args)
    if first_tensor is None:
        return _UNKNOWN_SIZE_BUCKET

    numel = int(first_tensor.numel())
    if numel <= 0:
        return _UNKNOWN_SIZE_BUCKET

    return int(math.log2(numel))


def _compose_size_aware_config_key(config_values, tensor_args):
    """Compose config key with an extra size bucket suffix."""
    return (*config_values, _size_bucket_from_tensor_args(tensor_args))


def _tune_meta_for_size_bucket(
    *,
    kernel_before_auto_tuning,
    args,
    all_meta_values,
    meta_param_to_index,
    core_dim_limit,
    seed_meta_values=None,
):
    """Benchmark candidate metas and return best one for current runtime args."""
    import triton

    if _has_multi_axis_block_sizes(meta_param_to_index):
        candidate_order = _prioritize_multi_axis_candidates(
            all_meta_values=all_meta_values,
            seed_meta_values=seed_meta_values,
        )
        safe_candidates = _select_multi_axis_candidates(
            candidate_order=candidate_order,
            all_meta_values=all_meta_values,
            seed_meta_values=seed_meta_values,
            max_candidates=_multi_axis_max_candidates(),
        )
        tune_budget_ms = _multi_axis_max_tune_ms()
    else:
        safe_candidates = tuple(
            meta
            for meta in all_meta_values
            if _is_meta_safe_for_tensor_args(args, meta, meta_param_to_index, core_dim_limit)
        )
        tune_budget_ms = None

    if not safe_candidates:
        return None

    best_meta = None
    best_timing = float("inf")
    started = time.perf_counter()

    for meta_values in safe_candidates:
        if tune_budget_ms is not None:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if elapsed_ms >= tune_budget_ms:
                break

        try:
            timing = triton.testing.do_bench(
                lambda meta_values=meta_values: kernel_before_auto_tuning(
                    *args, *meta_values
                )
            )
        except Exception:
            timing = float("inf")

        if timing < best_timing:
            best_timing = timing
            best_meta = meta_values

    return best_meta


def _prioritize_multi_axis_candidates(*, all_meta_values, seed_meta_values):
    candidates = list(all_meta_values)
    if not candidates:
        return candidates

    if seed_meta_values is None:
        return sorted(candidates, reverse=True)

    def distance(meta_values):
        return sum(abs(int(a) - int(b)) for a, b in zip(meta_values, seed_meta_values))
    by_near = sorted(candidates, key=lambda meta: (0 if meta == seed_meta_values else 1, distance(meta)))
    by_far = sorted(candidates, key=lambda meta: distance(meta), reverse=True)

    # Interleave neighborhood exploitation with far-distance exploration
    # to avoid being trapped by a bad legacy seed.
    ordered = []
    used = set()

    def append_once(meta):
        key = tuple(meta)
        if key in used:
            return False
        ordered.append(meta)
        used.add(key)
        return True

    if seed_meta_values in by_near:
        append_once(seed_meta_values)

    near_i = 0
    far_i = 0
    while len(ordered) < len(candidates):
        for _ in range(3):
            while near_i < len(by_near):
                meta = by_near[near_i]
                near_i += 1
                if append_once(meta):
                    break
            else:
                break
        while far_i < len(by_far):
            meta = by_far[far_i]
            far_i += 1
            if append_once(meta):
                break
        if near_i >= len(by_near) and far_i >= len(by_far):
            break

    return ordered


def _select_multi_axis_candidates(
    *, candidate_order, all_meta_values, seed_meta_values, max_candidates
):
    """Select bounded candidates while forcing exploration against bad seeds."""
    if not candidate_order:
        return tuple()

    capped = list(candidate_order[:max_candidates])
    if seed_meta_values is None or max_candidates <= 1:
        return tuple(capped)

    def distance(meta_values):
        return sum(abs(int(a) - int(b)) for a, b in zip(meta_values, seed_meta_values))

    far_candidates = sorted(all_meta_values, key=distance, reverse=True)
    explore_slots = min(len(capped), max(1, max_candidates // 4))
    protected = list(capped[: max(0, len(capped) - explore_slots)])
    selected_set = {tuple(meta) for meta in protected}

    # Refill tail using far candidates to avoid being trapped by local-neighborhood picks.
    for meta in far_candidates:
        key = tuple(meta)
        if key in selected_set:
            continue
        protected.append(meta)
        selected_set.add(key)
        if len(protected) >= len(capped):
            break

    # Keep deterministic execution order with exploration distributed early.
    if len(protected) <= 2:
        return tuple(protected)

    head = protected[::2]
    tail = protected[1::2]
    return tuple(head + tail)


def build_kernel_with_auto_tuning(
    config_to_best_meta_arguments,
    non_tensor_param_names,
    *,
    kernel_name,
    tensor_param_names,
    kernel_before_auto_tuning,
    all_meta_values,
    arg_to_int,
    kernel_launch_error_cls,
):
    """Wrap dispatch kernel with size-aware meta selection."""
    num_non_meta_premake_params = len(next(iter(config_to_best_meta_arguments)))
    config_param_names = non_tensor_param_names[:num_non_meta_premake_params]
    meta_param_names = non_tensor_param_names[num_non_meta_premake_params:]
    meta_param_to_index = {name: i for i, name in enumerate(meta_param_names)}
    all_meta_values = tuple(all_meta_values)
    core_dim_hard_limit = (1 << 16) - 1

    def kernel_after_auto_tuning(*args, **kwargs):
        if kwargs:
            raise TypeError("Ascend AOT dispatcher only supports positional arguments.")

        expected_num_args = len(tensor_param_names) + len(config_param_names)

        if len(args) != expected_num_args:
            raise TypeError(
                f"Expected {expected_num_args} arguments, got {len(args)}."
            )

        config_values = tuple(
            arg_to_int(value) for value in args[len(tensor_param_names) :]
        )
        size_aware_config_values = _compose_size_aware_config_key(
            config_values, args[: len(tensor_param_names)]
        )

        selection_source = "size_cache_hit"
        meta_values = config_to_best_meta_arguments.get(size_aware_config_values)
        if meta_values is None:
            selection_source = "size_bucket_tune"
            meta_values = _tune_meta_for_size_bucket(
                kernel_before_auto_tuning=kernel_before_auto_tuning,
                args=args,
                all_meta_values=all_meta_values,
                meta_param_to_index=meta_param_to_index,
                core_dim_limit=core_dim_hard_limit,
                seed_meta_values=config_to_best_meta_arguments.get(config_values),
            )
            if meta_values is not None:
                config_to_best_meta_arguments[size_aware_config_values] = meta_values
            else:
                # Backward compatibility for old csv entries keyed without size bucket.
                selection_source = "legacy_cache_hit"
                meta_values = config_to_best_meta_arguments.get(config_values)

        if meta_values is None:
            selection_source = "default_fallback"
            if config_to_best_meta_arguments:
                meta_values = next(iter(config_to_best_meta_arguments.values()))
            else:
                raise kernel_launch_error_cls(
                    "No auto-tuning entries for Ascend AOT kernel."
                )

        meta_values = _pick_safe_meta_values(
            args[: len(tensor_param_names)],
            meta_values,
            all_meta_values,
            meta_param_to_index,
            core_dim_hard_limit,
        )

        size_bucket = (
            size_aware_config_values[-1]
            if size_aware_config_values
            else _UNKNOWN_SIZE_BUCKET
        )
        _debug_meta_selection(
            kernel_name=kernel_name,
            size_bucket=size_bucket,
            config_values=config_values,
            meta_values=meta_values,
            source=selection_source,
        )

        return kernel_before_auto_tuning(*args, *meta_values)

    return kernel_after_auto_tuning


# -----------------------------------------------------------------------------
# Runtime Fallback (Ascend coreDim limit)
# -----------------------------------------------------------------------------


def is_core_dim_limit_error(error):
    """Recognize Ascend runtime errors caused by coreDim limit overflow."""
    message = str(error)
    return "coreDim=" in message and "UINT16_MAX" in message


def _read_cached_meta(kernel_state, config_key):
    """Read cached meta for key, with compatibility fallback to legacy key."""
    csv_path = kernel_state._output_dir / f"{kernel_state._kernel_name}.csv"
    cached = kernel_state._read_auto_tuning_cache(csv_path) or {}
    meta = cached.get(config_key)
    if meta is not None:
        return meta

    # Backward compatibility: try old key layout without size bucket.
    if config_key:
        return cached.get(config_key[:-1])

    return None


def _find_runtime_fallback_meta(kernel_state, tensor_args, config_args, config_key, kwargs):
    """Find a runtime-valid fallback meta from remaining candidates."""
    candidates = sorted(kernel_state._all_meta_values, reverse=True)

    current = _read_cached_meta(kernel_state, config_key)
    if current in candidates:
        candidates.remove(current)

    for meta_values in candidates:
        try:
            kernel_state._kernel_before_auto_tuning(
                *tensor_args, *config_args, *meta_values, **kwargs
            )
            return meta_values
        except RuntimeError as error:
            if is_core_dim_limit_error(error):
                continue
            raise
        except kernel_state._kernel_launch_error_cls:
            continue

    return None


def try_runtime_meta_fallback(kernel_state, args, kwargs):
    """Retry launch with fallback meta when coreDim overflow is encountered."""
    try:
        return kernel_state._kernel_after_auto_tuning(*args, **kwargs), True
    except RuntimeError as error:
        if not is_core_dim_limit_error(error):
            raise

        tensor_args, config_args, config_key = kernel_state._split_args(args)
        fallback_meta = _find_runtime_fallback_meta(
            kernel_state, tensor_args, config_args, config_key, kwargs
        )

        if fallback_meta is None:
            raise

        updated = {config_key: fallback_meta}
        csv_path = kernel_state._output_dir / f"{kernel_state._kernel_name}.csv"
        kernel_state._append_auto_tuning_cache(csv_path, updated)
        kernel_state._known_configs.update(updated.keys())

        return (
            kernel_state._kernel_before_auto_tuning(
                *tensor_args, *config_args, *fallback_meta
            ),
            True,
        )
