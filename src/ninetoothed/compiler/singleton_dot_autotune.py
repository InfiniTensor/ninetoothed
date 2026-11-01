"""Select singleton-dot schedules with a fixed model shortlist and measured timings.

The parameter space is the original and multiply/reduce implementations crossed
with legal power-of-two warp counts. Rank each implementation by block work per
thread plus reduction coordination and device occupancy, measure a small
shortlist and the original default, then select the fastest valid candidate.
"""

from dataclasses import replace
from math import ceil, isfinite, log2, prod

from ninetoothed.utils import device_scope, get_device_properties

_PASS = "ssa.triton.singleton_dot"
_ORIGINAL = "original"
_REDUCTION = "singleton_reduction"


class TunedSingletonDot:
    """Dispatch a selected handle, retuning when the public tensor signature changes."""

    def __init__(
        self,
        chosen,
        candidates,
        hardware,
        signature,
        output_index,
        resolve,
        *,
        cache_hit,
        search,
    ):
        from ninetoothed.compiler.runtime import dispatch_handles

        self.candidates = tuple(candidates)
        self.hardware = hardware
        self.cache_hit = cache_hit
        self.search = search
        self._output_index = output_index
        self._resolve = resolve
        self._selections = {signature: chosen}
        self._launch = dispatch_handles(self._select_handle)

    def __call__(self, *args, **kwargs):
        _validate_args(args, kwargs)

        with device_scope(args[0].device):
            return self._launch(*args)

    def _select_handle(self, args, kwargs):
        from ninetoothed.auto_tuner import AutoTuner

        _check_output(args, self._output_index)
        key = AutoTuner.argument_key(args, kwargs)

        if key not in self._selections:
            self._selections[key] = self._resolve(args).selected(*args)

        return self._selections[key]["handle"]

    def selected(self, *args, **kwargs):
        """Return the measured selection for a previously invoked signature."""
        from ninetoothed.auto_tuner import AutoTuner

        return self._selections.get(AutoTuner.argument_key(args, kwargs))


def autotune_singleton_dot(
    base_request,
    reduction_request,
    example_args,
    *,
    candidates_per_path=4,
    benchmark=None,
):
    """Measure a fixed shortlist and return its fastest valid implementation.

    Only the warp count and implementation vary; all other request options stay
    fixed. ``benchmark`` may override the compiler device timer and must return
    milliseconds. Custom timers do not share persistent selection results.
    Tuning uses separate output buffers and writes the public output once.
    """
    args = tuple(example_args)
    _validate_args(args, {})

    with device_scope(args[0].device):
        result = _tune(
            base_request, reduction_request, args, candidates_per_path, benchmark
        )
        result(*args)

        return result


def _tune(base_request, reduction_request, args, candidates_per_path, benchmark):
    from ninetoothed.auto_tuner import AutoTuner
    from ninetoothed.compiler.benchmark import benchmark_kernel
    from ninetoothed.compiler.cache import (
        CACHE_DIR,
        compilation_cache_key,
        read_manifest,
        stable_digest,
        write_manifest,
    )
    from ninetoothed.compiler.driver import DEFAULT_COMPILER

    persistent = benchmark is None
    measure = benchmark_kernel if persistent else benchmark

    if (
        isinstance(candidates_per_path, bool)
        or not isinstance(candidates_per_path, int)
        or candidates_per_path < 3
    ):
        raise ValueError("Candidates_per_path must be at least three.")

    if base_request.application is not reduction_request.application:
        raise ValueError(
            "Original and reduction requests must use the same application."
        )

    if _tensor_signatures(base_request) != _tensor_signatures(reduction_request):
        raise ValueError(
            "Original and reduction requests must have identical public tensors."
        )

    hardware = get_device_properties(args[0].device)

    if base_request.num_warps is not None:
        _legal_warps(hardware, requested=base_request.num_warps)

    requests = {
        _ORIGINAL: _with_pipeline(base_request, enable=False),
        _REDUCTION: _with_pipeline(reduction_request, enable=True),
    }
    base = DEFAULT_COMPILER.compile(replace(requests[_ORIGINAL], max_num_configs=1))
    base_warps = base.launch_plan.tuning_candidates[0]["num_warps"]
    legal = _legal_warps(hardware, requested=base_warps)
    reduction = DEFAULT_COMPILER.compile(
        replace(requests[_REDUCTION], num_warps=base_warps, max_num_configs=1)
    )

    if base.target.as_metadata() != reduction.target.as_metadata():
        raise ValueError("Original and reduction requests must target the same device.")

    abi = base.launch_abi

    if len(args) != len(abi.public_args) or len(abi.outputs) != 1:
        raise ValueError(
            "Singleton-dot tuning requires one output and a complete invocation."
        )

    output_index = abi.public_args.index(abi.outputs[0])

    if not args[output_index].is_floating_point():
        raise ValueError("Singleton-dot tuning requires a floating-point output.")

    _check_output(args, output_index)
    signature = AutoTuner.argument_key(args)
    enabled = (
        reduction.artifact.metadata.get("ssa_metadata", {})
        .get("schedule", {})
        .get("singleton_dot", {})
        .get("enabled", False)
    )
    probes = {_ORIGINAL: base, _REDUCTION: reduction}
    paths = (_ORIGINAL, _REDUCTION) if enabled else (_ORIGINAL,)
    shortlist = [(_ORIGINAL, base_warps)]

    models = {}

    for kind in paths:
        models[kind] = _warp_model(probes[kind], legal, hardware)
        chosen_warps = _candidate_warps(
            models[kind], legal, base_warps, candidates_per_path
        )
        shortlist.extend((kind, warp) for warp in chosen_warps)

    shortlist = tuple(dict.fromkeys(shortlist))
    compiled = {(_ORIGINAL, base_warps): base, (_REDUCTION, base_warps): reduction}
    candidates = {}

    def candidate_for(identity):
        if identity not in candidates:
            kind, warp = identity
            compilation = compiled.get(identity)

            if compilation is None:
                compilation = DEFAULT_COMPILER.compile(
                    replace(requests[kind], num_warps=warp, max_num_configs=1)
                )

            other = compilation.launch_abi

            if (other.public_args, other.outputs) != (abi.public_args, abi.outputs):
                raise ValueError(
                    "Original and reduction candidates must have the same public ABI."
                )

            candidates[identity] = {
                "kind": kind,
                "warps": warp,
                "handle": DEFAULT_COMPILER.materialize(compilation),
                "key": compilation_cache_key(compilation),
            }

        return candidates[identity]

    def result_for(chosen, search, cache_hit):
        return TunedSingletonDot(
            chosen,
            tuple(candidates.values()),
            hardware,
            signature,
            output_index,
            lambda new_args: _tune(
                base_request,
                reduction_request,
                new_args,
                candidates_per_path,
                benchmark,
            ),
            cache_hit=cache_hit,
            search=search,
        )

    cache_key = stable_digest(
        {
            "original": compilation_cache_key(base),
            "singleton_reduction": compilation_cache_key(reduction),
            "signature": signature,
            "hardware": hardware,
            "shortlist": shortlist,
        }
    )
    cache_path = CACHE_DIR / "singleton_dot_selection" / f"{cache_key}.json"
    saved = read_manifest(cache_path) if persistent else None

    if saved and saved.get("schema") == 2:
        identity = (saved.get("kind"), saved.get("warps"))

        if identity in shortlist:
            try:
                chosen = candidate_for(identity)
            except (RuntimeError, ValueError, OSError, TypeError):
                chosen = None

            if chosen is not None and chosen["key"] == saved.get("key"):
                return result_for(chosen, saved.get("search", {}), True)

    reference_args = _scratch_output(args, output_index)
    candidate_for((_ORIGINAL, base_warps))["handle"](*reference_args)
    search = {
        "parameters": legal,
        "model": models,
        "shortlist": shortlist,
        "scores": {},
        "failures": {},
    }
    measured = {}

    for identity in shortlist:
        label = f"{identity[0]}/{identity[1]}"

        try:
            candidate = candidate_for(identity)
            trial_args = _scratch_output(args, output_index)
            candidate["handle"](*trial_args)
            _check_close(trial_args[output_index], reference_args[output_index])
            timing = measure(candidate["handle"], trial_args, {})

            if not isfinite(timing) or timing < 0:
                raise ValueError(
                    "The benchmark must return a finite nonnegative timing."
                )

            measured[identity] = timing
            search["scores"][label] = timing
        except Exception as exc:  # noqa: BLE001
            search["failures"][label] = f"{type(exc).__name__}: {exc}"

    if not measured:
        raise RuntimeError(
            f"No valid singleton-dot candidate was found: {search['failures']}."
        )

    chosen = candidate_for(min(measured, key=measured.__getitem__))

    if persistent:
        write_manifest(
            cache_path,
            {
                "schema": 2,
                "kind": chosen["kind"],
                "warps": chosen["warps"],
                "key": chosen["key"],
                "search": search,
            },
        )

    return result_for(chosen, search, False)


def _with_pipeline(request, *, enable):
    from ninetoothed.compiler.passes import Pipeline, build, default_spec

    if request.pipeline is None and not enable:
        return request

    if isinstance(request.pipeline, Pipeline):
        raise ValueError(
            "Singleton-dot tuning requires a declarative pipeline; "
            "executable Pipeline instances contain pass objects that cannot "
            "be safely rewritten."
        )

    spec = build(
        request.pipeline if request.pipeline is not None else default_spec("triton"),
        backend="triton",
    ).spec

    passes = [name for name in spec.passes if name != _PASS]

    if enable:
        if "ssa.decompose_linalg" not in passes:
            raise ValueError("The pipeline must include ssa.decompose_linalg.")

        passes.insert(passes.index("ssa.decompose_linalg"), _PASS)

    return replace(request, pipeline=replace(spec, passes=tuple(passes)))


def _legal_warps(hardware, *, requested=None):
    """Bound new search candidates; reject requests only against known limits."""
    width = hardware["warp_size"]
    limits = [
        hardware.get(name) for name in ("max_threads_per_block", "max_threads_per_sm")
    ]
    known = [limit // width for limit in limits if limit is not None]
    hard_limit = min(known) if known else None

    if requested is not None and (
        isinstance(requested, bool)
        or not isinstance(requested, int)
        or requested < 1
        or requested & (requested - 1)
        or (hard_limit is not None and requested > hard_limit)
    ):
        raise ValueError("The original request must have one legal warp count.")

    if hard_limit is None:
        return (requested,) if requested is not None else ()

    if hard_limit < 1:
        raise ValueError("The device cannot run a one-warp block.")

    return tuple(1 << index for index in range(hard_limit.bit_length()))


def _candidate_warps(model, legal_warps, baseline, limit):
    """Retain the model optimum, low parallelism, baseline, and a middle point."""
    best = min(legal_warps, key=lambda warp: (model["scores"][warp], warp))
    selected = list(dict.fromkeys((best, min(legal_warps), baseline)))

    while len(selected) < min(limit, len(legal_warps)):
        middle = (log2(min(selected)) + log2(max(selected))) / 2
        remaining = set(legal_warps) - set(selected)
        selected.append(
            min(
                remaining,
                key=lambda warp: (
                    abs(log2(warp) - middle),
                    model["scores"][warp],
                    warp,
                ),
            )
        )

    return tuple(selected)


def _warp_model(compilation, legal_warps, hardware):
    """Rank FP32-word work, outer-reduction coordination, and program waves."""
    program = compilation.artifact.metadata["ssa"]
    operations = [op for block in program["blocks"] for op in block["operations"]]
    values = list(program["inputs"]) + [
        value for op in operations for value in op["results"]
    ]
    widths = {"float16": 2, "bfloat16": 2, "float64": 8}
    types = {value["name"]: value["type"] for value in values}

    def size(type_):
        return prod(int(str(axis)) for axis in type_["shape"])

    try:
        elements = max(
            (
                size(type_) * widths.get(type_["dtype"], 4) / 4
                for type_ in types.values()
                if type_["kind"] == "tensor"
            ),
            default=1,
        )
        outputs = 0

        for op in operations:
            if op["opcode"].startswith("reduce."):
                rank = len(types[op["operands"][0]]["shape"])

                if rank <= 1 or int(op["attrs"]["axis"]) % rank != rank - 1:
                    outputs = max(outputs, size(op["results"][0]["type"]))

        programs = prod(int(axis.render()) for axis in compilation.launch_plan.grid)
    except (TypeError, ValueError):
        return {"known_shape": False, "scores": {warp: 0 for warp in legal_warps}}

    width = hardware["warp_size"]
    sm_count = hardware.get("multiprocessor_count", 1)
    sm_threads = hardware.get("max_threads_per_sm")
    scores = {}

    for warps in legal_warps:
        threads = width * warps
        resident = max(1, sm_threads // threads) if sm_threads else 1
        waves = ceil(programs / max(1, sm_count * resident))
        scores[warps] = max(1, waves) * (
            ceil(elements / threads) + outputs * log2(warps)
        )

    return {
        "known_shape": True,
        "block_words": elements,
        "reduction_outputs": outputs,
        "programs": programs,
        "scores": scores,
    }


def _validate_args(args, kwargs):
    if (
        kwargs
        or not args
        or any(
            not all(
                hasattr(arg, name) for name in ("shape", "dtype", "device", "stride")
            )
            for arg in args
        )
    ):
        raise ValueError("Singleton-dot tuning requires positional tensor arguments.")

    if (
        any(arg.device != args[0].device for arg in args)
        or getattr(args[0].device, "type", None) == "cpu"
    ):
        raise ValueError(
            "Singleton-dot tuning requires tensors on one accelerator device."
        )


def _check_output(args, output_index):
    if output_index >= len(args):
        raise ValueError("Singleton-dot tuning requires a complete invocation.")

    output_storage = args[output_index].untyped_storage().data_ptr()

    if any(
        tensor.untyped_storage().data_ptr() == output_storage
        for index, tensor in enumerate(args)
        if index != output_index
    ):
        raise ValueError("Singleton-dot tuning does not support aliased outputs.")


def _tensor_signatures(request):
    return tuple(
        (tuple(map(str, tensor.shape)), str(tensor.dtype)) for tensor in request.tensors
    )


def _scratch_output(args, output_index):
    scratch = list(args)
    output = scratch[output_index]
    scratch[output_index] = output.new_empty_strided(output.shape, output.stride())

    return tuple(scratch)


def _check_close(actual, reference):
    from ninetoothed.dtype import normalize_dtype

    tolerance = {"float16": (0.01, 0.003), "bfloat16": (0.04, 0.02)}
    rtol, atol = tolerance.get(normalize_dtype(str(reference.dtype)), (2e-5, 5e-4))
    close = actual.float().isclose(
        reference.float(), rtol=rtol, atol=atol, equal_nan=True
    )

    if not bool(close.all()):
        raise ValueError("Candidate output differs from the original implementation.")


__all__ = ["TunedSingletonDot", "autotune_singleton_dot"]
