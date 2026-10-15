"""Opt-in, hardware-bounded selection of singleton-dot schedules.

The two requests may use different arrangements (for example an N=16 dot
tile versus an N=1 reduction tile), but must have the same public tensors.
The default compiler pipeline is unaffected.
"""

from dataclasses import replace
from functools import partial
from math import isinf, log2, prod, sqrt

_PASS = "ssa.triton.singleton_dot"


class TunedSingletonDot:
    """Callable selected from original and A09 kernels for one runtime signature."""

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
        self._chosen = chosen
        self.candidates = tuple(candidates)
        self.hardware = hardware
        self.cache_hit = cache_hit
        self.search = search
        self._signature = signature
        self._output_index = output_index
        self._resolve = resolve
        self._variants = {}

        import torch

        from ninetoothed.compiler.runtime import dispatch_handles

        self._launch = dispatch_handles(self._select_handle)

        if torch.cuda.device_count() > 1:
            dispatch = self._launch

            def launch(*args, **kwargs):
                device = args[0].device if args else None

                if (
                    device is not None
                    and device.type == "cuda"
                    and torch.cuda.current_device() != device.index
                ):
                    with torch.cuda.device(device):
                        return dispatch(*args, **kwargs)

                return dispatch(*args, **kwargs)

            self._launch = launch

    def __call__(self, *args, **kwargs):
        return self._launch(*args, **kwargs)

    def _select_handle(self, args, kwargs):
        if kwargs:
            raise ValueError("Singleton-dot tuning requires positional arguments.")

        from ninetoothed.auto_tuner import AutoTuner

        key = AutoTuner.argument_key(args, {})

        if len(args) > self._output_index:
            _check_output(args, self._output_index)

        if key != self._signature:
            if key not in self._variants:
                self._variants[key] = self._resolve(args)

            return self._variants[key].selected(*args)["handle"]

        return self._chosen["handle"]

    def selected(self, *args, **kwargs):
        """Return the selected candidate after a call with this signature."""
        from ninetoothed.auto_tuner import AutoTuner

        key = AutoTuner.argument_key(args, kwargs)

        if key != self._signature:
            variant = self._variants.get(key)

            return variant.selected(*args, **kwargs) if variant else None

        return self._chosen


def autotune_singleton_dot(base_request, a09_request, example_args, *, min_gain=0.03):
    """Tune original/A09 paths and warp counts using complete CUDA Graph calls.

    ``example_args`` is a full positional invocation, including its output.
    Tuning happens eagerly and writes that output once. A09 must beat the
    original path by a speedup increment of ``min_gain`` to be selected:
    ``original_time / candidate_time - 1 > min_gain``. For example, 100 to
    65 us is a 53.8% speedup increment and a 35% latency reduction. The result
    is cached for the tensor signature and device.
    """
    return _autotune_on_device(
        base_request, a09_request, example_args, min_gain=min_gain, run_final=True
    )


def _autotune_on_device(
    base_request, a09_request, example_args, *, min_gain, run_final
):
    import torch

    args = tuple(example_args)

    if not args or any(not isinstance(arg, torch.Tensor) for arg in args):
        raise ValueError("Example_args must contain the public tensor arguments.")

    device = args[0].device

    if device.type != "cuda" or any(arg.device != device for arg in args):
        raise ValueError("Singleton-dot tuning requires tensors on one CUDA device.")

    with torch.cuda.device(device):
        return _autotune_singleton_dot_impl(
            base_request, a09_request, args, min_gain=min_gain, run_final=run_final
        )


def _autotune_singleton_dot_impl(
    base_request, a09_request, args, *, min_gain, run_final
):
    import torch

    from ninetoothed.auto_tuner import AutoTuner, cuda_graph_benchmark
    from ninetoothed.backends.triton_singleton_dot import singleton_dot_shape
    from ninetoothed.compiler.cache import (
        CACHE_DIR,
        compilation_cache_key,
        read_manifest,
        stable_digest,
        write_manifest,
    )
    from ninetoothed.compiler.driver import DEFAULT_COMPILER

    if not 0 <= min_gain < 1:
        raise ValueError("Min_gain must be in [0, 1).")

    device = args[0].device

    if _tensor_signatures(base_request) != _tensor_signatures(a09_request):
        raise ValueError(
            "Original and A09 requests must have identical public tensors."
        )

    if base_request.application is not a09_request.application:
        raise ValueError("Original and A09 requests must use the same application.")

    props = torch.cuda.get_device_properties(device)
    warp_size = getattr(props, "warp_size", None)
    block_limit = getattr(props, "max_threads_per_block", None)
    sm_limit = getattr(props, "max_threads_per_multi_processor", None)
    hardware = {
        "name": props.name,
        "capability": (props.major, props.minor),
        "uuid": str(getattr(props, "uuid", device)),
        "multiprocessors": props.multi_processor_count,
        "warp_size": int(warp_size or 32),
        "max_threads_per_block": int(block_limit) if block_limit is not None else None,
        "max_threads_per_sm": int(sm_limit) if sm_limit is not None else None,
        "incomplete": any(
            value is None for value in (warp_size, block_limit, sm_limit)
        ),
    }
    requested_warps = base_request.num_warps
    legal_warps = _legal_warps(hardware, requested=requested_warps)

    base_request = _with_pipeline(base_request, enable=False)
    a09_request = _with_pipeline(a09_request, enable=True)
    signature = AutoTuner.argument_key(args, {})

    def make_candidate(kind, warp, compilation, key=None):
        return {
            "kind": kind,
            "warps": warp,
            "compilation": compilation,
            "handle": DEFAULT_COMPILER.materialize(compilation),
            "key": key or compilation_cache_key(compilation),
        }

    def result_for(chosen, candidates, output_index, *, cache_hit, search):
        return TunedSingletonDot(
            chosen,
            candidates,
            hardware,
            signature,
            output_index,
            lambda new_args: _autotune_on_device(
                base_request, a09_request, new_args, min_gain=min_gain, run_final=False
            ),
            cache_hit=cache_hit,
            search=search,
        )

    def restore(saved):
        selected = saved.get("selected", {})

        if saved.get("schema") != 1 or not isinstance(selected, dict):
            return None

        kind, warp = selected.get("kind"), selected.get("warps")

        output_index = saved.get("output_index")

        if (
            kind not in {"original", "a09"}
            or not isinstance(output_index, int)
            or not 0 <= output_index < len(args)
        ):
            return None

        try:
            _legal_warps(hardware, requested=warp)

            if warp is None:
                return None

            request = base_request if kind == "original" else a09_request

            if kind == "original" and saved.get("baseline"):
                candidate_request = replace(request, max_num_configs=1)
            else:
                candidate_request = replace(request, num_warps=warp, max_num_configs=1)

            compilation = DEFAULT_COMPILER.compile(candidate_request)

            abi = compilation.launch_abi

            if (
                len(args) != len(abi.public_args)
                or len(abi.outputs) != 1
                or abi.public_args.index(abi.outputs[0]) != output_index
                or compilation_cache_key(compilation) != selected.get("key")
                or not args[output_index].is_floating_point()
            ):
                return None

            chosen = make_candidate(kind, warp, compilation, selected["key"])

        except (RuntimeError, ValueError, OSError, TypeError):
            return None

        _check_output(args, output_index)
        result = result_for(
            chosen,
            (chosen,),
            output_index,
            cache_hit=True,
            search=dict(saved.get("search", {})),
        )

        if run_final:
            result(*args)

        return result

    quick_key = _quick_selection_key(
        base_request, a09_request, signature, hardware, min_gain, stable_digest
    )
    cache_path = (
        CACHE_DIR / "singleton_dot_selection" / f"{quick_key}.json"
        if quick_key is not None
        else None
    )

    if (
        cache_path is not None
        and (cached := restore(read_manifest(cache_path) or {})) is not None
    ):
        return cached

    base_probe = DEFAULT_COMPILER.compile(replace(base_request, max_num_configs=1))
    tuning_candidates = base_probe.launch_plan.tuning_candidates

    if not tuning_candidates:
        raise ValueError("The original request has no warp configuration.")

    base_warps = tuning_candidates[0]["num_warps"]

    legal_warps = _legal_warps(hardware, requested=base_warps)

    probe = DEFAULT_COMPILER.compile(
        replace(a09_request, num_warps=base_warps, max_num_configs=1)
    )

    if base_probe.target.as_metadata() != probe.target.as_metadata():
        raise ValueError("Original and A09 requests must target the same device.")

    values = {v.name: v for v in probe.kernel.ssa.inputs}
    operations = probe.kernel.ssa.blocks[0].operations
    values.update((v.name, v) for op in operations for v in op.results)
    dots = tuple(
        shape for op in operations if (shape := singleton_dot_shape(op, values))
    )
    enabled = bool(
        probe.artifact.metadata.get("ssa_metadata", {})
        .get("schedule", {})
        .get("singleton_dot", {})
        .get("enabled", False)
    )

    abi = base_probe.launch_abi

    if len(args) != len(abi.public_args) or len(abi.outputs) != 1:
        raise ValueError(
            "Singleton-dot tuning requires one output and a complete invocation."
        )

    if (probe.launch_abi.public_args, probe.launch_abi.outputs) != (
        abi.public_args,
        abi.outputs,
    ):
        raise ValueError("Original and A09 candidates must have the same public ABI.")

    output_index = abi.public_args.index(abi.outputs[0])
    output = args[output_index]

    if not output.is_floating_point():
        raise ValueError("Singleton-dot tuning requires a floating-point output.")

    _check_output(args, output_index)

    tolerance = _tolerance(output.dtype, torch)
    compiled = {("original", base_warps): base_probe, ("a09", base_warps): probe}
    candidates_by_id = {}

    def candidate_for(kind, warp):
        identity = (kind, warp)

        if identity not in candidates_by_id:
            compilation = compiled.get(identity)

            if compilation is None:
                request = base_request if kind == "original" else a09_request
                compilation = DEFAULT_COMPILER.compile(
                    replace(request, num_warps=warp, max_num_configs=1)
                )
                compiled[identity] = compilation

            other = compilation.launch_abi

            if (other.public_args, other.outputs) != (abi.public_args, abi.outputs):
                raise ValueError(
                    "Original and A09 candidates must have the same public ABI."
                )

            candidates_by_id[identity] = make_candidate(kind, warp, compilation)

        return candidates_by_id[identity]

    if cache_path is None:
        cache_identity = stable_digest(
            {
                "policy": "singleton-dot-model-v16",
                "original": compilation_cache_key(base_probe),
                "a09": compilation_cache_key(probe),
                "signature": signature,
                "hardware": hardware,
                "min_gain": min_gain,
                "tolerance": tolerance,
            }
        )
        cache_path = CACHE_DIR / "singleton_dot_selection" / f"{cache_identity}.json"
        cached = restore(read_manifest(cache_path) or {})

        if cached is not None:
            return cached

    reference = candidate_for("original", base_warps)["handle"]
    timing_namespace = stable_digest(
        {
            "policy": "singleton-dot-candidate-graph-v10",
            "hardware": hardware,
            "reference": compilation_cache_key(base_probe),
            "a09": compilation_cache_key(probe),
            "tolerance": tolerance,
        }
    )
    measured = {}
    graphs = {}
    search = {
        "initial": [],
        "expanded": [],
        "failures": {},
        "pass": probe.artifact.metadata.get("ssa_metadata", {})
        .get("schedule", {})
        .get("singleton_dot", {}),
    }

    def benchmark(function, call_args, call_kwargs):
        if call_kwargs:
            raise ValueError("Singleton-dot tuning requires positional arguments.")

        function = candidate_for(*function.args[0])["handle"]
        trial_args = _scratch_output(call_args, output_index, torch)

        if function is not reference:
            reference_args = _scratch_output(call_args, output_index, torch)
            reference(*reference_args)
            function(*trial_args)
            torch.cuda.synchronize(device)
            torch.testing.assert_close(
                trial_args[output_index], reference_args[output_index], **tolerance
            )

        return cuda_graph_benchmark(
            function, trial_args, {}, min_total_ms=8, graphs=graphs
        )

    identities = tuple(
        (kind, warp) for kind in ("original", "a09") for warp in legal_warps
    )

    def invoke(identity, *call_args):
        return candidate_for(*identity)["handle"](*call_args)

    timer = AutoTuner(
        tuple(partial(invoke, identity) for identity in identities),
        identities,
        benchmark=benchmark,
        cache_namespace=timing_namespace,
    )

    def strategy(measure):
        def score(kind, warp):
            identity = (kind, warp)

            if identity not in measured:
                index = identities.index(identity)
                measured[identity] = measure(index)

                if isinf(measured[identity]):
                    search["failures"][f"{kind}/{warp}"] = (
                        timer.failure(index, args)
                        or "benchmark returned an infinite timing"
                    )
            return measured[identity]

        features = _model_features(
            base_probe, probe, args, output_index, hardware, dots
        )
        ranked = {
            kind: _rank_warps(features[kind], kind, legal_warps, hardware)
            for kind in ("original", "a09")
        }
        initial = [("original", base_warps)]

        if enabled:
            best_a09 = ranked["a09"][0]
            initial.append(("a09", best_a09))
            neighbor = _adjacent_ranked(best_a09, ranked["a09"], legal_warps)

            if neighbor is not None:
                initial.append(("a09", neighbor))

        search["initial"] = [list(identity) for identity in initial]

        for kind, warp in initial:
            score(kind, warp)

        def a09_clearly_ahead():
            if not enabled or any(path["uncertain"] for path in features.values()):
                return False

            a09_times = [
                value
                for (kind, _), value in measured.items()
                if kind == "a09" and not isinf(value)
            ]
            original_times = [
                value
                for (kind, _), value in measured.items()
                if kind == "original" and not isinf(value)
            ]

            return (
                len(a09_times) >= 2
                and bool(original_times)
                and min(a09_times) * 1.10 < min(original_times)
            )

        if enabled and not a09_clearly_ahead():
            baseline_index = legal_warps.index(base_warps)
            first_window = legal_warps[max(0, baseline_index - 1) : baseline_index + 2]
            recommended = min(first_window, key=ranked["original"].index)

            if recommended != base_warps:
                search["expanded"].append(["original", recommended])
                score("original", recommended)

        def weighted(identity):
            return measured[identity] * (1 + min_gain if identity[0] == "a09" else 1)

        for kind in ("a09", "original") if enabled else ("original",):
            if kind == "original" and a09_clearly_ahead():
                continue

            if kind == "original" and not features[kind]["uncertain"]:
                tested_original = [
                    identity
                    for identity, value in measured.items()
                    if identity[0] == "original" and not isinf(value)
                ]

                if len(tested_original) >= 2:
                    best_original = min(tested_original, key=measured.__getitem__)

                    if (
                        best_original[1] != base_warps
                        and measured[best_original] * 1.03
                        >= measured[("original", base_warps)]
                    ):
                        continue

            while True:
                challenger = _next_expansion(
                    kind,
                    measured,
                    ranked[kind],
                    legal_warps,
                    uncertain=features[kind]["uncertain"],
                )

                if challenger is None:
                    break

                best_kind = min(
                    (
                        identity
                        for identity in measured
                        if identity[0] == kind and not isinf(measured[identity])
                    ),
                    key=weighted,
                )
                identity = (kind, challenger)
                search["expanded"].append(list(identity))
                score(*identity)

                if (
                    not features[kind]["uncertain"]
                    and not isinf(measured[identity])
                    and weighted(identity) >= weighted(best_kind) * 0.97
                ):
                    break

        valid = [identity for identity, value in measured.items() if not isinf(value)]

        if not valid:
            raise RuntimeError("No valid singleton-dot candidate was found.")

        chosen_id = min(valid, key=weighted)
        baseline_id = ("original", base_warps)

        if chosen_id != baseline_id:
            baseline_handle = candidates_by_id[baseline_id]["handle"]
            chosen_handle = candidate_for(*chosen_id)["handle"]
            trial_args = _scratch_output(args, output_index, torch)
            before = cuda_graph_benchmark(
                baseline_handle, trial_args, {}, graphs=graphs
            )
            trial = cuda_graph_benchmark(chosen_handle, trial_args, {}, graphs=graphs)
            after = cuda_graph_benchmark(baseline_handle, trial_args, {}, graphs=graphs)
            baseline_time = (before + after) / 2
            search["guard"] = {
                "original": baseline_time,
                "chosen": trial,
                "drift": abs(after / before - 1),
            }

            required_gain = min_gain if chosen_id[0] == "a09" else 0.0

            if (
                search["guard"]["drift"] > 0.03
                or trial * (1 + required_gain) >= baseline_time
            ):
                chosen_id = baseline_id

        search["features"] = features
        search["model_order"] = ranked

        return chosen_id

    chosen_id = timer.search(strategy, args)
    valid = [identity for identity, value in measured.items() if not isinf(value)]
    baseline_id = ("original", base_warps)
    chosen = candidate_for(*chosen_id)
    search["scores"] = {
        f"{kind}/{warp}": value
        for (kind, warp), value in measured.items()
        if not isinf(value)
    }
    write_manifest(
        cache_path,
        {
            "schema": 1,
            "selected": {
                "kind": chosen["kind"],
                "warps": chosen["warps"],
                "key": chosen["key"],
            },
            "baseline": chosen_id == baseline_id,
            "output_index": output_index,
            "search": search,
        },
    )

    result = result_for(
        chosen,
        tuple(candidate_for(*identity) for identity in valid),
        output_index,
        cache_hit=False,
        search=search,
    )

    if run_final:
        result(*args)

    return result


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

    limit = min(known + [4]) if any(value is None for value in limits) else hard_limit

    if limit < 1:
        raise ValueError("The device cannot run a one-warp block.")

    warps = {1 << index for index in range(limit.bit_length())}

    if requested is not None:
        warps.add(requested)
    return tuple(sorted(warps))


def _check_output(args, output_index):
    output = args[output_index]

    output_storage = output.untyped_storage().data_ptr()

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


def _quick_selection_key(
    base_request, a09_request, signature, hardware, min_gain, digest
):
    """Index a winner only when both requests have inspectable source."""
    from ninetoothed.compiler.cache import request_cache_key

    original, optimized = map(request_cache_key, (base_request, a09_request))

    if original is None or optimized is None:
        return None

    return digest(
        {
            "policy": "singleton-dot-quick-v3",
            "original": original,
            "a09": optimized,
            "signature": signature,
            "hardware": hardware,
            "min_gain": min_gain,
        }
    )


def _program_count(compilation, args):
    """Resolve the actual launch grid; unknown grids remain uncertain."""
    grid = compilation.launch_plan.grid

    if not grid:
        return None

    try:
        return prod(int(axis.render()) for axis in grid)
    except ValueError:
        pass

    import sympy

    from ninetoothed.compiler.runtime import _binding_value

    public = dict(zip(compilation.launch_abi.public_args, args))
    bindings = dict(compilation.request.specialization_values or {})
    candidates = compilation.launch_plan.tuning_candidates

    if candidates:
        bindings.update(candidates[0].get("meta_parameters", {}))

    try:
        for binding in compilation.launch_abi.kernel_args:
            if binding.kind in {"shape", "stride", "meta", "constexpr", "scalar"}:
                value = bindings.get(binding.name)

                if value is None:
                    value = _binding_value(binding, public)

                if isinstance(value, (bool, int, float)):
                    bindings[binding.name] = value

        bindings["ceildiv"] = lambda a, b: (a + b - 1) // b
        dimensions = [
            sympy.sympify(sympy.sympify(axis.render(), locals=bindings))
            for axis in grid
        ]

        if any(
            value.free_symbols or value.is_integer is not True for value in dimensions
        ):
            return None
        return prod(map(int, dimensions))
    except (
        KeyError,
        TypeError,
        ValueError,
        SyntaxError,
        ZeroDivisionError,
        sympy.SympifyError,
    ):
        return None


def _model_features(base, a09, args, output_index, hardware, dots):
    """Extract warp-independent proxies from emitted SSA and actual launch grids."""
    from ninetoothed.dtype import normalize_dtype

    widths = {
        "bool": 1,
        "int8": 1,
        "uint8": 1,
        "int16": 2,
        "uint16": 2,
        "int32": 4,
        "uint32": 4,
        "int64": 8,
        "uint64": 8,
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
        "float64": 8,
    }

    logical_work = sum(m * k * n for m, k, n in dots)
    inputs = [arg for index, arg in enumerate(args) if index != output_index]
    contiguous = sum(arg.ndim == 0 or arg.stride(-1) == 1 for arg in inputs)
    dtype_bytes = max((arg.element_size() for arg in inputs), default=1)

    def path_features(compilation, kind):
        program = compilation.artifact.metadata["ssa"]
        values = {value["name"]: value for value in program["inputs"]}
        operations = tuple(
            op for block in program["blocks"] for op in block["operations"]
        )
        values.update((v["name"], v) for op in operations for v in op["results"])
        sizes = {}
        unknown = False

        for name, value in values.items():
            type_ = value["type"]
            width = widths.get(normalize_dtype(type_["dtype"]))

            try:
                count = prod(int(str(axis)) for axis in type_["shape"])
            except (TypeError, ValueError):
                count, unknown = 0, True

            if type_["kind"] == "tensor" and width is None:
                unknown = True

            sizes[name] = count * (width or dtype_bytes)

        last_use = {
            name: max(
                (
                    index
                    for index, op in enumerate(operations)
                    if name in op["operands"]
                ),
                default=-1,
            )
            for name in values
        }
        live = {value["name"] for value in program["inputs"]}
        peak_live = sum(sizes[name] for name in live)

        for index, op in enumerate(operations):
            live.update(value["name"] for value in op["results"])
            peak_live = max(peak_live, sum(sizes[name] for name in live))
            live.difference_update(
                name for name in tuple(live) if last_use[name] <= index
            )

        actual_work = 0

        for op in operations:
            if op["opcode"] not in {"linalg.dot", "linalg.matmul"}:
                continue

            try:
                m, k = map(int, values[op["operands"][0]]["type"]["shape"])
                _, n = map(int, values[op["operands"][1]]["type"]["shape"])
                actual_work += m * k * n
            except (KeyError, TypeError, ValueError):
                unknown = True

        work = actual_work if kind == "original" and actual_work else logical_work
        programs = _program_count(compilation, args)

        return {
            "work": work,
            "dots": len(dots),
            "matvec": any(n == 1 and m > 1 for m, _, n in dots),
            "padding": actual_work / max(1, logical_work)
            if kind == "original"
            else 1.0,
            "peak_live_bytes": max(peak_live, work * dtype_bytes),
            "ir_operations": len(operations),
            "operations": sum(
                op["opcode"].startswith(("arith.", "reduce.", "math."))
                for op in operations
            ),
            "programs": programs or 1,
            "contiguous_fraction": contiguous / max(1, len(inputs)),
            "dtype_bytes": dtype_bytes,
            "uncertain": (
                unknown
                or not dots
                or (kind == "original" and not actual_work)
                or contiguous != len(inputs)
                or not programs
                or any(op["regions"] for op in operations)
                or hardware["incomplete"]
            ),
            "sms": hardware["multiprocessors"],
        }

    return {
        "original": path_features(base, "original"),
        "a09": path_features(a09, "a09"),
    }


def _rank_warps(features, kind, legal_warps, hardware):
    """Rank legal warps without compiling any candidate."""
    if kind == "original":
        work_per_lane = 48
    elif features["matvec"]:
        work_per_lane = 32
    else:
        work_per_lane = 128

    fused_factor = sqrt(max(1, features["dots"]))

    if kind == "a09":
        fused_factor *= sqrt(max(1, features["operations"] / 2))

    grid_factor = max(1.0, features["programs"] / max(1, features["sms"]))
    coalescing = 0.75 + 0.25 * features["contiguous_fraction"]
    ideal = max(
        1.0,
        features["work"]
        / (hardware["warp_size"] * work_per_lane * fused_factor * grid_factor)
        / coalescing,
    )

    def estimate(warps):
        per_lane_bytes = features["peak_live_bytes"] / (warps * hardware["warp_size"])
        pressure = 0.02 * max(0, per_lane_bytes / 512 - 1)
        cross_warp = (
            0.08 if kind == "a09" and not features["matvec"] else 0.02
        ) * log2(warps)
        padding = (
            0.04 * log2(max(1, features["padding"])) * log2(warps)
            if kind == "original"
            else 0
        )

        return abs(log2(warps / ideal)) + pressure + cross_warp + padding

    return tuple(sorted(legal_warps, key=lambda warps: (estimate(warps), warps)))


def _adjacent_ranked(center, ranked, legal_warps, *, tested=()):
    index = legal_warps.index(center)
    options = [
        legal_warps[neighbor]
        for neighbor in (index - 1, index + 1)
        if 0 <= neighbor < len(legal_warps) and legal_warps[neighbor] not in tested
    ]

    return min(options, key=ranked.index) if options else None


def _next_expansion(kind, measured, ranked, legal_warps, *, uncertain):
    same_kind = [
        identity
        for identity, value in measured.items()
        if identity[0] == kind and not isinf(value)
    ]

    if not same_kind:
        return None

    best_kind = min(same_kind, key=measured.__getitem__)
    tested = {warp for current_kind, warp in measured if current_kind == kind}
    adjacent = _adjacent_ranked(
        best_kind[1],
        ranked,
        legal_warps,
        tested=tested,
    )

    if adjacent is not None or not uncertain:
        return adjacent

    frontier = set()

    for warp in tested:
        index = legal_warps.index(warp)

        for neighbor in (index - 1, index + 1):
            if 0 <= neighbor < len(legal_warps):
                frontier.add(legal_warps[neighbor])

    frontier.difference_update(tested)

    return min(frontier, key=ranked.index) if frontier else None


def _scratch_output(args, output_index, torch):
    scratch = list(args)
    output = scratch[output_index]
    scratch[output_index] = torch.empty_strided(
        output.shape, output.stride(), dtype=output.dtype, device=output.device
    )

    return tuple(scratch)


def _tolerance(dtype, torch):
    if dtype == torch.float16:
        return {"rtol": 0.01, "atol": 0.003}

    if dtype == torch.bfloat16:
        return {"rtol": 0.04, "atol": 0.02}

    return {"rtol": 2e-5, "atol": 5e-4}


__all__ = ["TunedSingletonDot", "autotune_singleton_dot"]
