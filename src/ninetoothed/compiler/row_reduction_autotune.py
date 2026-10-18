"""Measure physical row partitions without changing the logical arrangement."""

from dataclasses import replace
from math import ceil, isfinite, isqrt, prod

_PASS = "ssa.group_reduction_rows"


def autotune_row_reductions(request, example_args, *, max_candidates=4):
    """Return an AutoTuner over the original and a small row-partition shortlist.

    Only physical rows per execution group vary. Logical shapes, arithmetic,
    layout and other passes stay fixed. Each candidate is checked against the
    original using private output storage before the shared timer measures it.
    The returned tuner is bound to one device and uses ordinary Handle guards.
    """
    from ninetoothed.auto_tuner import AutoTuner
    from ninetoothed.compiler.benchmark import benchmark_kernel
    from ninetoothed.compiler.cache import compilation_cache_key
    from ninetoothed.compiler.driver import DEFAULT_COMPILER
    from ninetoothed.utils import device_scope, get_device_properties

    args = tuple(example_args)

    if (
        isinstance(max_candidates, bool)
        or not isinstance(max_candidates, int)
        or max_candidates < 2
    ):
        raise ValueError("Max_candidates must be at least two, including the original.")

    if (
        not args
        or any(not hasattr(arg, "device") for arg in args)
        or args[0].device.type == "cpu"
    ):
        raise ValueError("Row tuning requires accelerator tensor arguments.")

    device = args[0].device

    with device_scope(device):
        request = _request(request, enabled=False)
        base = DEFAULT_COMPILER.compile(request)

    abi = base.launch_abi

    if len(abi.outputs) != 1 or len(args) != len(abi.public_args):
        raise ValueError("Row tuning requires one output and a complete invocation.")

    stores = [
        op for op in base.kernel.ssa.blocks[0].operations if op.opcode == "mem.store"
    ]

    if (
        any(
            binding.access != ("write" if binding.source == abi.outputs[0] else "read")
            for binding in abi.kernel_args
            if binding.kind in {"tensor", "jagged_values", "jagged_offsets"}
            or binding.access is not None
        )
        or len(stores) != 1
        or set(stores[0].attrs) - {"target"}
    ):
        raise ValueError("Row tuning requires one whole, write-only output store.")

    output_index = abi.public_args.index(abi.outputs[0])
    arg_count = len(args)

    def validate(values, kwargs):
        if (
            kwargs
            or len(values) != arg_count
            or any(value.device != device for value in values)
        ):
            raise ValueError(
                "Row tuning requires positional tensors on its original device."
            )

        output = values[output_index]

        if not output.is_floating_point() or any(
            value.untyped_storage().data_ptr() == output.untyped_storage().data_ptr()
            for index, value in enumerate(values)
            if index != output_index
        ):
            raise ValueError("Row tuning requires an unaliased floating-point output.")

    validate(args, {})

    with device_scope(device):
        hardware = get_device_properties(device)
        original = DEFAULT_COMPILER.materialize(base)
        handles, keys = [original], [compilation_cache_key(base)]
        search = {"rows": {}, "failures": {}}
        probe = DEFAULT_COMPILER.compile(_request(request, enabled=True))
        group = probe.artifact.metadata["ssa_schedule"].get("row_group", {})

        if group.get("rows", 1) > 1:
            for rows in _shortlist(probe, hardware, max_candidates - 1):
                try:
                    candidate = (
                        probe
                        if rows == group["rows"]
                        else DEFAULT_COMPILER.compile(
                            _request(request, enabled=True, rows=rows)
                        )
                    )
                    selected = candidate.artifact.metadata["ssa_schedule"]["row_group"]

                    if not selected.get("enabled"):
                        search["failures"][rows] = selected["reason"]
                        continue

                    key = compilation_cache_key(candidate)

                    if key not in keys:
                        handles.append(DEFAULT_COMPILER.materialize(candidate))
                        keys.append(key)
                        search["rows"][key] = rows
                except Exception as error:  # noqa: BLE001
                    search["failures"][rows] = str(error)

        def measure(handle, values, kwargs):
            reference = list(values)
            trial = list(values)
            output = values[output_index]
            reference[output_index] = output.new_empty_strided(
                output.shape, output.stride()
            )
            trial[output_index] = output.new_empty_strided(
                output.shape, output.stride()
            )
            original(*reference)
            handle(*trial)
            tolerance = {"float16": (0.01, 0.003), "bfloat16": (0.04, 0.02)}
            rtol, atol = tolerance.get(str(output.dtype).split(".")[-1], (2e-5, 5e-4))

            if not bool(
                trial[output_index]
                .isclose(reference[output_index], rtol=rtol, atol=atol, equal_nan=True)
                .all()
            ):
                raise ValueError("Candidate output differs from the original.")

            timing = benchmark_kernel(handle, tuple(trial), kwargs)

            if not isfinite(timing) or timing < 0:
                raise ValueError(
                    "The benchmark must return finite nonnegative milliseconds."
                )

            return timing

        class RowTuner(AutoTuner):
            def __call__(self, *values, **kwargs):
                with device_scope(device):
                    return super().__call__(*values, **kwargs)

        tuner = RowTuner(
            handles,
            keys,
            benchmark=measure,
            cache_namespace={"row_group": hardware},
            validator=validate,
        )
        tuner.search = search
        tuner(*args)

        return tuner


def _request(request, *, enabled, rows=None):
    from ninetoothed.compiler.driver import resolve_compile_target
    from ninetoothed.compiler.passes import Pipeline, build, default_spec

    target = resolve_compile_target(
        request.backend, platform=request.platform, compute_arch=request.compute_arch
    )
    request = replace(
        request,
        backend=target.backend,
        platform=target.platform.name,
        compute_arch=target.compute_arch,
    )

    if isinstance(request.pipeline, Pipeline):
        raise ValueError("Row tuning requires a declarative pipeline.")

    spec = build(
        request.pipeline or default_spec(request.backend), backend=request.backend
    ).spec
    passes = tuple(name for name in spec.passes if name != _PASS)
    options = {
        name: value for name, value in spec.pass_options.items() if name != _PASS
    }
    overrides = {
        name: value
        for name, value in (request.pass_options or {}).items()
        if name != _PASS
    }

    if enabled:
        at = passes.index("ssa.select_schedule") + 1
        passes = (*passes[:at], _PASS, *passes[at:])
        overrides[_PASS] = {} if rows is None else {"rows": rows}

    return replace(
        request,
        pipeline=replace(spec, passes=passes, pass_options=options),
        pass_options=overrides,
        max_num_configs=1,
    )


def _shortlist(compilation, hardware, limit):
    """Retain full and single-row partitions, then rank intermediate divisors."""
    schedule = compilation.artifact.metadata["ssa_schedule"]
    rows = schedule["row_group"]["rows"]
    divisors = {
        factor
        for divisor in range(1, isqrt(rows) + 1)
        if rows % divisor == 0
        for factor in (divisor, rows // divisor)
    }
    selected = list(dict.fromkeys((rows, 1)))[:limit]
    warp_size = hardware["warp_size"]
    threads = (
        int(compilation.launch_plan.block[0].render())
        if compilation.launch_plan.block
        else warp_size * compilation.launch_plan.tuning_candidates[0]["num_warps"]
    )
    resident = max(1, hardware.get("max_threads_per_sm", threads) // threads)
    capacity = hardware.get("multiprocessor_count", 1) * resident

    try:
        width = int(schedule["reduction"]["extent"])
        tiles = prod(int(axis) for axis in schedule["reduction"]["program_shape"])
        scores = {
            size: max(1, ceil(tiles * rows / size / capacity))
            * ceil(size * width / threads)
            for size in divisors
        }
    except (TypeError, ValueError):
        scores = {size: abs(size - isqrt(rows)) for size in divisors}

    return tuple(
        (
            selected
            + sorted(divisors - set(selected), key=lambda size: (scores[size], size))
        )[:limit]
    )
