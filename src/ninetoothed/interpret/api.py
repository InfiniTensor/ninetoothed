"""Public entry points of the NineToothed CPU reference interpreter.

:func:`interpret` runs the normal lowering chain and executes the resulting
``ssa.Program`` with NumPy.  :func:`interpret_program` takes a program that is
already lowered, so pass pipelines and backends can share one executor.
"""

import inspect
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from ninetoothed import naming
from ninetoothed.frontend.layout import tensor_specs
from ninetoothed.frontend.python import from_application
from ninetoothed.symbol import Symbol
from ninetoothed.tensor import Tensor

from .dtypes import resolve_dtype
from .errors import (
    MissingSymbolError,
    ProgramDomainError,
)
from .interpreter import Interpreter, derive_program_domain, domain_size
from .memory import CPUMemory, build_tensor_runtime, tile_access_map
from .trace import Tracer

#: Fallback value for a constexpr symbol that the caller did not bind.  It
#: matches the default the compiler driver picks for tuning parameters.
DEFAULT_CONSTEXPR_VALUE = 256


@dataclass
class Interpretation:
    """The result of one CPU interpretation.

    :param outputs: ``{tensor_name: numpy.ndarray}`` for every stored tensor.
    :param program: The executed ``ssa.Program``.
    :param symbols: The resolved symbol values.
    :param tensor_specs: The tensor specs produced by the arrangement.
    :param launch_shape: The shape of the program-instance grid.
    :param trace: The recorded trace events (empty when tracing is off).
    :param memory: The CPU memory holding the backing buffers.
    :param arrangement: The arrangement function, when one was used.
    :param application: The application function.
    :param inputs: The runtime arrays, in parameter order.
    :param tensors: The symbolic tensors the arrangement was built from.
    :param pipeline: The pass pipeline the program was lowered with, when one was.
    :param seed: The random seed the caller recorded for the inputs, when one was.
    """

    outputs: dict = field(default_factory=dict)
    program: Any = None
    symbols: dict = field(default_factory=dict)
    tensor_specs: tuple = ()
    launch_shape: tuple = ()
    trace: tuple = ()
    memory: Any = None
    arrangement: Any = None
    application: Any = None
    inputs: tuple = ()
    tensors: tuple = ()
    pipeline: Any = None
    seed: int | None = None

    def output(self, name=None):
        """Return one output tensor.

        :param name: The tensor name; defaults to the primary output.
        :return: The ``numpy.ndarray`` stored by the program.
        """
        if name is None:
            if not self.outputs:
                raise ProgramDomainError("The interpretation produced no outputs.")

            name = next(iter(self.outputs))

        try:
            return self.outputs[name]
        except KeyError as exc:
            raise ProgramDomainError(
                f"Unknown output `{name}`; available outputs: "
                f"{', '.join(self.outputs) or '<none>'}."
            ) from exc

    def trace_events(self, *, program_id=None, opcode=None):
        """Return trace events matching an optional filter."""
        events = self.trace

        if program_id is not None:
            events = tuple(event for event in events if event.program_id == program_id)

        if opcode is not None:
            events = tuple(event for event in events if event.opcode == opcode)

        return events

    def render_trace(self, *, limit=None):
        """Render the recorded trace as readable text."""
        events = self.trace if limit is None else self.trace[:limit]
        lines = []

        for event in events:
            indent = "  " * event.depth
            operands = ", ".join(repr(item) for item in event.operands)
            results = ", ".join(repr(item) for item in event.results)
            mask = f" mask={event.mask}" if event.mask else ""
            lines.append(
                f"[p{event.program_id}]{indent} {event.location} {event.opcode}"
                f"({operands}){mask}" + (f" -> {results}" if results else "")
            )

        return "\n".join(lines)


def interpret(
    arrangement,
    application,
    tensors=(),
    inputs=None,
    *,
    symbols=None,
    scalars=None,
    backend=None,
    platform=None,
    compute_arch=None,
    pipeline=None,
    pass_options=None,
    pass_registry=None,
    program_ids=None,
    trace=None,
    tracer=None,
    seed=None,
    strict_domain=True,
):
    """Execute a NineToothed application on the CPU with NumPy.

    :param arrangement: The arrangement function, or ``None`` when the
        application carries ``Tensor`` annotations.
    :param application: The application function.
    :param tensors: Either runtime arrays or symbolic ``Tensor`` descriptors.
    :param inputs: Runtime arrays, in parameter order or keyed by parameter name.
        Defaults to ``tensors`` when those are already arrays.
    :param symbols: Values for symbolic dimensions and constexpr parameters.
    :param scalars: Values for non-tensor program inputs.
    :param backend: The backend whose pass pipeline should be applied.
    :param platform: The platform used when resolving the pass pipeline.
    :param compute_arch: The compute architecture used for the pass pipeline.
    :param pipeline: An explicit SSA pass pipeline (``None`` keeps the program
        exactly as the frontend produced it).
    :param pass_options: Per-pass options for the pipeline.
    :param pass_registry: A pass registry that extends the built-in one, so
        locally defined passes can be named in ``pipeline``.
    :param program_ids: Restrict execution to these program instances.
    :param trace: A :class:`~ninetoothed.interpret.trace.Tracer`, a mapping of
        tracer keyword arguments, or ``None``.
    :param tracer: An explicit tracer instance (alias of ``trace``).
    :param seed: The random seed the inputs were generated from, if any.  It is
        recorded only, so a failing run can be replayed from the seed.
    :param strict_domain: Validate that every tiled tensor shares the launch domain.
    :return: An :class:`Interpretation`.
    """
    params = tuple(inspect.signature(application).parameters)
    symbolic, buffers, scalars = _normalize_inputs(params, tensors, inputs, scalars)
    arranged = _arrange(arrangement, application, params, symbolic, buffers)
    specs = tensor_specs(params, arranged)
    resolved = resolve_symbols(
        specs,
        buffers,
        symbols,
        arranged=arranged,
        extra_aliases=arrangement_symbol_aliases(arrangement),
    )

    program = from_application(
        application,
        specs,
        kind=getattr(application, "__name__", "application"),
        strict=True,
    )

    if program is None:
        name = getattr(application, "__name__", "application")
        message = (
            f"Cannot lower `{name}` to `ssa.Program`; the interpreter only "
            "executes the SSA path."
        )

        if name in {"<lambda>", "<module>"}:
            message += (
                " The frontend lowers an application by reading its source, so it "
                "must be a module-level function; a `lambda` or a function defined "
                "in a REPL cannot be lowered."
            )

        raise ProgramDomainError(message)

    if pipeline is not None:
        program = apply_pipeline(
            program,
            specs,
            pipeline,
            backend=backend,
            platform=platform,
            compute_arch=compute_arch,
            pass_options=pass_options,
            pass_registry=pass_registry,
            symbols=resolved,
        )

    tracer = _resolve_tracer(trace, tracer)
    buffer_map = _buffer_map(program, buffers)
    memory = _build_memory(program, buffer_map, resolved, scalars)
    launch_shape = derive_program_domain(program, memory)

    if strict_domain:
        _validate_domain(program, memory, launch_shape)

    interpreter = Interpreter(
        program=program,
        memory=memory,
        symbols=resolved,
        total=domain_size(launch_shape),
        tracer=tracer,
        scalars=scalars,
    )
    outputs = interpreter.run(program_ids=program_ids)

    return Interpretation(
        outputs=outputs,
        program=program,
        symbols=resolved,
        tensor_specs=specs,
        launch_shape=launch_shape,
        trace=tuple(tracer.events) if tracer is not None else (),
        memory=memory,
        arrangement=arrangement,
        application=application,
        inputs=tuple(buffers),
        tensors=symbolic,
        pipeline=pipeline,
        seed=seed,
    )


def interpret_program(
    program,
    buffers,
    *,
    symbols=None,
    scalars=None,
    total=None,
    tracer=None,
    strict_domain=True,
):
    """Execute an already lowered ``ssa.Program`` on the CPU.

    :param program: The SSA program to execute.
    :param buffers: ``{tensor_name: numpy.ndarray}`` for the program inputs.
    :param symbols: Extra symbol values (shapes, strides and constexprs are
        derived automatically).
    :param scalars: Values for non-tensor program inputs.
    :param total: Override the derived number of program instances.
    :param tracer: An optional tracer.
    :param strict_domain: Validate that every tiled tensor shares the launch domain.
    :return: An :class:`Interpretation`.
    """
    resolved = resolve_program_symbols(program, buffers, symbols)
    memory = _build_memory(program, buffers, resolved, scalars or {})
    launch_shape = derive_program_domain(program, memory)

    if strict_domain:
        _validate_domain(program, memory, launch_shape)

    instance_count = domain_size(launch_shape) if total is None else int(total)
    interpreter = Interpreter(
        program=program,
        memory=memory,
        symbols=resolved,
        total=instance_count,
        tracer=tracer,
        scalars=scalars or {},
    )
    outputs = interpreter.run()

    return Interpretation(
        outputs=outputs,
        program=program,
        symbols=resolved,
        launch_shape=launch_shape,
        trace=tuple(tracer.events) if tracer is not None else (),
        memory=memory,
        inputs=tuple(
            buffers.get(value.name)
            for value in program.inputs
            if buffers.get(value.name) is not None
        ),
    )


def access_map(interpretation, name, *, program_ids=None, level=None):
    """Return the resolved access map of one tensor, per program instance.

    It answers which source element each element of the tile reads or writes,
    and whether the access is masked.  The result can be compared directly
    against :func:`ninetoothed.eval._eval`, the compiler's own view of the same
    mapping.

    :param interpretation: An :class:`Interpretation`.
    :param name: The tensor name.
    :param program_ids: Restrict to these program instances.
    :param level: The dtype level to inspect; defaults to the addressable one.
    :return: ``{program_id: (offsets, mask)}`` where both are NumPy arrays
        covering the full tile hierarchy of that instance.
    """
    tensor = interpretation.memory.get(name)
    total = domain_size(interpretation.launch_shape)
    instances = (
        range(total) if program_ids is None else tuple(int(i) for i in program_ids)
    )
    result = {}

    for program_id in instances:
        context = interpretation.memory.context(program_id)

        if level is None:
            result[program_id] = tile_access_map(tensor, context)
        else:
            view = tensor.root_view(int(level))
            result[program_id] = context.evaluate_access(view)

    return result


def _stacked_access(interpretation, name, level, index):
    """Stack one component of the access map over the launch grid.

    The result is shaped ``view_shape + tile_shape``, matching the layout
    :func:`ninetoothed.eval._eval` reports.
    """
    resolved = access_map(interpretation, name, level=level)

    if not resolved:
        raise ProgramDomainError(f"Tensor `{name}` has no program instance to inspect.")

    ordered = [resolved[program_id][index] for program_id in sorted(resolved)]
    stacked = np.stack([np.asarray(item) for item in ordered])
    tensor = interpretation.memory.get(name)
    view_shape = tuple(int(dim) for dim in tensor.view_shape)
    tile_shape = tuple(int(dim) for dim in np.shape(ordered[0]))

    if view_shape and stacked.shape[0] == int(np.prod(view_shape)):
        return stacked.reshape(view_shape + tile_shape)

    return stacked


def access_offsets(interpretation, name, *, level=None):
    """Return the resolved source offsets of a tensor across the launch grid.

    Masked-out elements still carry their unmasked offset; pair this with
    :func:`access_mask` (or fold them with ``np.where(mask, offsets, -1)``) to
    get the mapping :func:`ninetoothed.eval._eval` reports.

    :param interpretation: An :class:`Interpretation`.
    :param name: The tensor name.
    :param level: The dtype level to inspect.
    :return: A NumPy array of shape ``launch_shape + tile_shape``.
    """
    return _stacked_access(interpretation, name, level, 0)


def access_mask(interpretation, name, *, level=None):
    """Return the bounds predicate of a tensor across the launch grid.

    :param interpretation: An :class:`Interpretation`.
    :param name: The tensor name.
    :param level: The dtype level to inspect.
    :return: A boolean NumPy array of shape ``launch_shape + tile_shape``.
    """
    return _stacked_access(interpretation, name, level, 1)


def random_inputs(specs, *, seed, float_range=(-1.0, 1.0), integer_range=(0, 8)):
    """Build deterministic random input arrays from a seed.

    Generate the inputs here, pass the same ``seed`` to :func:`interpret`, and a
    reproduction rendered by
    :func:`ninetoothed.interpret.diff.render_reproduction` will carry everything
    needed to rebuild them.

    :param specs: A sequence of ``(shape, dtype)`` pairs.  A bare shape is read as
        ``float32``.  Both a single pair and a sequence of pairs are accepted.
    :param seed: The seed for :func:`numpy.random.default_rng`.
    :param float_range: ``(low, high)`` for floating point dtypes.
    :param integer_range: ``(low, high)`` for integer and boolean dtypes.
    :return: A tuple of NumPy arrays, one per spec.
    :raises ProgramDomainError: When a dtype cannot be generated.
    """
    if not specs:
        return ()

    if (
        len(specs) == 2
        and isinstance(specs[0], (tuple, list))
        and isinstance(specs[1], (str, type, np.dtype))
    ):
        specs = (specs,)

    rng = np.random.default_rng(seed)
    arrays = []

    for spec in specs:
        if isinstance(spec, (tuple, list)) and len(spec) == 2:
            shape, dtype = spec
        else:
            shape, dtype = spec, "float32"

        try:
            numpy_dtype = np.dtype(resolve_dtype(str(dtype)))
        except (TypeError, ValueError) as error:
            raise ProgramDomainError(
                f"Cannot generate random inputs for dtype `{dtype}`."
            ) from error

        if numpy_dtype.kind == "f":
            arrays.append(
                rng.uniform(*float_range, size=tuple(shape)).astype(numpy_dtype)
            )
        elif numpy_dtype.kind == "b":
            arrays.append(rng.integers(0, 2, size=tuple(shape)).astype(numpy_dtype))
        elif numpy_dtype.kind in "iu":
            low, high = integer_range
            arrays.append(
                rng.integers(low, high, size=tuple(shape)).astype(numpy_dtype)
            )
        else:
            raise ProgramDomainError(
                f"Cannot generate random inputs for dtype `{dtype}`."
            )

    return tuple(arrays)


def apply_pipeline(
    program,
    specs,
    pipeline,
    *,
    backend=None,
    platform=None,
    compute_arch=None,
    pass_options=None,
    pass_registry=None,
    symbols=None,
):
    """Run an SSA pass pipeline on a program.

    :param program: The SSA program.
    :param specs: The tensor specs used by the passes.
    :param pipeline: A pipeline name list, spec, or ``Pipeline`` instance.
    :param backend: The backend used to resolve backend-specific passes.
    :param platform: The platform used to resolve the target profile.
    :param compute_arch: The compute architecture used for the target profile.
    :param pass_options: Per-pass options.
    :param pass_registry: A pass registry that extends the built-in one.
    :param symbols: Extra symbol values used for specialization.
    :return: The lowered program.
    """
    from ninetoothed.compiler.passes import lower_for_target

    return lower_for_target(
        program,
        backend=backend,
        platform=platform,
        compute_arch=compute_arch,
        tensors=specs,
        pass_pipeline=pipeline,
        pass_options=pass_options,
        pass_registry=pass_registry,
    )


# -- input normalization ---------------------------------------------------


def _normalize_inputs(params, tensors, inputs, scalars):
    tensors = tuple(tensors)
    scalars = dict(scalars or {})

    if inputs is None:
        if all(isinstance(item, np.ndarray) for item in tensors):
            buffers = tuple(tensors)
            symbolic = tuple(Tensor(array.ndim) for array in buffers)
        else:
            raise ProgramDomainError(
                "Runtime arrays are required: pass them as `tensors` or through "
                "the `inputs` argument."
            )
    elif isinstance(inputs, Mapping):
        buffers = tuple(inputs[name] for name in params)
        symbolic = (
            tuple(tensors)
            if tensors
            else tuple(Tensor(np.asarray(array).ndim) for array in buffers)
        )
    else:
        buffers = tuple(inputs)
        symbolic = (
            tuple(tensors)
            if tensors
            else tuple(Tensor(np.asarray(array).ndim) for array in buffers)
        )

    if len(buffers) != len(params):
        raise ProgramDomainError(
            f"Expected {len(params)} runtime tensor(s) for "
            f"`{', '.join(params)}`; got {len(buffers)}."
        )

    if len(symbolic) != len(params):
        raise ProgramDomainError(
            f"Expected {len(params)} symbolic tensor(s); got {len(symbolic)}."
        )

    for name, tensor, buffer in zip(params, symbolic, buffers):
        array = np.asarray(buffer)

        if isinstance(tensor, Tensor):
            source = getattr(tensor, "source", tensor)

            if getattr(source, "dtype", None) is None:
                source.dtype = str(array.dtype)
        else:
            scalars.setdefault(name, array.reshape(()).item())

    return symbolic, buffers, scalars


def _arrange(arrangement, application, params, symbolic, buffers):
    if arrangement is None:
        annotations = inspect.get_annotations(application, eval_str=False)

        try:
            arranged = tuple(annotations[name] for name in params)
        except KeyError as exc:
            raise ProgramDomainError(
                f"Cannot lower `{getattr(application, '__name__', 'application')}`: "
                f"parameter `{exc.args[0]}` has no `Tensor` annotation and no "
                "arrangement was provided."
            ) from exc
    else:
        import copy

        arranged_value = arrangement(*copy.deepcopy(symbolic))
        arranged = (
            arranged_value if isinstance(arranged_value, tuple) else (arranged_value,)
        )

    if len(arranged) != len(params):
        raise ProgramDomainError(
            f"The arrangement returned {len(arranged)} value(s) for {len(params)} "
            "application parameter(s)."
        )

    return tuple(arranged)


# -- symbol resolution -----------------------------------------------------


def resolve_symbols(
    specs, buffers, user_symbols=None, *, arranged=(), extra_aliases=None
):
    """Resolve every symbol used by a program.

    Shapes and strides come from the runtime arrays; constexpr and meta symbols
    come from ``user_symbols`` or from the compiler driver's default rule.

    :param specs: The tensor specs produced by the arrangement.
    :param buffers: The runtime arrays, in parameter order.
    :param user_symbols: Caller-provided symbol values.
    :param arranged: The arranged symbolic tensors (used for meta defaults).
    :param extra_aliases: ``{alias: symbol_name}`` accepted in ``user_symbols``.
    :return: ``{symbol_name: int}``.
    """
    symbols: dict[str, int] = {}

    for spec, buffer in zip(specs, buffers):
        array = np.ascontiguousarray(np.asarray(buffer))
        attrs = spec.attrs

        for name, extent in zip(attrs.get("source_shape", ()), array.shape):
            symbols[str(name)] = int(extent)

        strides = tuple(
            int(stride) // max(int(array.itemsize), 1) for stride in array.strides
        )

        for name, stride in zip(attrs.get("source_strides", ()), strides):
            symbols[str(name)] = int(stride)

    for symbol in _iter_symbols(arranged):
        name = str(symbol)

        if name in symbols:
            continue

        symbols[name] = _default_symbol_value(symbol)

    symbols.update(_normalize_user_symbols(symbols, user_symbols, extra_aliases))

    return symbols


def resolve_program_symbols(program, buffers, user_symbols=None):
    """Resolve the symbols of an already lowered program from its buffers."""
    symbols: dict[str, int] = {}

    for value in program.inputs:
        array = buffers.get(value.name)

        if array is None:
            continue

        array = np.ascontiguousarray(np.asarray(array))
        attrs = value.type.attrs

        for name, extent in zip(attrs.get("source_shape", ()), array.shape):
            symbols[str(name)] = int(extent)

        strides = tuple(
            int(stride) // max(int(array.itemsize), 1) for stride in array.strides
        )

        for name, stride in zip(attrs.get("source_strides", ()), strides):
            symbols[str(name)] = int(stride)

    for name in program.metadata.get("symbols", ()):
        name = str(name)

        if name in symbols:
            continue

        if naming.is_constexpr(name):
            symbols[name] = DEFAULT_CONSTEXPR_VALUE

    symbols.update(_normalize_user_symbols(symbols, user_symbols))

    return symbols


def _iter_symbols(arranged):
    seen = set()

    for tensor in arranged:
        if not hasattr(tensor, "names"):
            continue

        for symbol in tensor.names():
            name = str(symbol)

            if name in seen or not hasattr(symbol, "lower_bound"):
                continue

            seen.add(name)

            yield symbol


def _default_symbol_value(symbol):
    """Return the compiler driver's default value for a meta symbol."""
    lower = int(getattr(symbol, "lower_bound", 1))
    upper = int(getattr(symbol, "upper_bound", DEFAULT_CONSTEXPR_VALUE))
    value = min(max(DEFAULT_CONSTEXPR_VALUE, lower), upper)

    if getattr(symbol, "power_of_two", False):
        value = 1 << max(0, value.bit_length() - 1)

    return int(value)


def _strip_counter(name):
    """Drop a trailing ``_<digits>`` uniquifying suffix, when present."""
    head, separator, tail = name.rpartition("_")

    if head and separator and tail.isdigit():
        return head

    return name


def _strip_once(name):
    """Yield every single-step alias of a symbol name."""
    yield naming.remove_prefixes(name)

    if name.startswith("ninetoothed_"):
        yield name[len("ninetoothed_") :]

    yield _strip_counter(name)


def _symbol_aliases(name):
    """Return every spelling a user may use to bind ``name``.

    Symbols are renamed several times between the Python source and the SSA
    program: the frontend adds ``ninetoothed_``, the ``constexpr``/``meta``
    markers add ``ninetoothed_<marker>_prefix_``, and each declaration appends a
    uniquifying ``_<digits>`` counter.  Every intermediate spelling is accepted,
    so callers do not have to reproduce that chain.
    """
    aliases = {name}
    pending = [name]

    while pending:
        for candidate in _strip_once(pending.pop()):
            if candidate and candidate not in aliases:
                aliases.add(candidate)
                pending.append(candidate)

    return aliases


def _normalize_user_symbols(known, user_symbols, extra_aliases=None):
    if not user_symbols:
        return {}

    aliases = {}

    for name in known:
        for alias in _symbol_aliases(name):
            aliases.setdefault(alias, name)

    for alias, name in (extra_aliases or {}).items():
        if name in known:
            aliases.setdefault(alias, name)

    resolved = {}

    for alias, value in user_symbols.items():
        name = aliases.get(alias)

        if name is None:
            continue

        existing = resolved.get(name)

        if existing is not None and existing != int(value):
            raise MissingSymbolError(
                f"Symbol `{name}` was bound twice with conflicting values "
                f"({existing} and {int(value)})."
            )

        resolved[name] = int(value)

    unknown = set(user_symbols) - set(aliases)

    if unknown:
        raise MissingSymbolError(
            f"Unknown symbol(s) {', '.join(sorted(unknown))}; the program uses "
            f"{', '.join(sorted(known)) or '<none>'}."
        )

    return resolved


def arrangement_symbol_aliases(arrangement):
    """Return ``{parameter_name: symbol_name}`` for a symbolic arrangement.

    An arrangement such as ``def arrangement(x, out, WIDTH=block_size())`` binds
    the symbol to the parameter ``WIDTH``.  The parameter name is accepted as an
    alias of the symbol's own (prefixed) name, so ``symbols={"WIDTH": 128}``
    works.
    """
    if arrangement is None:
        return {}

    try:
        parameters = inspect.signature(arrangement).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins and the like
        return {}

    aliases = {}

    for name, parameter in parameters.items():
        default = parameter.default

        if isinstance(default, Symbol):
            aliases.setdefault(name, str(default))

    return aliases


# -- memory ----------------------------------------------------------------


def _buffer_map(program, buffers):
    """Return ``{input_name: array}`` for a positional buffer sequence."""
    mapping = {}

    for index, value in enumerate(program.inputs):
        if index < len(buffers):
            mapping[value.name] = buffers[index]

    return mapping


def _build_memory(program, buffers, symbols, scalars):
    memory = CPUMemory(symbols)

    for value in program.inputs:
        buffer = buffers.get(value.name) if isinstance(buffers, Mapping) else None

        if buffer is None:
            continue

        array = np.asarray(buffer)

        if array.ndim == 0:
            scalars.setdefault(value.name, array.reshape(()).item())

            continue

        memory.add(build_tensor_runtime(value.name, value.type, array, symbols))

    return memory


def _validate_domain(program, memory, launch_shape):
    size = domain_size(launch_shape)

    for name, tensor in memory.tensors.items():
        if not tensor.is_tiled:
            continue

        extent = tensor.launch_extent

        if extent != size:
            raise ProgramDomainError(
                f"Tensor `{name}` implies {extent} program instance(s) but the "
                f"primary output implies {size}; the arrangement mixes launch "
                "domains the interpreter cannot reconcile."
            )


def _resolve_tracer(trace, tracer):
    if tracer is not None:
        return tracer

    if trace is None:
        return None

    if isinstance(trace, Tracer):
        return trace

    if isinstance(trace, Mapping):
        return Tracer(**dict(trace))

    if trace is True:
        return Tracer()

    raise ProgramDomainError(
        "The `trace` argument must be a Tracer, a mapping of tracer options, or True."
    )


__all__ = [
    "DEFAULT_CONSTEXPR_VALUE",
    "Interpretation",
    "access_map",
    "access_mask",
    "access_offsets",
    "apply_pipeline",
    "arrangement_symbol_aliases",
    "interpret",
    "interpret_program",
    "resolve_program_symbols",
    "resolve_symbols",
]
