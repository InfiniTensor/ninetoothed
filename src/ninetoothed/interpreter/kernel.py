"""Public CPU interpreter entry points.

``interpret`` mirrors ``ninetoothed.make``: it lowers an arrangement and an
application, then returns a callable kernel. The returned kernel runs on the CPU
with NumPy, accepts NumPy arrays or PyTorch CPU tensors, and never calls a GPU
backend.
"""

import math

import numpy as np

from ninetoothed.interpreter.access import derive_access
from ninetoothed.interpreter.debugger import DebugSession
from ninetoothed.interpreter.errors import (
    InvalidArgumentError,
    InvalidProgramError,
    UnsupportedAccessError,
)
from ninetoothed.interpreter.executor import Executor
from ninetoothed.interpreter.lowering import Interpretation, lower
from ninetoothed.interpreter.memory import TensorBuffer, storage_view
from ninetoothed.interpreter.ops import supported_operations, unsupported_operations
from ninetoothed.interpreter.runtime import RuntimeState
from ninetoothed.interpreter.trace import ExecutionTrace, TraceRecorder
from ninetoothed.symbol import Symbol


class Interpreter:
    """A NineToothed kernel executed by the CPU reference interpreter.

    :param interpretation: The lowered interpretation.
    :param meta: Explicit values for the meta and constexpr symbols.
    :param trace: Whether to record an execution trace.
    :param trace_options: The filtering and detail settings of the trace.
    """

    def __init__(
        self,
        interpretation: Interpretation,
        *,
        meta=None,
        trace=False,
        trace_options=None,
    ):
        self.interpretation = interpretation
        self.name = interpretation.program.kind
        self.meta_values = dict(interpretation.meta_defaults)
        self.provided_symbols = set()
        self.last_trace = ExecutionTrace()
        self.trace_recorder = TraceRecorder(trace_options) if trace else None

        for key, value in dict(meta or {}).items():
            symbol = self.symbol_of(key)

            if symbol is None:
                raise InvalidArgumentError(
                    f"Cannot find `{key}` among the meta and constexpr symbols of "
                    f"`{self.name}`."
                )

            self.meta_values[symbol] = int(value)
            self.provided_symbols.add(key)

    @property
    def program(self):
        """Return the SSA program the interpreter executes."""
        return self.interpretation.program

    @property
    def raw_program(self):
        """Return the SSA program before the pass pipeline."""
        return self.interpretation.raw_program

    @property
    def parameters(self) -> tuple:
        """Return the application parameter names."""
        return self.interpretation.parameters

    @property
    def pass_trace(self) -> tuple:
        """Return the names of the SSA passes that ran."""
        return self.interpretation.pass_trace

    def symbol_of(self, name: str):
        """Return the mangled name of a user-facing symbol."""
        names = self.interpretation.symbol_names

        if name in names:
            return names[name]

        if name in self.meta_values:
            return name

        return None

    def supported_operations(self) -> tuple:
        """Return the interpreter's operation support matrix."""
        return supported_operations()

    def unsupported_operations(self) -> tuple:
        """Return the operations the interpreter rejects, with the reason."""
        return unsupported_operations()

    def __call__(self, *args, **kwargs):
        """Run the application on the CPU and return the primary output."""
        return self.run(*args, **kwargs)

    def debug(self, *args, breakpoints=(), watch=(), **kwargs) -> DebugSession:
        """Open an interactive session over the application.

        The session does not run anything yet; it replays the application for
        every stop the caller asks for.

        :param args: The tensor and scalar arguments, in parameter order.
        :param breakpoints: The breakpoints the session stops at.
        :param watch: The SSA names that are reported at every stop.
        :param kwargs: The constexpr symbols and the meta symbols.
        :return: The interactive session.
        """
        return DebugSession(self, *args, breakpoints=breakpoints, watch=watch, **kwargs)

    def run(self, *args, inspector=None, **kwargs):
        """Run the application on the CPU.

        :param args: The tensor and scalar arguments, in parameter order.
        :param inspector: An optional callable that is offered every operation
            before it runs and may stop the execution.
        :param kwargs: The remaining arguments, the constexpr symbols, and the meta
            symbols.
        :return: The primary output as a NumPy array, or ``None`` when the
            application stores nothing.
        """
        values, meta = self._bind(args, kwargs)
        plan = self._plan(values, meta)
        recorder = self.trace_recorder

        if recorder is not None:
            recorder.trace = ExecutionTrace()

        executor = Executor(
            program=self.program,
            buffers=plan.buffers,
            accesses=plan.accesses,
            scalars=plan.scalars,
            symbols=plan.symbols,
            sources=plan.sources,
            program_shape=plan.program_shape,
            recorder=recorder,
            inspector=inspector,
        )

        executor.run()

        if recorder is not None:
            self.last_trace = recorder.trace

        return plan.output()

    def _bind(self, args, kwargs):
        interpretation = self.interpretation
        options = dict(kwargs)
        values = {}

        for index, name in enumerate(interpretation.parameters):
            if index < len(args):
                values[name] = args[index]

        meta = dict(self.meta_values)
        provided = set(self.provided_symbols)

        for name in list(options):
            if name in interpretation.parameters and name not in values:
                values[name] = options.pop(name)

                continue

            symbol = self.symbol_of(name)

            if symbol is None:
                raise InvalidArgumentError(
                    f"Cannot find `{name}` among the parameters and symbols of "
                    f"`{self.name}`."
                )

            meta[symbol] = int(options.pop(name))
            provided.add(name)

        missing = tuple(
            name for name in interpretation.parameters if name not in values
        )

        if missing:
            raise InvalidArgumentError(
                f"Cannot run `{self.name}`: missing the arguments {', '.join(missing)}."
            )

        required = tuple(
            name for name in interpretation.required_symbols if name not in provided
        )

        if required:
            raise InvalidArgumentError(
                f"Cannot run `{self.name}`: missing the constexpr symbols "
                f"{', '.join(required)}."
            )

        return values, meta

    def _plan(self, values, meta):
        interpretation = self.interpretation
        arranged_of = dict(zip(interpretation.parameters, interpretation.arranged))
        spec_of = {spec.name: spec for spec in interpretation.specs}
        symbols = {}
        sources = {}
        subs = {}
        shapes = {}

        for name in interpretation.tensor_parameters:
            array = values[name]
            arranged = arranged_of[name]
            source = arranged.source
            shape = tuple(int(dim) for dim in np.shape(array))

            if len(shape) != int(source.ndim):
                raise InvalidArgumentError(
                    f"Tensor `{name}` must have {source.ndim} dimensions, got "
                    f"{len(shape)}."
                )

            shapes[name] = shape
            subs[source] = {"shape": shape}
            sources[name] = tuple(
                str(dim) for dim in spec_of[name].attrs.get("source_shape", ())
            )

            strides = _default_strides(shape)

            for dim in range(len(shape)):
                symbols[str(source.shape[dim])] = shape[dim]
                symbols[source.stride_string(dim)] = strides[dim]

        buffers = {
            name: TensorBuffer(
                name=name,
                storage=storage_view(values[name], name=name),
                shape=shapes[name],
            )
            for name in interpretation.tensor_parameters
        }

        for name, value in meta.items():
            symbols[name] = int(value)
            subs[Symbol(name)] = int(value)

        state = RuntimeState(symbols=symbols)
        accesses = {}

        for name in interpretation.tensor_parameters:
            access = derive_access(
                name,
                arranged_of[name],
                subs,
                other=spec_of[name].attrs.get("other"),
            )

            _check_value_shape(name, access, spec_of[name], state)

            accesses[name] = access

        program_shape = _program_shape(self.program, accesses)

        for name, access in accesses.items():
            if access.num_programs != max(1, math.prod(program_shape)):
                raise UnsupportedAccessError(
                    f"Tensor `{name}` covers {access.num_programs} program instances "
                    f"while the launch covers {max(1, math.prod(program_shape))}."
                )

        scalars = {name: values[name] for name in interpretation.scalar_parameters}

        return _Plan(
            buffers=buffers,
            accesses=accesses,
            scalars=scalars,
            symbols=symbols,
            sources=sources,
            program_shape=program_shape,
            stores=_store_targets(self.program),
        )


class _Plan:
    """The resolved execution plan of one interpreter call."""

    def __init__(
        self, *, buffers, accesses, scalars, symbols, sources, program_shape, stores
    ):
        self.buffers = buffers
        self.accesses = accesses
        self.scalars = scalars
        self.symbols = symbols
        self.sources = sources
        self.program_shape = program_shape
        self.stores = stores

    def output(self):
        """Return the primary output of the application."""
        for name in self.stores:
            if name in self.buffers:
                buffer = self.buffers[name]

                return np.reshape(buffer.storage, buffer.shape)

        return None


def interpret(
    arrangement,
    application=None,
    tensors=(),
    *,
    backend=None,
    kernel_name=None,
    pipeline=None,
    pass_options=None,
    run_passes=True,
    meta=None,
    trace=False,
    trace_options=None,
):
    """Return a CPU interpreter kernel for a NineToothed application.

    :param arrangement: The arrangement function, or the application itself when
        the tensors are annotated in place.
    :param application: The application function.
    :param tensors: The declared symbolic tensors, in parameter order.
    :param backend: The backend whose SSA shape the interpreter mirrors.
    :param kernel_name: The SSA program name.
    :param pipeline: An optional SSA pass pipeline specification.
    :param pass_options: Optional per-pass options.
    :param run_passes: Whether to run the SSA pass pipeline.
    :param meta: Explicit values for the meta and constexpr symbols.
    :param trace: Whether to record an execution trace.
    :param trace_options: The filtering and detail settings of the trace.
    :return: The interpreter kernel.
    """
    if application is None:
        arrangement, application = None, arrangement

    interpretation = lower(
        arrangement,
        application,
        tensors,
        backend=backend,
        kernel_name=kernel_name,
        pipeline=pipeline,
        pass_options=pass_options,
        run_passes=run_passes,
    )

    return Interpreter(
        interpretation, meta=meta, trace=trace, trace_options=trace_options
    )


def _check_value_shape(name, access, spec, state) -> None:
    expected = state.evaluate_shape(spec_type_shape(spec))
    actual = tuple(access.value_shape)

    if actual[: len(expected)] != expected:
        raise InvalidProgramError(
            f"Tensor `{name}` is arranged into the value shape {actual} while the "
            f"SSA program expects {expected}."
        )


def spec_type_shape(spec) -> tuple:
    """Return the symbolic value shape the SSA program expects for a tensor."""
    return tuple(
        dim.render() if hasattr(dim, "render") else str(dim)
        for dim in spec.layout.application_shape
    )


def _program_shape(program, accesses):
    for operation in _walk_operations(program.blocks[0].operations):
        if operation.opcode != "mem.store" or len(operation.operands) < 2:
            continue

        target = operation.operands[1]

        if target in accesses:
            return accesses[target].program_shape

    if not accesses:
        return ()

    return next(iter(accesses.values())).program_shape


def _store_targets(program) -> tuple:
    targets = []

    for operation in _walk_operations(program.blocks[0].operations):
        if operation.opcode != "mem.store" or len(operation.operands) < 2:
            continue

        target = operation.operands[1]

        if target not in targets:
            targets.append(target)

    return tuple(targets)


def _walk_operations(operations):
    for operation in operations:
        yield operation

        for region in operation.regions:
            yield from _walk_operations(region.operations)


def _default_strides(shape) -> tuple:
    shape = tuple(int(dim) for dim in shape)
    strides = [1] * len(shape)

    for dim in range(len(shape) - 2, -1, -1):
        strides[dim] = strides[dim + 1] * shape[dim + 1]

    return tuple(strides)


__all__ = ["Interpreter", "interpret"]
