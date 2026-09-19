"""NumPy execution of the shared NineToothed SSA and arrangement layouts."""

from dataclasses import replace

from ninetoothed.frontend.preparation import prepare_application
from ninetoothed.ir import ssa
from ninetoothed.naming import remove_prefixes

from .runtime import (
    InterpretationError,
    InterpretationResult,
    TraceEvent,
    UnsupportedOperationError,
    interpret_program,
)


class InterpretedKernel:
    """A prepared application callable that writes to CPU output buffers."""

    def __init__(
        self,
        prepared,
        *,
        grid=None,
        meta=None,
        trace=False,
        frontend_program=None,
        backend=None,
        callback=None,
        watch=(),
    ):
        self._prepared = prepared
        self.program = prepared.program
        self.frontend_program = (
            prepared.program if frontend_program is None else frontend_program
        )
        self.backend = backend
        self.parameters = prepared.parameters
        self.tensors = prepared.tensors
        self.grid = grid
        self.meta = dict(meta or {})
        self.trace = trace
        self.callback = callback
        self.watch = tuple(watch)
        # Match the compiler's schedule-parameter defaults. Explicit meta wins.

        for tensor in prepared.arranged:
            for symbol in tensor.names():
                if not hasattr(symbol, "lower_bound"):
                    continue

                value = min(max(256, int(symbol.lower_bound)), int(symbol.upper_bound))

                if getattr(symbol, "power_of_two", False):
                    value = 1 << max(0, value.bit_length() - 1)

                value = self.meta.get(remove_prefixes(str(symbol)), value)
                self.meta.setdefault(str(symbol), value)

    def __call__(self, *args, **kwargs):
        if len(args) > len(self._prepared.parameters):
            raise TypeError("Too many arguments for interpreted kernel.")

        inputs = dict(zip(self._prepared.parameters, args))

        for name, value in kwargs.items():
            if name not in self._prepared.parameters or name in inputs:
                raise TypeError(f"Unexpected or duplicate argument `{name}`.")

            inputs[name] = value
        return interpret_program(
            self.program,
            inputs,
            tensors=self.tensors,
            grid=self.grid,
            symbols=self.meta,
            trace=self.trace,
            callback=self.callback,
            watch=self.watch,
        )

    def with_program(self, program):
        """Reuse identical layout and argument bindings with a transformed SSA."""
        return InterpretedKernel(
            replace(self._prepared, program=program),
            grid=self.grid,
            meta=self.meta,
            trace=self.trace,
            frontend_program=self.frontend_program,
            backend=self.backend,
            callback=self.callback,
            watch=self.watch,
        )


def interpret(
    arrangement,
    application=None,
    tensors=(),
    *,
    grid=None,
    meta=None,
    trace=False,
    backend=None,
    pipeline=None,
    pass_options=None,
    callback=None,
    watch=(),
):
    """Prepare a CPU callable using the same frontend as compiled backends.

    Use ``interpret(arrangement, application, tensor_descriptors)`` or
    ``interpret(annotated_application)``. By default this runs frontend SSA.
    An explicit ``backend`` runs that backend's real SSA pass pipeline first;
    it never emits GPU source or materializes/executes a GPU backend.
    """
    if application is None:
        application, arrangement = arrangement, None

    prepared = prepare_application(
        application, arrangement=arrangement, tensors=tensors
    )
    frontend_program = prepared.program

    if backend is not None:
        from ninetoothed.compiler.passes import lower_for_target

        try:
            program = lower_for_target(
                prepared.program,
                backend=backend,
                tensors=prepared.tensors,
                pass_pipeline=pipeline,
                pass_options=pass_options,
            )
        except ssa.VerificationError as exc:
            raise InterpretationError(
                f"Backend `{backend}` SSA pipeline produced invalid SSA: {exc}."
            ) from exc

        prepared = replace(prepared, program=program)
    elif pipeline is not None or pass_options is not None:
        raise ValueError("A backend must be specified when choosing SSA passes.")
    return InterpretedKernel(
        prepared,
        grid=grid,
        meta=meta,
        trace=trace,
        frontend_program=frontend_program,
        backend=backend,
        callback=callback,
        watch=watch,
    )


__all__ = [
    "interpret",
    "interpret_program",
    "InterpretedKernel",
    "InterpretationResult",
    "InterpretationError",
    "UnsupportedOperationError",
    "TraceEvent",
]
