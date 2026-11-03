"""Program-instance execution for the CPU reference interpreter.

The executor walks the lowered SSA program once per program instance, in
ascending program id order. Memory accesses go through the offsets and the mask
derived from the arrangement, so lanes that are masked out never touch the CPU
storage, even when their offsets are out of bounds.
"""

import math

import numpy as np

from ninetoothed.interpreter import ops
from ninetoothed.interpreter.errors import (
    InvalidProgramError,
    UnsupportedAccessError,
    UnsupportedOperationError,
)
from ninetoothed.interpreter.runtime import RuntimeState


class Executor:
    """Execute one lowered SSA program on CPU memory.

    :param program: The lowered SSA program.
    :param buffers: The CPU buffers of the tensor arguments, keyed by SSA name.
    :param accesses: The access mappings of the tensor arguments.
    :param scalars: The runtime values of the scalar arguments.
    :param symbols: The resolved layout symbols.
    :param sources: The symbolic source shapes of the tensor arguments.
    :param program_shape: The shape of the program-instance domain.
    :param recorder: The optional trace recorder.
    :param inspector: The optional callable that may stop the execution before an
        operation runs.
    """

    def __init__(
        self,
        *,
        program,
        buffers,
        accesses,
        scalars,
        symbols,
        sources,
        program_shape,
        recorder=None,
        inspector=None,
    ):
        self.program = program
        self.buffers = buffers
        self.accesses = accesses
        self.scalars = scalars
        self.symbols = symbols
        self.sources = sources
        self.program_shape = tuple(program_shape)
        self.recorder = recorder
        self.inspector = inspector
        self.operations_run = 0

    @property
    def num_programs(self) -> int:
        """Return the number of program instances to execute."""
        return max(1, math.prod(self.program_shape))

    def run(self) -> None:
        """Execute every program instance in program id order."""
        for program_id in range(self.num_programs):
            self.run_instance(program_id)

    def run_instance(self, program_id: int) -> None:
        """Execute one program instance.

        :param program_id: The linearized program instance index.
        """
        state = self.create_state(program_id)
        self.load_arguments(state)
        self.run_block(self.program.blocks[0], state, path="entry")

    def create_state(self, program_id: int) -> RuntimeState:
        """Create the runtime state of one program instance."""
        return RuntimeState(
            symbols=self.symbols,
            memory=self.buffers,
            accesses=self.accesses,
            sources=self.sources,
            program_id=program_id,
            program_shape=self.program_shape,
        )

    def load_arguments(self, state: RuntimeState) -> None:
        """Bind the tensor and scalar arguments of one program instance.

        :param state: The runtime state to bind into.
        """
        for name, buffer in self.buffers.items():
            access = state.access(name)
            offsets, mask = access.window(state.program_id)
            values = buffer.gather(offsets, mask, access.other)

            state.bind(name, values)
            state.mask = mask

        state.environment.update(self.scalars)

    def run_block(self, block, state: RuntimeState, *, path: str):
        """Execute every operation of a block.

        :param block: The SSA block.
        :param state: The runtime state.
        :param path: The SSA location prefix of the block.
        :return: The values yielded by the block, if it ends with ``scf.yield``.
        """
        yielded = None

        for index, operation in enumerate(block.operations):
            location = f"{path}:{index}:{operation.opcode}"

            self.checkpoint(operation, location, state)

            if operation.opcode == "scf.yield":
                operands = tuple(state.lookup(name) for name in operation.operands)
                yielded = operands

                self.record(operation, location, state, outputs=())

                continue

            self.run_operation(operation, location, state)

        return yielded

    def run_operation(self, operation, location: str, state: RuntimeState) -> None:
        """Execute one SSA operation.

        :param operation: The SSA operation.
        :param location: The SSA location of the operation.
        :param state: The runtime state.
        """
        inputs = tuple(
            (name, state.environment.get(name)) for name in operation.operands
        )
        context = ops.OperationContext(
            operation=operation,
            location=location,
            suffix="",
            executor=self,
            state=state,
        )

        ops.execute(context)

        outputs = tuple(
            (result.name, state.environment.get(result.name))
            for result in operation.results
        )

        self.record(operation, location, state, inputs=inputs, outputs=outputs)

    def checkpoint(self, operation, location: str, state: RuntimeState) -> None:
        """Offer one operation to the inspector before it runs.

        :param operation: The SSA operation that is about to run.
        :param location: The SSA location of the operation.
        :param state: The runtime state of the current program instance.
        :raises ExecutionStopped: When the inspector asks to stop the execution.
        """
        if self.inspector is None:
            return

        index = self.operations_run
        self.operations_run = index + 1

        if self.inspector(
            operation=operation, location=location, state=state, index=index
        ):
            raise ExecutionStopped(
                index=index,
                program_id=state.program_id,
                location=location,
                opcode=operation.opcode,
            )

    def record(self, operation, location, state, *, inputs=(), outputs=()) -> None:
        """Record one executed operation when tracing is enabled."""
        if self.recorder is None:
            return

        if not self.recorder.enabled(
            program_id=state.program_id, opcode=operation.opcode
        ):
            return

        self.recorder.record(
            program_id=state.program_id,
            location=location,
            opcode=operation.opcode,
            operands=operation.operands,
            inputs=inputs,
            outputs=outputs,
            mask=state.mask if operation.opcode.startswith("mem.") else None,
        )

    def store(self, context: ops.OperationContext, target: str, value) -> None:
        """Store a value through the arrangement mask of the target tensor.

        :param context: The execution context of the ``mem.store`` operation.
        :param target: The SSA name of the destination tensor.
        :param value: The value to store.
        """
        access = context.state.accesses.get(target)
        buffer = context.state.memory.get(target)

        if access is None or buffer is None:
            raise UnsupportedOperationError(
                context.opcode,
                context.location,
                f"the destination `{target}` is not an interpreted tensor argument",
            )

        offsets, mask = access.window(context.state.program_id)
        context.state.mask = mask

        stored = window_value(value, access.value_shape, context)

        buffer.scatter(offsets, mask, stored)

    def run_region(self, region, state: RuntimeState, *, path: str):
        """Execute one SSA region and return its yielded values."""
        return self.run_block(region, state, path=path)


def window_value(value, value_shape, context: ops.OperationContext):
    """Return the value to store into one arrangement window.

    The backend stores a value whose shape matches its access window, and
    replicating a smaller value over the window is not what the emitted code
    does, so the interpreter refuses to guess. A value with the window's element
    count is accepted, and every other shape is reported instead of being
    silently broadcast into a result the backend would not produce.

    :param value: The value produced by the SSA program.
    :param value_shape: The shape of the target arrangement window.
    :param context: The execution context of the ``mem.store`` operation.
    :return: The value shaped like the window.
    :raises UnsupportedAccessError: When the value does not fit the window.
    """
    shape = tuple(int(dim) for dim in value_shape)
    stored = np.asarray(value)

    if shape == stored.shape:
        return value

    if stored.size == math.prod(shape):
        return stored.reshape(shape)

    raise UnsupportedAccessError(
        f"Cannot store a value of shape {stored.shape} into the arrangement window "
        f"of shape {shape} at `{context.location}`: the interpreter only reshapes a "
        "value that already has the window's element count, and never replicates it."
    )


def run_for(executor: Executor, context: ops.OperationContext) -> None:
    """Execute an ``scf.for`` operation.

    :param executor: The executor running the program.
    :param context: The execution context of the operation.
    """
    operation = context.operation
    state = context.state

    if not operation.regions:
        raise InvalidProgramError(
            f"The operation at `{context.location}` has no loop body region."
        )

    lower = int(np.asarray(state.lookup(operation.operands[0])))
    upper = int(np.asarray(state.lookup(operation.operands[1])))
    step = int(np.asarray(state.lookup(operation.operands[2])))
    carried = [state.lookup(name) for name in operation.operands[3:]]
    region = operation.regions[0]
    induction_name = region.args[0].name if region.args else None
    carried_names = [argument.name for argument in region.args[1:]]

    for index in range(lower, upper, step):
        scoped = state.child()

        if induction_name is not None:
            scoped.bind(induction_name, index)

        for name, value in zip(carried_names, carried):
            scoped.bind(name, value)

        yielded = executor.run_region(
            region, scoped, path=f"{context.location}/region0"
        )

        if yielded is None:
            raise InvalidProgramError(
                f"The loop body of `{context.location}` does not yield."
            )

        carried = list(yielded)
        state.mask = scoped.mask

    for result, value in zip(operation.results, carried):
        state.bind(result.name, value)


def run_if(executor: Executor, context: ops.OperationContext) -> None:
    """Execute an ``scf.if`` operation.

    :param executor: The executor running the program.
    :param context: The execution context of the operation.
    """
    operation = context.operation
    state = context.state
    condition = np.asarray(state.lookup(operation.operands[0]))

    if condition.ndim != 0:
        raise UnsupportedOperationError(
            context.opcode,
            context.location,
            "the condition must be a scalar",
        )

    chosen = 0 if bool(condition) else 1

    if not operation.results:
        if chosen < len(operation.regions):
            scoped = state.child()
            executor.run_region(
                operation.regions[chosen],
                scoped,
                path=f"{context.location}/region{chosen}",
            )
            state.mask = scoped.mask

        return

    if len(operation.regions) != 2:
        raise InvalidProgramError(
            f"The operation at `{context.location}` produces results but has "
            f"{len(operation.regions)} regions."
        )

    scoped = state.child()
    yielded = executor.run_region(
        operation.regions[chosen], scoped, path=f"{context.location}/region{chosen}"
    )

    if yielded is None:
        raise InvalidProgramError(
            f"The chosen branch of `{context.location}` does not yield."
        )

    for result, value in zip(operation.results, yielded):
        state.bind(result.name, value)


class ExecutionStopped(Exception):
    """The signal an inspector raises through the interpreter to stop a run.

    The interpreter keeps no resumable state, so stopping unwinds the execution.
    The session that drove the run catches this signal and reports the stop.

    :param index: The ordinal of the operation that did not run.
    :param program_id: The program instance of that operation.
    :param location: The SSA location of that operation.
    :param opcode: The SSA opcode of that operation.
    """

    def __init__(self, *, index: int, program_id: int, location: str, opcode: str):
        super().__init__(f"Execution stopped before `{location}`.")
        self.index = index
        self.program_id = program_id
        self.location = location
        self.opcode = opcode


__all__ = ["ExecutionStopped", "Executor", "run_for", "run_if", "window_value"]
