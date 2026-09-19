"""The CPU reference interpreter core.

The interpreter walks an :class:`ninetoothed.ir.ssa.Program` once per program
instance.  It never calls a GPU backend: every operation is executed with NumPy
against the CPU memory model built from the layout information the frontend
already attached to the program.
"""

import contextlib
import math

import numpy as np

from .dtypes import resolve_dtype
from .errors import (
    MissingSymbolError,
    ProgramDomainError,
    TraceStop,
    UnsupportedControlFlowError,
    UnsupportedOperationError,
)
from .expr import ExpressionError, parse_expression
from .registry import require_handler
from .trace import Tracer
from .values import scalar, view

#: Operations that terminate a region instead of producing results.
_REGION_TERMINATORS = {"scf.yield"}


class InterpreterState:
    """Mutable runtime state for one program instance.

    :param interpreter: The owning :class:`Interpreter`.
    :param memory: The CPU memory holding every backing buffer.
    :param symbols: The resolved symbol values.
    :param program_id: The linear id of the program instance.
    :param total: The total number of program instances.
    :param tracer: The tracer collecting events.
    """

    def __init__(self, interpreter, memory, symbols, program_id, total, tracer):
        self.interpreter = interpreter
        self.memory = memory
        self.symbols = dict(symbols)
        self.program_id = int(program_id)
        self.total = int(total)
        self.tracer = tracer
        self.values: dict[str, object] = {}
        self.location = "<none>"
        self.depth = 0
        self.context = memory.context(self.program_id)
        self._scopes: list[list[str]] = []

    # -- value access ------------------------------------------------------

    def value(self, name):
        """Return a runtime value by SSA name."""
        try:
            return self.values[name]
        except KeyError as exc:
            raise UnsupportedOperationError(
                f"SSA value `{name}` is not defined at this point.",
                location=self.location,
            ) from exc

    def bind(self, name, value):
        """Bind an SSA name to a runtime value."""
        self.values[name] = value

    @contextlib.contextmanager
    def scope(self):
        """Track names introduced inside a region so they can be dropped."""
        recorded = set(self.values)
        self._scopes.append(recorded)

        try:
            yield
        finally:
            self._scopes.pop()

    # -- symbol and shape resolution ---------------------------------------

    def resolve_int(self, text):
        """Resolve a (possibly symbolic) integer expression."""
        if isinstance(text, (int, np.integer)) and not isinstance(text, bool):
            return int(text)

        if isinstance(text, bool):
            return int(text)

        if isinstance(text, float):
            return int(text)

        text = str(text).strip()

        try:
            return int(text)
        except ValueError:
            pass

        resolved = self.evaluate_attribute(text)

        if resolved is None:
            raise MissingSymbolError(
                f"Cannot resolve `{text}`; bind it through the `symbols` "
                "argument of `ninetoothed.interpret`."
            )

        if isinstance(resolved, np.ndarray):
            if resolved.size != 1:
                raise MissingSymbolError(
                    f"Cannot resolve `{text}` to a scalar; got {resolved!r}."
                )

            resolved = resolved.reshape(()).item()

        return int(resolved)

    def resolve_shape(self, shape):
        """Resolve a tuple of possibly symbolic extents."""
        return tuple(self.resolve_int(dim) for dim in shape)

    def numpy_dtype(self, dtype):
        """Return the NumPy dtype for a NineToothed dtype name."""
        return resolve_dtype(dtype)

    def evaluate_attribute(self, text):
        """Evaluate an attribute expression against values and symbols.

        :param text: The expression text.
        :return: The evaluated value, or ``None`` when it cannot be resolved.
        """
        try:
            expression = parse_expression(str(text))
        except ExpressionError:
            return None

        namespace = dict(self.symbols)
        namespace.update(self.values)

        try:
            return expression.evaluate(namespace)
        except Exception:  # noqa: BLE001 - resolution is best effort
            return None

    # -- execution ---------------------------------------------------------

    def execute_operations(self, operations, path):
        """Execute a straight-line list of operations."""
        for index, operation in enumerate(operations):
            if operation.opcode in _REGION_TERMINATORS:
                raise UnsupportedControlFlowError(
                    f"Operation `{operation.opcode}` appeared outside a region.",
                    opcode=operation.opcode,
                    location=f"{path}:{index}:{operation.opcode}",
                )

            self.execute_operation(operation, path=path, index=index)

    def execute_region(self, block, parent):
        """Execute a region and return the values yielded by ``scf.yield``.

        The location of the enclosing operation (``self.location``) is used to
        build a readable path for the nested operations, so a failure inside a
        loop or branch still points at the exact SSA node.
        """
        path = f"{self.location}/{block.name or 'region'}"

        for index, operation in enumerate(block.operations):
            if operation.opcode == "scf.yield":
                return tuple(operation.operands)

            self.execute_operation(operation, path=path, index=index)

        raise UnsupportedControlFlowError(
            "Region does not terminate with `scf.yield`.",
            opcode=parent.opcode,
            location=self.location,
        )

    def execute_operation(self, operation, *, path, index):
        """Execute one SSA operation."""
        location = f"{path}:{index}:{operation.opcode}"
        self.location = location

        breakpoint = (
            self.tracer.breakpoint_for(operation.opcode, location)
            if self.tracer is not None
            else None
        )

        if breakpoint is not None:
            event = self.tracer.record(
                program_id=self.program_id,
                depth=self.depth,
                operation=operation,
                state=self,
            )

            if breakpoint(event) is False:
                raise TraceStop(event)

        handler = require_handler(operation.opcode, location=location)

        if operation.regions:
            self.depth += 1

        try:
            handler(self, operation)
        finally:
            if operation.regions:
                self.depth -= 1

        if self.tracer is not None and self.tracer.accepts(
            self.program_id, operation.opcode
        ):
            self.tracer.record(
                program_id=self.program_id,
                depth=self.depth,
                operation=operation,
                state=self,
                mask=_mask_of(operation, self),
            )


def _mask_of(operation, state):
    """Return a textual description of the mask applied by a memory operation."""
    if operation.opcode != "mem.store":
        return None

    target = state.values.get(operation.operands[1])

    if target is None or target.kind != "view":
        return None

    view = target.data

    try:
        _, mask = state.context.evaluate_access(view)
    except Exception:  # noqa: BLE001 - the trace must never fail the run
        return None

    mask = np.asarray(mask)

    if mask.ndim == 0:
        return "all" if bool(mask) else "none"

    return f"{int(mask.sum())}/{mask.size} active"


class Interpreter:
    """Executes an ``ssa.Program`` on the CPU.

    :param program: The SSA program to execute.
    :param memory: The CPU memory holding the input buffers.
    :param symbols: The resolved symbol values.
    :param total: The number of program instances to execute.
    :param tracer: An optional :class:`~ninetoothed.interpret.trace.Tracer`.
    :param scalars: Values for non-tensor (constexpr) program inputs.
    """

    def __init__(self, program, memory, symbols, total, tracer=None, scalars=None):
        self.program = program
        self.memory = memory
        self.symbols = dict(symbols)
        self.total = int(total)
        self.tracer = tracer
        self.scalars = dict(scalars or {})

    def run(self, program_ids=None):
        """Execute the program for every program instance.

        :param program_ids: Restrict execution to these program instances.
        :return: ``{tensor_name: numpy.ndarray}`` with the stored outputs.
        """
        instances = (
            range(self.total)
            if program_ids is None
            else tuple(int(i) for i in program_ids)
        )
        outputs: dict[str, np.ndarray] = {}

        for program_id in instances:
            outputs = self.run_instance(program_id)

        return outputs

    def run_instance(self, program_id):
        """Execute the program for a single program instance."""
        if program_id < 0 or program_id >= self.total:
            raise ProgramDomainError(
                f"Program id {program_id} is outside the launch domain of "
                f"{self.total} instance(s)."
            )

        state = InterpreterState(
            interpreter=self,
            memory=self.memory,
            symbols=self.symbols,
            program_id=program_id,
            total=self.total,
            tracer=self.tracer,
        )
        block = self.program.blocks[0] if self.program.blocks else None

        if block is None:
            return {}

        for value in self.program.inputs:
            state.bind(value.name, self._input_value(value))

        state.execute_operations(block.operations, path=block.name or "entry")

        return self._collect_outputs()

    def _input_value(self, value):
        """Build the runtime value bound to one program input."""
        tensor = self.memory.tensors.get(value.name)

        if tensor is None:
            if value.name not in self.scalars:
                raise MissingSymbolError(
                    f"Program input `{value.name}` has no runtime value; pass it "
                    "through the `scalars` argument of `ninetoothed.interpret`."
                )

            return scalar(value.type, self.scalars[value.name])

        level = int(value.type.attrs.get("dtype_level", 0))

        return view(value.type, tensor.root_view(level))

    def _collect_outputs(self):
        names = _stored_tensor_names(self.program)
        outputs = {}

        for name in names:
            tensor = self.memory.tensors.get(name)

            if tensor is None:
                continue

            outputs[name] = tensor.buffer.reshape(tensor.source_shape).copy()

        return outputs


def _stored_tensor_names(program):
    """Return the tensor names written by ``mem.store`` in program order."""
    names = []

    def visit(block):
        for operation in block.operations:
            if operation.opcode == "mem.store" and len(operation.operands) == 2:
                target = operation.operands[1]

                if target not in names:
                    names.append(target)

            for region in operation.regions:
                visit(region)

    for block in program.blocks:
        visit(block)

    return tuple(names)


def derive_program_domain(program, memory):
    """Return the launch shape of a program.

    The launch domain is the view domain of the primary output tensor.  A tensor
    that was tiled exposes its tile grid; a tensor that was only sliced is a
    single tile covering the whole view domain.

    :param program: The SSA program.
    :param memory: The CPU memory holding the tensor runtimes.
    :return: A tuple with the per-instance launch shape.
    """
    names = _stored_tensor_names(program)

    if not names:
        raise ProgramDomainError(
            "The program contains no `mem.store`, so it has no output domain."
        )

    primary = memory.tensors.get(names[0])

    if primary is None:
        raise ProgramDomainError(
            f"The primary output `{names[0]}` is not backed by a CPU tensor."
        )

    if not primary.is_tiled:
        return (1,)

    if not primary.view_shape:
        return (1,)

    return tuple(int(dim) for dim in primary.view_shape)


def domain_size(shape):
    """Return the number of program instances implied by a launch shape."""
    return int(math.prod(shape)) if shape else 1


__all__ = [
    "Interpreter",
    "InterpreterState",
    "Tracer",
    "derive_program_domain",
    "domain_size",
]
