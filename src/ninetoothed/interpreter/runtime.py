"""A deterministic, checked CPU evaluator for NineToothed's structured SSA."""

import ast
import itertools
import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from ninetoothed.ir import ssa
from ninetoothed.ir.provenance import operation_locations
from ninetoothed.naming import is_next_power_of_2, remove_prefixes

from .access import MemoryAccess, MemoryRecorder
from .expressions import (
    BINARY,
    UNARY,
    _next_power_of_2,
    evaluate,
    numpy_dtype,
    shape_value,
)
from .geometry import ExtractionGeometry
from .memory import Pointer, TensorRef, materialize


class InterpretationError(RuntimeError):
    """Execution failed at a known SSA location and program instance."""


class UnsupportedOperationError(InterpretationError):
    """An operation has no implemented CPU semantics."""


def _adapt_inputs(inputs):
    """Expose already-loaded Torch CPU buffers as zero-copy NumPy views.

    Looking in sys.modules is deliberate: the NumPy execution route never
    imports an optional framework merely to determine an argument's type.
    """
    torch = sys.modules.get("torch")
    adapted, originals, memo = {}, {}, {}
    tensor_type = getattr(torch, "Tensor", ()) if torch is not None else ()

    for name, value in inputs.items():
        if tensor_type and isinstance(value, tensor_type):
            if value.device.type != "cpu":
                raise InterpretationError(
                    f"Input `{name}` is on {value.device}; the CPU interpreter rejects non-CPU tensors."
                )

            if value.requires_grad:
                raise InterpretationError(
                    f"Input `{name}` requires_grad=True; explicitly detach it before CPU interpretation."
                )

            if value.layout != torch.strided:
                raise InterpretationError(
                    f"Input `{name}` has unsupported layout {value.layout}; only strided CPU tensors are supported."
                )

            if value.is_conj() or value.is_neg():
                raise InterpretationError(
                    f"Input `{name}` has a conjugate/negative view bit; a zero-copy NumPy view is unavailable."
                )

            originals[name] = value

            if id(value) not in memo:
                try:
                    memo[id(value)] = value.numpy()
                except (RuntimeError, TypeError) as exc:
                    raise InterpretationError(
                        f"Input `{name}` cannot be represented as a zero-copy NumPy view: {exc}."
                    ) from exc

            value = memo[id(value)]

        adapted[name] = value
    return adapted, originals


@dataclass(frozen=True)
class TraceEvent:
    """An immutable operation event with independent, serializable snapshots."""

    program_id: tuple
    location: str
    opcode: str
    results: dict
    iteration: tuple = ()
    watched: dict | None = None
    inputs: dict | None = None
    mask: dict | None = None
    lane: tuple | None = None
    memory: tuple[MemoryAccess, ...] | None = None
    sequence: int | None = None


@dataclass(frozen=True)
class InterpretationResult:
    """Written output buffers, deterministic trace, and the executed SSA."""

    outputs: dict
    trace: tuple
    program: ssa.Program


def _snapshot(value, *, reference_only=False):
    if isinstance(value, Pointer):
        return {
            "pointer_offsets": np.asarray(value.offset).tolist(),
            "dtype": str(value.array.dtype),
            "shape": list(value.array.shape),
        }

    if isinstance(value, TensorRef) and (
        reference_only or (value.levels and value.level < len(value.levels) - 1)
    ):
        return {
            "tensor": None if value.spec is None else value.spec.name,
            "dtype": str(value.array.dtype),
            "shape": list(value.shape),
            "outer_index": value.outer_index,
            "level": value.level,
            "extracted": value.extracted,
        }

    array = np.asarray(materialize(value))

    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "value": array.tolist(),
    }


def _bind_symbols(program, inputs, specs, supplied):
    symbols = dict(supplied or {})

    for value in program.inputs:
        if value.name not in inputs:
            raise InterpretationError(f"Missing SSA input `{value.name}`.")

        actual = inputs[value.name]

        if np.ndim(actual) == 0 and value.type.kind != "tensor":
            symbols[value.name] = np.asarray(actual).item()

    by_name = {spec.name: spec for spec in specs}
    checks = []

    for value in program.inputs:
        actual = inputs[value.name]

        if not isinstance(actual, np.ndarray):
            continue

        spec = by_name.get(value.name)
        dimensions = (
            spec.attrs.get("source_shape", ()) if spec is not None else value.type.shape
        )

        if dimensions and len(dimensions) != actual.ndim:
            raise InterpretationError(
                f"Input `{value.name}` has rank {actual.ndim}, expected {len(dimensions)}."
            )

        for text, size in zip(dimensions, actual.shape):
            text = str(text)

            if text.isidentifier():
                if text in symbols and symbols[text] != size:
                    raise InterpretationError(
                        f"Conflicting value for shape symbol `{text}`."
                    )

                symbols[text] = size
            else:
                checks.append((text, size))

        if spec is not None:
            for name, stride in zip(
                spec.attrs.get("source_strides", ()), actual.strides
            ):
                if str(name).isidentifier():
                    symbols[str(name)] = stride // actual.itemsize

    names = set(program.metadata.get("symbols", ()))

    for spec in specs:
        for value in (*spec.shape, *spec.attrs.get("application_shape", ())):
            # Only names from structural expressions, never Python execution.
            names.update(
                node.id
                for node in ast.walk(ast.parse(str(value), mode="eval"))
                if isinstance(node, ast.Name)
            )

    for name in names:
        plain = remove_prefixes(name)

        if name not in symbols and plain in symbols:
            symbols[name] = symbols[plain]

        if is_next_power_of_2(name) and plain in symbols:
            symbols[name] = _next_power_of_2(symbols[plain])

    for expression, actual in checks:
        if int(evaluate(expression, symbols)) != actual:
            raise InterpretationError(f"Input shape disagrees with `{expression}`.")
    return symbols


def _program_shapes(specs, symbols):
    return {
        spec.name: shape_value(spec.layout.view_shape, symbols)
        for spec in specs
        if spec.layout is not None and spec.layout.levels
    }


def _local_program_index(shape, master_shape, flat_index):
    if not shape or math.prod(shape) <= 1:
        return 0
    # Layouts may squeeze/insert singleton program axes, e.g. row reduction
    # x.tile((1, K)) has domain (rows, 1), while out.tile((1,)) has (rows,).
    # Removing only size-one axes preserves the flattened coordinate order.

    if tuple(size for size in shape if size != 1) == tuple(
        size for size in master_shape if size != 1
    ):
        return flat_index

    master_coords = np.unravel_index(flat_index, master_shape)

    if len(shape) > len(master_shape):
        raise ValueError("Input has more program dimensions than the launch domain.")

    aligned = master_shape[-len(shape) :]

    if any(local not in (1, master) for local, master in zip(shape, aligned)):
        raise ValueError("Tensor program domains cannot be broadcast together.")

    coords = tuple(
        0 if size == 1 else int(coord)
        for size, coord in zip(shape, master_coords[-len(shape) :])
    )

    return int(np.ravel_multi_index(coords, shape))


class _Execution:
    def __init__(
        self,
        program,
        inputs,
        specs,
        symbols,
        grid,
        *,
        trace,
        program_ids,
        opcodes,
        watch,
        callback,
        handlers,
    ):
        self.program = program
        self.inputs = inputs
        self.specs = {spec.name: spec for spec in specs}
        self.symbols = symbols
        self.shapes = _program_shapes(specs, symbols)
        self.master_shape = max(
            self.shapes.values(),
            key=lambda shape: (math.prod(shape), len(shape)),
            default=(1,),
        )

        if grid is None:
            grid = (math.prod(self.master_shape),)

        if isinstance(grid, int):
            grid = (grid,)

        if any(np.ndim(size) != 0 or int(size) != size for size in grid):
            raise InterpretationError("Grid dimensions must be integers.")

        self.grid = tuple(int(size) for size in grid)

        if not 1 <= len(self.grid) <= 3 or any(size < 0 for size in self.grid):
            raise InterpretationError(
                "Grid must have one to three nonnegative dimensions."
            )

        if self.shapes and math.prod(self.grid) != math.prod(self.master_shape):
            raise InterpretationError(
                "Explicit grid does not match the arrangement program domain."
            )

        self.tracing = trace
        self.program_ids = (
            None
            if program_ids is None
            else {
                tuple(pid) if not isinstance(pid, int) else (pid, 0, 0)
                for pid in program_ids
            }
        )
        self.opcodes = None if opcodes is None else set(opcodes)
        self.watch = tuple(watch)
        self.callback = callback
        self.handlers = dict(handlers or {})
        self.events = []
        self.sequence = 0
        self.memory = (
            MemoryRecorder({value.name: inputs[value.name] for value in program.inputs})
            if trace or callback is not None
            else None
        )
        self.program_id = (0, 0, 0)
        self.iteration = ()
        self.location = "entry"
        self.lane = None
        decomposed_stores = tuple(
            (index, op)
            for index, op in enumerate(program.blocks[0].operations)
            if op.opcode == "mem.store"
            and op.attrs.get("decomposition") in {"matmul", "transpose"}
        )
        self.scalar_output = None

        if decomposed_stores:
            outputs = {op.operands[1] for _index, op in decomposed_stores}

            if len(outputs) != 1 or len(decomposed_stores) != 1:
                raise UnsupportedOperationError(
                    "Scalar decomposition requires exactly one output store."
                )

            index, operation = decomposed_stores[0]
            extra_stores = tuple(
                location
                for location, other in operation_locations(program)
                if other is not operation
                and (
                    other.opcode == "mem.store"
                    or other.opcode.startswith(("atomic.", "mem.atomic"))
                )
            )

            if extra_stores:
                raise UnsupportedOperationError(
                    f"Operation {extra_stores[0]}: scalar decomposition mixed with "
                    "other output stores or atomic effects is not supported."
                )

            if (
                math.prod(self.grid) != 1
                and operation.attrs["decomposition"] != "matmul"
            ):
                raise UnsupportedOperationError(
                    f"Operation entry:{index}:mem.store: scalar {operation.attrs['decomposition']} decomposition "
                    "does not support multiple arranged programs; use a single program or preserve linalg."
                )

            self.scalar_output = operation.operands[1]
            self._validate_scalar_storage(operation, f"entry:{index}:mem.store")

            if math.prod(self.grid) > 1:
                self._validate_scalar_matmul_domain(
                    operation, f"entry:{index}:mem.store"
                )

        self.reuse_extraction_geometry = (
            self.scalar_output is not None
            and type(operation.attrs.get("decomposition")) is str
            and operation.attrs["decomposition"] == "matmul"
            and type(trace) is bool
            and callback is None
            and not self.watch
            and not self.handlers
            and program_ids is None
            and opcodes is None
        )

    def _validate_scalar_storage(self, operation, location):
        """Scalar lane replay cannot read storage that an earlier lane writes."""

        def require(condition, reason):
            if not condition:
                raise UnsupportedOperationError(
                    f"Operation {location}: scalar decomposition requires independent "
                    f"element-aligned output storage: {reason}."
                )

        output = self.inputs.get(self.scalar_output)
        require(isinstance(output, np.ndarray), "a named array output is required")
        arrays = tuple(
            value for value in self.inputs.values() if isinstance(value, np.ndarray)
        )
        require(
            all(
                value.itemsize > 0
                and all(stride % value.itemsize == 0 for stride in value.strides)
                for value in arrays
            ),
            "strides must be aligned to complete elements",
        )
        require(
            not any(
                np.may_share_memory(output, value)
                for name, value in self.inputs.items()
                if name != self.scalar_output and isinstance(value, np.ndarray)
            ),
            "potentially overlapping input and output storage is not supported",
        )
        metadata_operations = {"index.offset", "shape.dim", "tensor.stride"}

        for read_location, other in operation_locations(self.program):
            require(
                other is operation
                or self.scalar_output not in other.operands
                or other.opcode in metadata_operations,
                f"reading output values at {read_location} is not supported",
            )

        if math.prod(self.grid) <= 1:
            coordinates = np.indices(output.shape, sparse=True)
            offsets = sum(
                (index * stride for index, stride in zip(coordinates, output.strides)),
                np.zeros((), dtype=np.int64),
            )
            require(np.unique(offsets).size == output.size, "output writes overlap")

    def _validate_scalar_matmul_domain(self, operation, location):
        """Prove complete K tiles and independent M/N writes before executing SSA."""

        def require(condition, reason):
            if not condition:
                raise UnsupportedOperationError(
                    f"Operation {location}: scalar matmul across multiple arranged programs requires "
                    f"independent output tiles with complete K reduction: {reason}."
                )

        names = tuple(operation.attrs.get("matmul_operands", ()))
        require(len(names) == 2, "missing decomposed operand bindings")
        names = (*names, self.scalar_output)
        require(
            all(name in self.inputs for name in names),
            "computed operands are not supported",
        )
        arrays = tuple(self.inputs[name] for name in names)
        require(
            all(isinstance(value, np.ndarray) and value.ndim == 2 for value in arrays),
            "rank-2 tensors are required",
        )
        lhs, rhs, output = arrays
        require(
            lhs.shape[1] == rhs.shape[0]
            and output.shape == (lhs.shape[0], rhs.shape[1]),
            "source matrix dimensions disagree",
        )
        require(
            math.prod(self.shapes.get(self.scalar_output, ())) == math.prod(self.grid),
            "output does not cover the launch domain",
        )
        require(
            all(self.specs.get(name) is not None for name in names),
            "arranged tensor descriptors are required",
        )
        require(
            all(self.specs[name].attrs.get("other") in (None, 0) for name in names[:2]),
            "K padding must be zero",
        )
        seen = set()

        for flat_index in range(math.prod(self.grid)):
            refs = []

            for name, array in zip(names, arrays):
                try:
                    local = _local_program_index(
                        self.shapes.get(name, ()), self.master_shape, flat_index
                    )
                except ValueError as error:
                    raise UnsupportedOperationError(
                        f"Operation {location}: scalar matmul operand program domains "
                        "cannot be broadcast to the output."
                    ) from error

                refs.append(
                    TensorRef(array, self.specs[name], self.symbols, outer_index=local)
                )

            a, b, c = refs
            require(
                all(len(ref.shape) == 2 and len(ref.levels) == 1 for ref in refs),
                "one rank-2 value level is required",
            )
            require(
                a.shape[0] == c.shape[0]
                and b.shape[1] == c.shape[1]
                and a.shape[1] == b.shape[0],
                "logical matrix tile dimensions disagree",
            )
            require(
                a.shape[1] >= lhs.shape[1] and b.shape[0] >= rhs.shape[0],
                "K is split across programs without an accumulating loop",
            )
            (a_rows, a_k), a_mask = a._access()
            (b_k, b_columns), b_mask = b._access()
            (rows, columns), mask = c._access()
            require(
                np.all(rows == rows[:, :1]) and np.all(columns == columns[:1, :]),
                "output axes are not independent matrix axes",
            )
            k = np.arange(a.shape[1])
            require(
                np.all(a_rows == rows[:, :1]) and np.all(a_k == k[None, :]),
                "lhs coordinates do not match output rows and complete K",
            )
            require(
                np.all(b_k == k[:, None]) and np.all(b_columns == columns[:1, :]),
                "rhs coordinates do not match complete K and output columns",
            )
            expected_a = (a_rows >= 0) & (a_rows < lhs.shape[0]) & (a_k < lhs.shape[1])
            expected_b = (
                (b_k < rhs.shape[0]) & (b_columns >= 0) & (b_columns < rhs.shape[1])
            )
            expected_c = (
                (rows >= 0)
                & (rows < output.shape[0])
                & (columns >= 0)
                & (columns < output.shape[1])
            )
            require(
                np.array_equal(a_mask, expected_a)
                and np.array_equal(b_mask, expected_b)
                and np.array_equal(mask, expected_c),
                "matrix predicates omit or add logical lanes",
            )
            addresses = (
                rows[mask] * output.strides[0] + columns[mask] * output.strides[1]
            ).tolist()
            require(
                len(addresses) == len(set(addresses))
                and not seen.intersection(addresses),
                "output writes overlap",
            )
            seen.update(addresses)

        require(len(seen) == output.size, "output coverage is incomplete")

    def run(self):
        last_env = dict(self.symbols)

        for flat_index, coordinates in enumerate(
            itertools.product(*(range(size) for size in self.grid))
        ):
            self.program_id = (*coordinates, *((0,) * (3 - len(coordinates))))
            env = dict(self.symbols)

            for value in self.program.inputs:
                actual = self.inputs[value.name]

                if isinstance(actual, np.ndarray) and value.type.kind == "pointer":
                    env[value.name] = Pointer(actual, observer=self.memory)
                elif isinstance(actual, np.ndarray) and (
                    value.type.kind == "tensor"
                    or value.name in {output.name for output in self.program.outputs}
                ):
                    spec = self.specs.get(value.name)
                    shape = self.shapes.get(value.name, ())
                    local = _local_program_index(shape, self.master_shape, flat_index)
                    env[value.name] = TensorRef(
                        actual,
                        spec,
                        self.symbols,
                        outer_index=local,
                        observer=self.memory,
                        _geometry_cache=(
                            ExtractionGeometry()
                            if self.reuse_extraction_geometry
                            else None
                        ),
                    )
                else:
                    env[value.name] = np.asarray(
                        actual, dtype=numpy_dtype(value.type.dtype)
                    )

            if self.scalar_output is None:
                self.block(self.program.blocks[0], env, "entry")
                last_env = env
            else:
                target = env[self.scalar_output]

                if not isinstance(target, TensorRef) or len(target.shape) != 2:
                    raise UnsupportedOperationError(
                        "Scalar matmul/transpose decomposition requires a rank-2 tensor output."
                    )

                _coordinates, valid = target._access()

                for lane in np.ndindex(target.shape):
                    if not valid[lane]:
                        continue

                    self.lane = lane
                    lane_env = dict(env)
                    self.block(self.program.blocks[0], lane_env, "entry")
                    last_env = lane_env

                self.lane = None

        outputs = {}

        for output in self.program.outputs:
            if output.name in self.inputs:
                outputs[output.name] = self.inputs[output.name]
            elif output.name in last_env:
                outputs[output.name] = materialize(last_env[output.name])
            else:
                raise InterpretationError(
                    f"Output `{output.name}` is unavailable for an empty launch."
                )
        return InterpretationResult(outputs, tuple(self.events), self.program)

    def block(self, block, env, path):
        for index, op in enumerate(block.operations):
            location = f"{path}:{index}:{op.opcode}"
            self.location = location

            try:
                if self.memory is None:
                    input_snapshots, mask_snapshot = None, None
                else:
                    with self.memory.paused():
                        input_snapshots, mask_snapshot = self.trace_inputs(op, env)

                if op.opcode == "scf.yield":
                    values = tuple(env[name] for name in op.operands)
                    self.record(op, env, location, input_snapshots, mask_snapshot, ())

                    return values

                if self.memory is None:
                    accesses = None
                    values = self.operation(op, env, location)
                elif self.event_enabled(op):
                    with self.memory.capture() as captured:
                        values = self.operation(op, env, location)

                    accesses = tuple(captured)
                else:
                    with self.memory.paused():
                        values = self.operation(op, env, location)

                    accesses = None

                if len(op.results) == 1:
                    values = (values,)
                elif not op.results:
                    values = ()

                if len(values) != len(op.results):
                    raise ValueError("Operation result arity does not match SSA.")

                for result, value in zip(op.results, values):
                    if (
                        not isinstance(value, (TensorRef, Pointer))
                        and result.type.kind != "tuple"
                    ):
                        dtype = numpy_dtype(result.type.dtype)
                        value = np.asarray(value, dtype=dtype)

                    env[result.name] = value

                self.record(op, env, location, input_snapshots, mask_snapshot, accesses)
            except InterpretationError:
                raise
            except Exception as exc:
                raise InterpretationError(
                    f"Operation {location} at program {self.program_id}: {exc}."
                ) from exc
        return ()

    def event_enabled(self, op):
        if not self.tracing and self.callback is None:
            return False

        if self.program_ids is not None and self.program_id not in self.program_ids:
            return False

        if self.opcodes is not None and op.opcode not in self.opcodes:
            return False
        return True

    def trace_inputs(self, op, env):
        if not self.event_enabled(op):
            return None, None

        snapshots = {}

        for index, name in enumerate(op.operands):
            reference_only = op.opcode in {
                "mem.data_ptr",
                "mem.load",
                "shape.dim",
                "tensor.stride",
                "index.offset",
            } or (op.opcode == "mem.store" and index == 1)

            if name not in snapshots or not reference_only:
                snapshots[name] = _snapshot(env[name], reference_only=reference_only)

        mask = None

        if op.opcode in {"mem.load", "mem.store"}:
            load = op.opcode == "mem.load"
            target = env[op.operands[0 if load else 1]]
            mask_index = 1 if load else 2
            predicate = (
                materialize(env[op.operands[mask_index]])
                if len(op.operands) > mask_index
                else op.attrs.get("mask", True)
            )

            if isinstance(target, Pointer):
                _offsets, valid = target._indices(predicate)
            elif isinstance(target, TensorRef):
                indices = tuple(
                    self._scalar(self.numeric(name, env), "store index")
                    for name in op.attrs.get("indices", ())
                )

                if indices and op.attrs.get("source"):
                    valid = np.asarray(predicate, dtype=bool)
                else:
                    if (
                        indices
                        and target.levels
                        and target.level < len(target.levels) - 1
                    ):
                        target = target.extract(indices)

                    _coords, valid = target._access(predicate)

                    if indices and not (target.levels and target.extracted):
                        selected = np.zeros(valid.shape, dtype=bool)
                        selected[indices] = True
                        valid = valid & selected

                    if self.lane is not None and op.attrs.get("decomposition") in {
                        "matmul",
                        "transpose",
                    }:
                        selected = np.zeros(valid.shape, dtype=bool)
                        selected[self.lane] = True
                        valid = valid & selected
            else:
                raise ValueError("Memory trace requires a checked pointer/tensor.")

            mask = _snapshot(np.asarray(valid, dtype=bool))
        return snapshots, mask

    def record(self, op, env, location, input_snapshots, mask_snapshot, accesses):
        if self.memory is None:
            return

        sequence = self.sequence
        self.sequence += 1

        if not self.event_enabled(op):
            return

        # Region execution can leave a parent capture active while we snapshot.
        with self.memory.paused():
            self._record(
                op, env, location, input_snapshots, mask_snapshot, accesses, sequence
            )

    def _record(
        self, op, env, location, input_snapshots, mask_snapshot, accesses, sequence
    ):
        values = {
            result.name: _snapshot(
                env[result.name], reference_only=isinstance(env[result.name], TensorRef)
            )
            for result in op.results
        }

        if op.opcode == "mem.store":
            destination = env[op.operands[1]]

            if isinstance(destination, TensorRef):
                if self.lane is not None and op.attrs.get("decomposition") in {
                    "matmul",
                    "transpose",
                }:
                    coordinates, _valid = destination._access()
                    stored = destination.array[
                        tuple(coordinate[self.lane] for coordinate in coordinates)
                    ]
                    values[op.operands[1]] = _snapshot(stored)
                else:
                    # A trace must not turn a masked write into an unmasked read.
                    # Store inputs and the effective mask describe the write;
                    # preserve destination provenance without dereferencing it.
                    values[op.operands[1]] = _snapshot(destination, reference_only=True)

        watch = tuple(
            dict.fromkeys((*self.watch, *getattr(self.callback, "watch_symbols", ())))
        )
        watched = {
            name: _snapshot(env[name], reference_only=isinstance(env[name], TensorRef))
            for name in watch
            if name in env
        }
        event = TraceEvent(
            self.program_id,
            location,
            op.opcode,
            values,
            self.iteration,
            watched,
            input_snapshots,
            mask_snapshot,
            self.lane,
            accesses,
            sequence,
        )

        if self.tracing:
            self.events.append(event)

        if self.callback is not None:
            self.callback(event)

    def numeric(self, name, env):
        return materialize(env[name])

    def operation(self, op, env, location):
        code = op.opcode
        args = tuple(env[name] for name in op.operands)

        if code in self.handlers:
            if self.memory is not None:
                self.memory.unknown()
            return self.handlers[code](op, tuple(materialize(value) for value in args))

        if code == "arith.constant":
            value = op.attrs.get("value")

            if isinstance(value, str) and value in {"inf", "-inf"}:
                value = float(value)
            return value

        if code == "scf.for":
            start, stop, step = (
                self._scalar(materialize(value), "loop bound") for value in args[:3]
            )

            if any(int(value) != value for value in (start, stop, step)):
                raise ValueError("Loop bounds and step must be integers.")

            if step == 0:
                raise ValueError("A loop step cannot be zero.")

            carried = tuple(args[3:])
            region = op.regions[0]
            previous = self.iteration

            try:
                for induction in range(int(start), int(stop), int(step)):
                    self.iteration = (*previous, induction)
                    local = dict(env)

                    for parameter, value in zip(
                        region.args, (np.int64(induction), *carried)
                    ):
                        local[parameter.name] = value

                    carried = self.block(region, local, f"{location}/region0")
            finally:
                self.iteration = previous
            return carried[0] if len(op.results) == 1 else carried

        if code == "scf.if":
            condition = bool(self._scalar(materialize(args[0]), "if condition"))
            region_index = 0 if condition else 1
            values = ()

            if region_index < len(op.regions):
                values = self.block(
                    op.regions[region_index],
                    dict(env),
                    f"{location}/region{region_index}",
                )
            return values[0] if len(op.results) == 1 else values

        if code == "mem.data_ptr":
            target = args[0]

            if not isinstance(target, TensorRef):
                raise ValueError("Operation `data_ptr` requires a source tensor.")
            return Pointer(target.array, observer=target.observer)

        if code == "mem.load":
            pointer = args[0]

            if not isinstance(pointer, (Pointer, TensorRef)):
                raise ValueError(
                    "Operation `load` requires a checked pointer or tensor."
                )

            mask = materialize(args[1]) if len(args) > 1 else op.attrs.get("mask", True)
            other = materialize(args[2]) if len(args) > 2 else op.attrs.get("other", 0)

            return pointer.read(mask, other)

        if code == "mem.store":
            value, target = args[:2]
            mask = materialize(args[2]) if len(args) > 2 else op.attrs.get("mask", True)
            indices = tuple(
                self._scalar(self.numeric(name, env), "store index")
                for name in op.attrs.get("indices", ())
            )

            if indices:
                if not isinstance(target, TensorRef):
                    raise ValueError("Indexed store requires a tensor destination.")

                if op.attrs.get("source"):
                    self._source_store(target, indices, materialize(value), mask)
                elif target.levels and target.level < len(target.levels) - 1:
                    target.extract(indices).write(materialize(value), mask)
                else:
                    current = target.read()
                    current[indices] = materialize(value)
                    target.write(current, mask)
            else:
                if not isinstance(target, (Pointer, TensorRef)):
                    raise ValueError(
                        "Operation `store` requires a checked destination."
                    )

                if self.lane is not None and op.attrs.get("decomposition") in {
                    "matmul",
                    "transpose",
                }:
                    selected = np.zeros(target.shape, dtype=bool)
                    selected[self.lane] = True
                    mask = np.asarray(mask, dtype=bool) & selected

                target.write(materialize(value), mask)
            return None

        if code == "tensor.extract":
            indices = tuple(
                self._scalar(materialize(value), "extract index") for value in args[1:]
            )

            if isinstance(args[0], TensorRef):
                if op.attrs.get("source"):
                    self._check_source_indices(args[0].array, indices)

                    if self.memory is not None:
                        self.memory.source("read", args[0].array, indices)

                    return args[0].array[indices]
                return args[0].extract(indices)
            return np.asarray(args[0])[indices]

        if code == "tensor.view":
            subscript = self._subscript(op.attrs.get("subscript", ":"), env)

            return np.asarray(materialize(args[0]))[subscript]

        if code == "shape.dim":
            target = args[0]
            shape = (
                target.array.shape
                if isinstance(target, TensorRef) and op.attrs.get("source")
                else target.shape
            )

            return shape[int(op.attrs.get("dim", 0))]

        if code == "tensor.stride":
            target = args[0]
            dim = int(op.attrs.get("dim", 0))

            if isinstance(target, TensorRef) and op.attrs.get("source"):
                return target.array.strides[dim] // target.array.itemsize
            return math.prod(target.shape[dim + 1 :])

        if code == "index.offset":
            coordinate_space = op.attrs.get("coordinate_space", "source")

            if coordinate_space == "value":
                if self.lane is None or not isinstance(args[0], TensorRef):
                    raise UnsupportedOperationError(
                        "Value-space scalar offsets require an enclosing scalar-lane output domain."
                    )

                dim = int(op.attrs.get("dim", 0))

                if not -len(self.lane) <= dim < len(self.lane):
                    raise InterpretationError(
                        "Value-space offset dimension is outside the logical tile."
                    )
                return self.lane[dim]

            if coordinate_space != "source":
                raise UnsupportedOperationError(
                    f"Unsupported offset coordinate space `{coordinate_space}`."
                )

            if op.attrs.get("decomposition") in {"matmul", "transpose"}:
                if self.lane is None or not isinstance(args[0], TensorRef):
                    raise UnsupportedOperationError(
                        f"Scalar {op.attrs['decomposition']} decomposition at {location}, program {self.program_id}, "
                        "requires an enclosing supported scalar-lane output domain."
                    )

                coordinates, _mask = args[0]._access()

                return coordinates[int(op.attrs.get("dim", 0))][self.lane]
            return self._offset(args[0], op)

        if code == "tensor.cast":
            dtype = op.attrs.get("dtype", op.results[0].type.dtype)

            if isinstance(dtype, str) and dtype.endswith(".dtype"):
                reference = env[dtype[:-6]]
                dtype = (
                    reference.array.dtype
                    if isinstance(reference, TensorRef)
                    else np.asarray(reference).dtype
                )
            elif (
                isinstance(dtype, str)
                and len(dtype) >= 2
                and dtype[0] == dtype[-1]
                and dtype[0] in {"'", '"'}
            ):
                dtype = ast.literal_eval(dtype)

            return np.asarray(materialize(args[0])).astype(numpy_dtype(dtype))

        if code in {"tensor.zeros", "tensor.empty", "tensor.full"}:
            shape = shape_value(op.results[0].type.shape, self.symbols | env)
            dtype = numpy_dtype(op.results[0].type.dtype, "float32")
            value = materialize(args[0]) if args else op.attrs.get("value", 0)

            return np.full(shape, value if code == "tensor.full" else 0, dtype=dtype)

        if code == "select.where":
            return np.where(*(materialize(value) for value in args))

        if code in {"linalg.dot", "linalg.matmul"}:
            return np.matmul(*(materialize(value) for value in args[:2]))

        if code == "linalg.transpose":
            return np.swapaxes(materialize(args[0]), -1, -2)

        if code == "tuple.construct":
            return args

        if code.startswith("reduce."):
            functions = {"sum": np.sum, "max": np.max, "min": np.min}
            kind = code.split(".", 1)[1]

            if kind not in functions:
                return self.unsupported(op, location)

            array = np.asarray(materialize(args[0]))
            axis = op.attrs.get("axis")
            kwargs = {"axis": None if axis is None else int(axis)}

            if kind == "sum":
                kwargs["dtype"] = numpy_dtype(op.results[0].type.dtype, array.dtype)
            return functions[kind](array, **kwargs)

        if code.startswith(("arith.", "cmp.")):
            kind = code.split(".", 1)[1]

            if (
                len(args) == 2
                and isinstance(args[0], Pointer)
                and kind in {"add", "sub"}
            ):
                other = materialize(args[1])

                if isinstance(other, Pointer):
                    if kind != "sub" or args[0].array is not other.array:
                        raise ValueError(
                            "Pointer arithmetic requires the same allocation."
                        )
                    return np.asarray(args[0].offset) - other.offset
                return args[0].shift(other if kind == "add" else -other)

            if len(args) == 2 and isinstance(args[1], Pointer) and kind == "add":
                return args[1].shift(materialize(args[0]))

            aliases = {
                "subtract": "sub",
                "multiply": "mul",
                "and": "bitand",
                "or": "bitor",
                "bitwise_and": "bitand",
                "bitwise_or": "bitor",
                "bitwise_xor": "bitxor",
            }
            kind = aliases.get(kind, kind)
            functions = (
                BINARY
                | UNARY
                | {
                    "maximum": np.maximum,
                    "minimum": np.minimum,
                    "max": np.maximum,
                    "min": np.minimum,
                    "bitwise_left_shift": np.left_shift,
                    "bitwise_right_shift": np.right_shift,
                }
            )

            if kind in functions:
                return functions[kind](*(materialize(value) for value in args))
            return self.unsupported(op, location)

        if code.startswith("math."):
            functions = {
                "exp": np.exp,
                "exp2": np.exp2,
                "log": np.log,
                "log2": np.log2,
                "sqrt": np.sqrt,
                "rsqrt": lambda x: 1 / np.sqrt(x),
                "abs": np.abs,
                "sin": np.sin,
                "cos": np.cos,
                "tanh": np.tanh,
                "floor": np.floor,
                "ceil": np.ceil,
            }
            kind = code.split(".", 1)[1]

            if kind in functions:
                return functions[kind](*(materialize(value) for value in args))
            return self.unsupported(op, location)

        if code in {"call.program_id", "index.program_id", "program.id"}:
            axis = (
                int(self._scalar(materialize(args[0]), "program axis"))
                if args
                else int(op.attrs.get("axis", 0))
            )

            if not 0 <= axis < 3:
                raise ValueError("Program axis must be 0, 1, or 2.")
            return self.program_id[axis]

        if code == "call.num_programs":
            axis = (
                int(self._scalar(materialize(args[0]), "program axis"))
                if args
                else int(op.attrs.get("axis", 0))
            )

            if not 0 <= axis < 3:
                raise ValueError("Program axis must be 0, 1, or 2.")
            return (*self.grid, 1, 1)[axis]

        if code == "call.arange":
            return np.arange(
                *(
                    int(self._scalar(materialize(value), "arange bound"))
                    for value in args
                ),
                dtype=np.int32,
            )
        return self.unsupported(op, location)

    def unsupported(self, op, location):
        raise UnsupportedOperationError(
            f"Unsupported operation `{op.opcode}` at {location}, program {self.program_id}."
        )

    @staticmethod
    def _scalar(value, description):
        array = np.asarray(value)

        if array.ndim != 0:
            raise ValueError(
                f"Value {description} must be scalar, got shape {array.shape}."
            )
        return array.item()

    @staticmethod
    def _check_source_indices(array, indices):
        if len(indices) > array.ndim or any(
            int(index) != index or index < 0 or index >= size
            for index, size in zip(indices, array.shape)
        ):
            raise IndexError("Source index is outside its tensor.")

    def _source_store(self, target, indices, value, mask):
        if np.ndim(mask) != 0:
            raise ValueError("Source scalar store mask must be scalar.")

        if bool(mask):
            self._check_source_indices(target.array, indices)
            target.array[indices] = value

            if self.memory is not None:
                self.memory.source("write", target.array, indices)

    @staticmethod
    def _subscript(text, env):
        # Calling ast.unparse(tuple-of-slices) includes parentheses that are not legal
        # inside a subscript when any tuple item is a slice.
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]

        node = ast.parse(f"_value[{text}]", mode="eval").body.slice

        def item(value):
            if isinstance(value, ast.Tuple):
                return tuple(item(part) for part in value.elts)

            if isinstance(value, ast.Slice):
                return slice(
                    *(
                        None if part is None else item(part)
                        for part in (value.lower, value.upper, value.step)
                    )
                )

            if isinstance(value, ast.Constant):
                return value.value

            if isinstance(value, ast.Name) and value.id in env:
                return int(np.asarray(materialize(env[value.id])))

            if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
                return -item(value.operand)

            raise ValueError(f"Unsupported tensor view subscript `{text}`.")

        return item(node)

    def _offset(self, target, op):
        if not isinstance(target, TensorRef):
            raise ValueError("Operation `offsets` requires an arranged tensor.")

        dimension = int(op.attrs.get("dim", 0) or 0)

        if dimension < 0:
            dimension += target.array.ndim

        if not 0 <= dimension < target.array.ndim:
            raise ValueError("Offset source dimension is invalid.")

        coordinates, _mask = target._access()
        result = coordinates[dimension]
        shape = shape_value(op.results[0].type.shape, self.symbols)

        if result.shape == shape:
            return result

        if not shape:
            return result.reshape(-1)[0]

        if target.levels:
            dimensions = target.levels[target.level].target_dims
            selectors = tuple(
                slice(None)
                if value is not None and int(evaluate(value, self.symbols)) == dimension
                else 0
                for value in dimensions
            )

            if len(selectors) == result.ndim:
                selected = result[selectors]

                if selected.shape == shape:
                    return selected
        return np.reshape(result, shape)


def interpret_program(
    program,
    inputs: Mapping,
    *,
    tensors=(),
    grid=None,
    symbols=None,
    trace=False,
    program_ids=None,
    opcodes=None,
    watch=(),
    callback=None,
    handlers=None,
):
    """Execute an existing SSA Program on CPU buffers without a GPU runtime.

    ``inputs`` binds public SSA names to arrays/scalars. Output arrays are mutated
    in place. Raw pointer memory operations accept an optional mask operand;
    stores retain frontend operand order ``(value, destination, mask)``.
    ``handlers`` optionally maps an opcode to ``handler(operation, operands)``;
    numeric operands are NumPy values and the return value follows SSA arity.
    Already-loaded PyTorch CPU tensors are adapted without copying. Output names
    bound to Torch buffers return the same Torch objects, with writes visible in
    their storage. CUDA, requires-grad and non-strided tensors are rejected.
    """
    if not isinstance(program, ssa.Program):
        raise TypeError("Function `interpret_program` requires an ssa.Program.")

    ssa.verify_program(program)
    inputs, original_tensors = _adapt_inputs(dict(inputs))

    for name, value in inputs.items():
        if not isinstance(value, (np.ndarray, np.generic, bool, int, float)):
            raise TypeError(f"Input `{name}` must be a NumPy array or CPU scalar.")

        try:
            numpy_dtype(np.asarray(value).dtype)
        except ValueError as exc:
            raise InterpretationError(f"Input `{name}`: {exc}.") from exc

    for value in program.inputs:
        actual = inputs.get(value.name)
        declared = numpy_dtype(value.type.dtype)

        if (
            isinstance(actual, (np.ndarray, np.generic))
            and declared is not None
            and actual.dtype != declared
        ):
            raise InterpretationError(
                f"Input `{value.name}` dtype {actual.dtype} does not match declared {declared}."
            )

    resolved = _bind_symbols(program, inputs, tensors, symbols)
    execution = _Execution(
        program,
        inputs,
        tensors,
        resolved,
        grid,
        trace=trace,
        program_ids=program_ids,
        opcodes=opcodes,
        watch=watch,
        callback=callback,
        handlers=handlers,
    )
    result = execution.run()

    if original_tensors:
        result = replace(
            result,
            outputs={
                name: original_tensors.get(name, value)
                for name, value in result.outputs.items()
            },
        )
    return result
