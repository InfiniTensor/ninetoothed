"""CPU differential checks and portable, non-executable SSA replay bundles."""

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ninetoothed.ir import (
    AccessMap,
    IndexExpr,
    LayoutLevel,
    TensorLayout,
    TensorSpec,
    ir_to_dict,
    ssa,
)
from ninetoothed.ir.provenance import (
    operation_locations,
    record_pass,
    seed_origins,
    source_candidates,
)

from .localization import DependencySlice, backward_slice, compare_mapped_results
from .runtime import InterpretationError, _adapt_inputs, interpret_program
from .storage import copy_storage, restore_view


class DebuggerQuit(InterpretationError):
    """The user stopped CPU execution at a completed operation boundary."""


class StepDebugger:
    """A synchronous operation debugger, usable interactively or from a script.

    Pass this object as ``interpret_program(..., callback=debugger)``. Commands
    are read from an optional iterable, otherwise from ``input()``. Exhausting
    a scripted command stream continues execution. Pauses happen after an
    operation has completed; its inputs were captured before execution.
    """

    def __init__(
        self,
        *,
        commands=None,
        output=print,
        breakpoints=(),
        watch=(),
        stop_on_entry=True,
    ):
        self._commands = None if commands is None else iter(commands)
        self.output = output
        self.breakpoints = set(breakpoints)
        self._watch = list(dict.fromkeys(watch))
        self._stepping = bool(stop_on_entry)
        self.values = {}
        self.pauses = []
        self.events_seen = 0
        self.last_event = None

    @property
    def watch_symbols(self):
        """Return names the execution engine should snapshot at each event."""
        return tuple(self._watch)

    def add_breakpoint(self, location):
        """Break at an exact location, location prefix, or opcode name."""
        self.breakpoints.add(str(location))

    def remove_breakpoint(self, location):
        """Remove a breakpoint if it exists."""
        self.breakpoints.discard(str(location))

    def add_watch(self, name):
        """Track an SSA name in subsequent execution events."""
        if name not in self._watch:
            self._watch.append(str(name))

    def remove_watch(self, name):
        """Stop requesting an SSA name in subsequent events."""
        if name in self._watch:
            self._watch.remove(name)

    def inspect(self, name):
        """Return the latest observed snapshot of a value in this program."""
        return self.values[name]

    def step(self):
        """Pause at the next completed operation."""
        self._stepping = True

    def continue_(self):
        """Continue until a breakpoint, without disabling breakpoints."""
        self._stepping = False

    def __call__(self, event):
        if self.last_event is not None and (
            self.last_event.program_id,
            self.last_event.lane,
        ) != (event.program_id, event.lane):
            self.values.clear()

        self.last_event = event
        self.events_seen += 1
        self.values.update(event.inputs or {})
        self.values.update(event.results or {})
        self.values.update(event.watched or {})
        hit = any(
            event.opcode == breakpoint
            or event.location == breakpoint
            or event.location.startswith(f"{breakpoint}:")
            for breakpoint in self.breakpoints
        )

        if not self._stepping and not hit:
            return

        self.pauses.append(event)
        self.output(
            f"paused {event.program_id} {event.location} lane={event.lane} iteration={event.iteration}"
        )

        for name in self._watch:
            if name in self.values:
                self.output(
                    f"watch {name} = {json.dumps(self.values[name], sort_keys=True)}"
                )

        while True:
            if self._commands is None:
                command = input("cpu-debug> ")
            else:
                command = next(self._commands, "continue")

            parts = command.strip().split(maxsplit=1)
            action = parts[0] if parts else "step"
            argument = parts[1] if len(parts) > 1 else None

            if action in {"s", "step"}:
                self.step()

                return

            if action in {"c", "continue"}:
                self.continue_()

                return

            if action in {"q", "quit"}:
                raise DebuggerQuit(
                    f"Debugger stopped at {event.location}, program {event.program_id}."
                )

            if action in {"b", "break"} and argument:
                self.add_breakpoint(argument)
                self.output(f"breakpoint added: {argument}")
            elif action in {"d", "delete"} and argument:
                self.remove_breakpoint(argument)
                self.output(f"breakpoint removed: {argument}")
            elif action in {"w", "watch"} and argument:
                self.add_watch(argument)
                self.output(f"watch added: {argument}")
            elif action in {"p", "print"} and argument:
                self.output(
                    f"{argument} = {json.dumps(self.values.get(argument, '<not observed>'), sort_keys=True)}"
                )
            else:
                self.output(
                    "commands: step, continue, break LOCATION|OPCODE, delete LOCATION|OPCODE, watch NAME, print NAME, quit"
                )


@dataclass(frozen=True)
class OperationDifference:
    """First mismatching corresponding operation in two compatible traces."""

    program_id: tuple
    location: str
    opcode: str
    result_name: str
    iteration: tuple
    lane: tuple | None = None
    component: str = "result"


@dataclass(frozen=True)
class ProgramComparison:
    """Output differences, safe trace locations and declared source candidates.

    Source candidates describe the scope of a recorded transformation. They are
    not evidence of value correspondence or a unique cause of the output error.
    """

    equal: bool
    output_differences: tuple
    first_operation: OperationDifference | None
    traces_aligned: bool
    source_candidates: tuple = ()
    aligned_prefix_operation: OperationDifference | None = None
    reproducer: Path | None = None
    export_error: str | None = None
    retained_operation: "OperationLocalization | None" = None
    mapped_operation: "OperationLocalization | None" = None
    localization: "OperationLocalization | None" = None
    dependency_slice: DependencySlice | None = None
    mapping_issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class OperationLocalization:
    """An observed SSA difference, with its reference and alignment scope.

    This identifies a differing execution event, not unique causal blame.
    A prefix location is valid only before the first unmatched trace event.
    """

    operation: OperationDifference
    reference: str
    traces_aligned: bool
    basis: str = "full_trace"
    reference_location: str | None = None
    projection: str | None = None


@dataclass(frozen=True)
class PassCheck:
    """The first observed semantic failure in a sequence of SSA passes."""

    passed: bool
    checked_passes: tuple
    first_bad_pass: str | None
    difference: ProgramComparison | None = None
    error: str | None = None
    source_candidates: tuple = ()
    adjacent_difference: ProgramComparison | None = None
    localization: OperationLocalization | None = None
    reproducer: Path | None = None
    export_error: str | None = None
    dependency_slice: DependencySlice | None = None


def _copy_array_layout(value, *, strides=None, writeable=None):
    """Copy numeric storage while retaining signed strides and write permission."""
    if value.dtype.kind not in "biufc":
        raise TypeError("Differential replay requires numeric arrays.")

    strides = value.strides if strides is None else tuple(strides)

    if len(strides) != value.ndim or any(
        not isinstance(stride, int) for stride in strides
    ):
        raise ValueError("Array strides must contain one integer per dimension.")

    if value.size:
        extents = tuple(
            (size - 1) * stride for size, stride in zip(value.shape, strides)
        )
        lower = sum(min(0, extent) for extent in extents)
        upper = sum(max(0, extent) for extent in extents)
    else:
        lower = upper = 0

    storage = np.empty(upper - lower + value.itemsize, dtype=np.uint8)
    result = np.ndarray(
        value.shape, dtype=value.dtype, buffer=storage, offset=-lower, strides=strides
    )
    result[...] = value
    result.flags.writeable = value.flags.writeable if writeable is None else writeable

    return result


def _copy_inputs(inputs):
    inputs, _originals = _adapt_inputs(inputs)

    return copy_storage(inputs).values


def _same(first, second, rtol, atol):
    first, second = np.asarray(first), np.asarray(second)

    if first.shape != second.shape or first.dtype != second.dtype:
        return False

    if first.dtype.kind in "fc":
        return bool(np.allclose(first, second, rtol=rtol, atol=atol, equal_nan=True))
    return bool(np.array_equal(first, second))


def _same_snapshot(first, second, rtol, atol):
    if "value" not in first or "value" not in second:
        return first == second

    if first["shape"] != second["shape"] or first["dtype"] != second["dtype"]:
        return False
    return _same(
        np.asarray(first["value"], dtype=first["dtype"]),
        np.asarray(second["value"], dtype=second["dtype"]),
        rtol,
        atol,
    )


def _structure(program):
    def block(value):
        return tuple(
            (
                op.opcode,
                op.operands,
                tuple(result.name for result in op.results),
                tuple(block(region) for region in op.regions),
            )
            for op in value.operations
        )

    return tuple(block(value) for value in program.blocks)


def _event_difference(left, right, rtol, atol, *, inputs_first=False):
    components = [
        ("input", left.inputs or {}, right.inputs or {}),
        ("mask", {"mask": left.mask}, {"mask": right.mask}),
        ("result", left.results, right.results),
    ]

    if not inputs_first:
        components = [components[2], *components[:2]]

    for component, left_values, right_values in components:
        for name in sorted(set(left_values) | set(right_values)):
            lhs, rhs = left_values.get(name), right_values.get(name)

            if (lhs is None or rhs is None) and lhs == rhs:
                continue

            if lhs is None or rhs is None or not _same_snapshot(lhs, rhs, rtol, atol):
                return OperationDifference(
                    right.program_id,
                    right.location,
                    right.opcode,
                    name,
                    right.iteration,
                    right.lane,
                    component,
                )

    return None


def _retained_difference(reference, candidate, first, second, rtol, atol):
    """Compare explicitly retained operations across one recorded pass boundary.

    These are execution observations at unchanged consumers, not an assertion
    that generated intermediates correspond to their source operations. The
    entire filtered event sequence must align, including loops and scalar lanes.
    """
    history = candidate.metadata.get("provenance", {}).get("passes", ())

    if not history:
        return None

    latest = history[-1]

    for key, program in (
        ("input_fingerprint", reference),
        ("output_fingerprint", candidate),
    ):
        if latest[key] != hashlib.sha256(ssa.render(program).encode()).hexdigest():
            return None

    before = dict(operation_locations(reference))
    after = dict(operation_locations(candidate))
    locations = {}

    for relation in latest["relations"]:
        if relation["relation"] != "preserve":
            continue

        source, target = relation["inputs"], relation["outputs"]

        if len(source) != 1 or len(target) != 1:
            return None

        if (
            source[0] not in before
            or target[0] not in after
            or before[source[0]] != after[target[0]]
        ):
            return None

        if target[0] in locations or source[0] in locations.values():
            return None

        locations[target[0]] = source[0]

    sources = set(locations.values())
    left_events = tuple(event for event in first.trace if event.location in sources)
    right_events = tuple(event for event in second.trace if event.location in locations)
    left_keys = tuple(
        (event.program_id, event.location, event.iteration, event.lane)
        for event in left_events
    )
    right_keys = tuple(
        (event.program_id, locations[event.location], event.iteration, event.lane)
        for event in right_events
    )

    if left_keys != right_keys:
        return None

    for left, right in zip(left_events, right_events):
        operation = _event_difference(left, right, rtol, atol, inputs_first=True)

        if operation is not None:
            return OperationLocalization(
                operation,
                "reference",
                False,
                "retained_boundary",
                left.location,
            )

    return None


def _compare_programs(
    reference,
    candidate,
    inputs,
    *,
    tensors=(),
    grid=None,
    symbols=None,
    rtol=1e-3,
    atol=1e-3,
    diagnostics_version=2,
):
    """Compare independent executions without changing the caller's buffers.

    An operation is attributed only when SSA structures and full event ordering
    align. Restructuring passes can be checked by outputs and report declared
    source candidates, but provenance alone never establishes value equivalence.
    """
    options = dict(tensors=tensors, grid=grid, symbols=symbols, trace=True)
    first = interpret_program(reference, _copy_inputs(inputs), **options)
    second = interpret_program(candidate, _copy_inputs(inputs), **options)
    differing = tuple(
        name
        for name in sorted(set(first.outputs) | set(second.outputs))
        if name not in first.outputs
        or name not in second.outputs
        or not _same(first.outputs[name], second.outputs[name], rtol, atol)
    )

    def key(event):
        return (
            event.program_id,
            event.location,
            event.opcode,
            event.iteration,
            event.lane,
        )

    aligned = _structure(reference) == _structure(candidate) and tuple(
        map(key, first.trace)
    ) == tuple(map(key, second.trace))
    prefix_operation = None

    if _structure(reference) == _structure(candidate):
        for left, right in zip(first.trace, second.trace):
            if key(left) != key(right):
                break

            prefix_operation = _event_difference(left, right, rtol, atol)

            if prefix_operation is not None:
                break

    first_operation = prefix_operation if aligned else None
    candidates = (
        source_candidates(
            candidate,
            (first_operation.location,) if first_operation is not None else None,
        )
        if differing
        else ()
    )

    retained = (
        _retained_difference(reference, candidate, first, second, rtol, atol)
        if differing and not aligned
        else None
    )
    mismatch, mapping_issues = compare_mapped_results(
        reference, candidate, first, second, _same_snapshot, rtol, atol
    )
    mapped = None

    if differing and mismatch is not None:
        left, right = (
            first.trace[mismatch.reference_index],
            second.trace[mismatch.candidate_index],
        )
        mapped = OperationLocalization(
            OperationDifference(
                right.program_id,
                right.location,
                right.opcode,
                mismatch.candidate_result,
                right.iteration,
                right.lane,
            ),
            "reference",
            False,
            "mapped_result",
            left.location,
            projection=mismatch.projection,
        )

    observations = [item for item in (retained, mapped) if item is not None]

    if differing and prefix_operation is not None:
        observations.append(
            OperationLocalization(
                prefix_operation,
                "reference",
                aligned,
                "full_trace" if aligned else "aligned_prefix",
                prefix_operation.location,
            )
        )

    def observation_index(item):
        operation = item.operation

        return next(
            index
            for index, event in enumerate(second.trace)
            if (
                event.program_id == operation.program_id
                and event.location == operation.location
                and event.iteration == operation.iteration
                and event.lane == operation.lane
            )
        )

    localization = min(observations, key=observation_index, default=None)
    dependencies = (
        None
        if localization is None
        else backward_slice(
            candidate,
            second.trace,
            localization.operation,
            memory=diagnostics_version >= 2,
        )
    )

    return ProgramComparison(
        not differing,
        differing,
        first_operation,
        aligned,
        candidates,
        prefix_operation,
        retained_operation=retained,
        mapped_operation=mapped,
        localization=localization,
        dependency_slice=dependencies,
        mapping_issues=mapping_issues,
    )


def _capture_failure(directory, reference, candidate, inputs, **options):
    if directory is None:
        return None, None

    from .failure import export_failure

    try:
        return export_failure(directory, reference, candidate, inputs, **options), None
    except Exception as exc:
        # Diagnostic I/O must not replace the original semantic failure.
        return None, f"{type(exc).__name__}: {exc}"


def compare_programs(
    reference,
    candidate,
    inputs,
    *,
    tensors=(),
    grid=None,
    symbols=None,
    rtol=1e-3,
    atol=1e-3,
    failure_dir=None,
    seed=None,
):
    """Compare independent CPU executions and optionally capture any failure.

    Set ``failure_dir`` to a new directory to export SSA, exact numeric inputs,
    tolerance, seed and an executable differential replay automatically. An
    export failure is reported separately and never hides the semantic failure.
    Execution errors keep their exception type and gain ``reproducer`` and
    ``export_error`` attributes when capture is enabled.

    ``first_operation`` still requires full structural/event alignment.
    ``aligned_prefix_operation`` can additionally identify a difference before
    control flow diverges, but never after the first unmatched event. Neither
    field guesses equivalence across a restructuring pass from origins alone.
    ``retained_operation`` instead reports the earliest differing observation
    at a pass-declared unchanged operation, when its filtered traces align.
    ``mapped_operation`` checks an explicit result equality, including declared
    tile-to-lane projections. ``localization`` selects the earliest available
    observation; ``dependency_slice`` follows executed value and checked
    same-program memory dependencies, with explicit uncertainty boundaries.
    """
    options = dict(
        tensors=tuple(tensors), grid=grid, symbols=symbols, rtol=rtol, atol=atol
    )

    try:
        report = _compare_programs(reference, candidate, inputs, **options)
    except (InterpretationError, ValueError, TypeError) as exc:
        if failure_dir is not None:
            exc.reproducer, exc.export_error = _capture_failure(
                failure_dir,
                reference,
                candidate,
                inputs,
                **options,
                seed=seed,
                error=exc,
                phase="verification"
                if isinstance(exc, ssa.VerificationError)
                else "execution",
            )

        raise

    if not report.equal:
        directory, error = _capture_failure(
            failure_dir,
            reference,
            candidate,
            inputs,
            **options,
            seed=seed,
            report=report,
        )
        report = replace(report, reproducer=directory, export_error=error)

    return report


def check_passes(
    program,
    passes,
    inputs,
    *,
    tensors=(),
    grid=None,
    symbols=None,
    rtol=1e-3,
    atol=1e-3,
    failure_dir=None,
    seed=None,
):
    """Stop at the first bad named ``(name, Program -> Program)`` pass.

    Adapters can call existing pass objects with their genuine pass Context.
    Compare both the adjacent pass boundary and the original semantic reference,
    preventing accumulated small differences from escaping detection. The
    original must execute successfully before any pass can be blamed.

    ``difference`` retains the original-reference comparison. The additional
    ``adjacent_difference`` and ``localization`` describe the failing boundary
    without implying that restructured SSA values are automatically equivalent.
    With ``failure_dir``, capture the original, last-good and candidate SSA and
    the exact inputs immediately on the first failure; later passes do not run.
    """
    program = seed_origins(program)
    current = program
    checked = []
    inputs = _copy_inputs(inputs)
    tensors = tuple(tensors)
    options = dict(tensors=tensors, grid=grid, symbols=symbols, rtol=rtol, atol=atol)
    interpret_program(
        program, _copy_inputs(inputs), tensors=tensors, grid=grid, symbols=symbols
    )

    for name, transform in passes:
        checked.append(str(name))
        transformed = None
        previous = current
        phase = "transform"

        try:
            transformed = transform(previous)
            phase = "record"

            if not isinstance(transformed, ssa.Program):
                raise TypeError("An SSA pass must return an SSA Program.")

            transformed = record_pass(previous, transformed, str(name))
            ssa.verify_program(transformed)
            current = transformed
            phase = "execution"
            adjacent = _compare_programs(previous, current, inputs, **options)
            difference = (
                adjacent
                if previous is program
                else _compare_programs(program, current, inputs, **options)
            )
        except Exception as exc:
            # Python pass exceptions are diagnostics, never successful checks.
            report = PassCheck(
                False,
                tuple(checked),
                str(name),
                error=str(exc),
                source_candidates=(
                    source_candidates(transformed) if phase == "execution" else ()
                ),
            )
            directory, error = _capture_failure(
                failure_dir,
                program,
                transformed,
                inputs,
                **options,
                seed=seed,
                previous=previous,
                report=report,
                error=exc,
                phase=phase,
            )

            return replace(report, reproducer=directory, export_error=error)

        if not difference.equal or not adjacent.equal:
            localization = None
            dependencies = None

            for reference_name, comparison in (
                ("previous", adjacent),
                ("original", difference),
            ):
                if not comparison.equal and comparison.localization is not None:
                    localization = replace(
                        comparison.localization, reference=reference_name
                    )
                    dependencies = comparison.dependency_slice
                    break

            report = PassCheck(
                False,
                tuple(checked),
                str(name),
                difference,
                source_candidates=(
                    difference.source_candidates
                    if not difference.equal
                    else adjacent.source_candidates
                ),
                adjacent_difference=adjacent,
                localization=localization,
                dependency_slice=dependencies,
            )
            directory, error = _capture_failure(
                failure_dir,
                program,
                current,
                inputs,
                **options,
                seed=seed,
                previous=previous,
                report=report,
            )

            return replace(report, reproducer=directory, export_error=error)
    return PassCheck(True, tuple(checked), None)


def export_reproducer(
    directory, program, inputs, *, tensors=(), grid=None, symbols=None, seed=None
):
    """Save the supplied case as JSON SSA, numeric NPZ inputs and a replay script.

    This exports exactly the provided case; it does not claim to minimize its
    shapes or operations. Object arrays, pickles and executable SSA are excluded.
    Existing bundle files are never overwritten.
    Distinct overlapping views use checked byte storage; ordinary inputs keep
    the original schema. Allocation gaps are zero-filled, never copied.
    """
    inputs, _originals = _adapt_inputs(inputs)
    directory = Path(directory)
    files = ("program.json", "program.ssa", "inputs.npz", "manifest.json", "replay.py")

    if any((directory / filename).exists() for filename in files):
        raise FileExistsError("Refusing to overwrite an existing replay bundle.")

    ssa.verify_program(program)
    copied = copy_storage(inputs)
    arrays, bindings, aliases, names_by_id = {}, {}, {}, {}

    if copied.shared:
        arrays, bindings, aliases = (
            dict(copied.buffers),
            dict(copied.bindings),
            dict(copied.aliases),
        )

    for index, (name, value) in enumerate(inputs.items()):
        if copied.shared and isinstance(value, np.ndarray):
            continue

        if isinstance(value, np.ndarray) and id(value) in names_by_id:
            aliases[name] = names_by_id[id(value)]
            continue

        array = np.asarray(value)

        if array.dtype.kind not in "biufc":
            raise TypeError(f"Reproducer input `{name}` is not a numeric array/scalar.")

        key = f"input_{index}"
        arrays[key] = array
        bindings[name] = {
            "key": key,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "strides": list(array.strides),
            "writeable": bool(array.flags.writeable),
            "scalar": not isinstance(value, np.ndarray),
        }

        if isinstance(value, np.ndarray):
            names_by_id[id(value)] = name

    metadata = {
        "schema": 2 if copied.shared else 1,
        "seed": seed,
        "inputs": bindings,
        "aliases": aliases,
        "grid": grid,
        "symbols": dict(symbols or {}),
        "tensors": [ir_to_dict(spec) for spec in tensors],
    }
    program_text = json.dumps(ir_to_dict(program), indent=2)
    manifest_text = json.dumps(metadata, indent=2)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "program.json").write_text(program_text, encoding="utf-8")
    (directory / "program.ssa").write_text(ssa.render(program), encoding="utf-8")
    np.savez_compressed(directory / "inputs.npz", **arrays)
    (directory / "manifest.json").write_text(manifest_text, encoding="utf-8")
    (directory / "replay.py").write_text(
        '"""Replay this exact CPU case using an installed NineToothed checkout."""\n'
        "from pathlib import Path\n"
        "from ninetoothed.interpreter import interpret_program\n"
        "from ninetoothed.interpreter.debugger import load_reproducer\n"
        "program, inputs, options = load_reproducer(Path(__file__).parent)\n"
        "result = interpret_program(program, inputs, **options)\n"
        "print({name: (value.shape, str(value.dtype)) for name, value in result.outputs.items()})\n",
        encoding="utf-8",
    )

    return directory


def _program(data):
    def value(item):
        return ssa.Value(name=item["name"], type=ssa.Type(**item["type"]))

    def block(item):
        return ssa.Block(
            name=item["name"],
            args=tuple(value(arg) for arg in item["args"]),
            operations=tuple(
                ssa.Operation(
                    opcode=op["opcode"],
                    operands=tuple(op["operands"]),
                    results=tuple(value(result) for result in op["results"]),
                    attrs=op["attrs"],
                    regions=tuple(block(region) for region in op["regions"]),
                    origins=tuple(op.get("origins", ())),
                )
                for op in item["operations"]
            ),
        )

    return ssa.verify_program(
        ssa.Program(
            kind=data["kind"],
            inputs=tuple(value(item) for item in data["inputs"]),
            outputs=tuple(value(item) for item in data["outputs"]),
            blocks=tuple(block(item) for item in data["blocks"]),
            metadata=data["metadata"],
        )
    )


def _tensor(data):
    def expr(item):
        return IndexExpr(
            op=item["op"],
            value=item["value"],
            operands=tuple(expr(child) for child in item["operands"]),
        )

    def access(item):
        return AccessMap(
            source_indices=tuple(expr(value) for value in item["source_indices"]),
            linear_index=expr(item["linear_index"]),
            predicate=expr(item["predicate"]),
        )

    data = dict(data)
    layout = data["layout"]

    if layout is not None:
        data["layout"] = TensorLayout(
            **{
                name: tuple(expr(value) for value in layout[name])
                for name in (
                    "source_shape",
                    "source_strides",
                    "view_shape",
                    "application_shape",
                )
            },
            levels=tuple(
                LayoutLevel(
                    shape=tuple(expr(value) for value in level["shape"]),
                    target_dims=tuple(
                        None if value is None else expr(value)
                        for value in level["target_dims"]
                    ),
                )
                for level in layout["levels"]
            ),
            view_access=None
            if layout["view_access"] is None
            else access(layout["view_access"]),
            value_accesses=tuple(access(value) for value in layout["value_accesses"]),
        )
    return TensorSpec(**data)


def load_reproducer(directory):
    """Load only structured JSON and non-pickled numeric arrays from a bundle."""
    directory = Path(directory)
    metadata = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))

    if type(metadata.get("schema")) is not int or metadata["schema"] not in {1, 2}:
        raise ValueError("Unsupported replay bundle schema.")

    program = _program(
        json.loads((directory / "program.json").read_text(encoding="utf-8"))
    )
    inputs = {}

    with np.load(directory / "inputs.npz", allow_pickle=False) as arrays:
        buffers = {}

        for name, binding in metadata["inputs"].items():
            if metadata["schema"] == 2 and "storage" in binding:
                key = binding["storage"]

                if key not in buffers:
                    buffers[key] = arrays[key].copy()

                inputs[name] = restore_view(buffers[key], binding)
                continue

            array = arrays[binding["key"]].copy()

            if (
                list(array.shape) != binding["shape"]
                or str(array.dtype) != binding["dtype"]
            ):
                raise ValueError(f"Input `{name}` disagrees with the bundle manifest.")

            inputs[name] = (
                array[()]
                if binding["scalar"]
                else _copy_array_layout(
                    array,
                    strides=binding.get("strides"),
                    writeable=binding.get("writeable", True),
                )
            )

    for name, source in metadata.get("aliases", {}).items():
        inputs[name] = inputs[source]
    return (
        program,
        inputs,
        {
            "tensors": tuple(_tensor(item) for item in metadata["tensors"]),
            "grid": metadata["grid"],
            "symbols": metadata["symbols"],
        },
    )


__all__ = [
    "StepDebugger",
    "DebuggerQuit",
    "compare_programs",
    "check_passes",
    "export_reproducer",
    "load_reproducer",
    "ProgramComparison",
    "PassCheck",
    "OperationDifference",
    "OperationLocalization",
]
