"""Differential debugging for the CPU reference interpreter.

The interpreter earns its keep when it compares two runs of the same program
instead of inspecting one run on its own.  Two comparisons come up most often:
before and after a pass pipeline, where a differing output means the pipeline
broke semantics, and a reference run against a suspect one, for example a CPU
interpretation versus a GPU execution whose outputs were saved with
``numpy.save``.

Each comparison yields structured data for tests, a readable report, and a
self-contained reproduction snippet.
"""

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .api import Interpretation, interpret
from .errors import ProgramDomainError
from .trace import TraceEvent

#: Tolerance used when comparing floating point outputs by default.  The project
#: standard for ``float32`` against a NumPy/PyTorch reference is ``1e-3``; integer
#: and boolean outputs are always compared exactly.
DEFAULT_RTOL = 1e-3

#: Absolute tolerance used when comparing floating point outputs by default.
DEFAULT_ATOL = 1e-3

#: Maximum number of mismatching indices reported per output.
_MAX_REPORTED_MISMATCHES = 8


def _as_array(value):
    return np.asarray(value)


def _compare_arrays(left, right, *, rtol, atol):
    """Return ``(matches, mismatches, total, max_error, first_indexes)``."""
    left = _as_array(left)
    right = _as_array(right)

    if left.shape != right.shape:
        return False, int(left.size), int(left.size), float("inf"), ()

    total = int(left.size)

    if left.dtype.kind in "biu" and right.dtype.kind in "biu":
        differing = left != right
        max_error = float(
            np.max(np.abs(left.astype(np.float64) - right)) if total else 0.0
        )
    else:
        with np.errstate(all="ignore"):
            difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
            tolerance = atol + rtol * np.abs(right.astype(np.float64))
            differing = ~(difference <= tolerance)
            difference = np.where(np.isnan(difference), np.inf, difference)

        max_error = float(difference.max()) if total else 0.0

    mismatches = int(differing.sum())

    if mismatches == 0:
        return True, 0, total, max_error, ()

    indexes = np.argwhere(differing)

    return (
        False,
        mismatches,
        total,
        max_error,
        tuple(
            tuple(int(item) for item in index)
            for index in indexes[:_MAX_REPORTED_MISMATCHES]
        ),
    )


@dataclass
class OutputDiff:
    """The comparison of one output tensor between two runs."""

    name: str
    matches: bool
    mismatches: int = 0
    total: int = 0
    max_error: float = 0.0
    shape: tuple = ()
    dtype: str = ""
    first_mismatches: tuple = ()
    reason: str | None = None

    def to_dict(self):
        """Return a JSON-friendly representation."""
        return {
            "name": self.name,
            "matches": self.matches,
            "mismatches": self.mismatches,
            "total": self.total,
            "max_error": self.max_error,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "first_mismatches": [list(index) for index in self.first_mismatches],
            "reason": self.reason,
        }

    def render(self):
        """Render this comparison as one line of text."""
        if self.matches:
            return f"  [match] {self.name} <{self.dtype}{self.shape}>"

        if self.reason is not None:
            return f"  [DIFF ] {self.name}: {self.reason}"

        return (
            f"  [DIFF ] {self.name} <{self.dtype}{self.shape}> "
            f"{self.mismatches}/{self.total} element(s) differ, "
            f"max error {self.max_error:.6g}, first at {self.first_mismatches[:3]}"
        )


@dataclass
class TraceDivergence:
    """The first point where two traces disagree."""

    sequence: int
    program_id: int
    opcode: str
    reason: str
    left: str | None = None
    right: str | None = None

    def to_dict(self):
        """Return a JSON-friendly representation."""
        return {
            "sequence": self.sequence,
            "program_id": self.program_id,
            "opcode": self.opcode,
            "reason": self.reason,
            "left": self.left,
            "right": self.right,
        }

    def render(self):
        """Render this divergence as one line of text."""
        lines = [
            f"  [DIFF ] first divergence at event {self.sequence} "
            f"(program {self.program_id}, `{self.opcode}`): {self.reason}"
        ]

        if self.left is not None:
            lines.append(f"          left : {self.left}")

        if self.right is not None:
            lines.append(f"          right: {self.right}")

        return "\n".join(lines)


@dataclass
class TraceDiff:
    """The comparison of two execution traces."""

    comparable: bool
    divergences: tuple = ()
    left_events: int = 0
    right_events: int = 0
    reason: str | None = None

    @property
    def matches(self):
        """Return whether the two traces agree."""
        return self.comparable and not self.divergences

    def to_dict(self):
        """Return a JSON-friendly representation."""
        return {
            "comparable": self.comparable,
            "matches": self.matches,
            "left_events": self.left_events,
            "right_events": self.right_events,
            "reason": self.reason,
            "divergences": [item.to_dict() for item in self.divergences],
        }

    def render(self):
        """Render the trace comparison as text."""
        if not self.comparable:
            return f"  [skip ] traces not comparable: {self.reason}"

        if self.matches:
            return f"  [match] traces agree ({self.left_events} event(s))"

        return "\n".join(item.render() for item in self.divergences)


@dataclass
class ProgramDiff:
    """The full comparison of two interpretations of the same program.

    :param outputs: ``{tensor_name: OutputDiff}``.
    :param trace: The trace comparison, when both runs were traced.
    :param left: The reference interpretation.
    :param right: The interpretation under test.
    :param label: A human-readable description of what is being compared.
    """

    outputs: dict = field(default_factory=dict)
    trace: TraceDiff | None = None
    left: Interpretation | None = None
    right: Interpretation | None = None
    label: str = "interpretation"

    @property
    def matches(self):
        """Return whether every output (and the trace) agrees."""
        outputs_match = all(item.matches for item in self.outputs.values())

        return outputs_match and (self.trace is None or self.trace.matches)

    @property
    def mismatching_outputs(self):
        """Return the names of the outputs that differ."""
        return tuple(name for name, item in self.outputs.items() if not item.matches)

    def to_dict(self):
        """Return a JSON-friendly representation."""
        return {
            "label": self.label,
            "matches": self.matches,
            "mismatching_outputs": list(self.mismatching_outputs),
            "outputs": {name: item.to_dict() for name, item in self.outputs.items()},
            "trace": None if self.trace is None else self.trace.to_dict(),
        }

    def to_json(self, *, indent=2):
        """Return the comparison serialized as JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def render(self):
        """Render the comparison as readable text."""
        lines = [
            f"diff: {self.label}",
            f"result: {'MATCH' if self.matches else 'DIFF'}",
            "outputs:",
        ]
        lines.extend(item.render() for item in self.outputs.values())

        if self.trace is not None:
            lines.append("trace:")
            lines.append(self.trace.render())

        return "\n".join(lines)

    def reproduction(self):
        """Return a :class:`Reproduction` for the interpretation under test.

        :raises ProgramDomainError: When the comparison carries no right-hand run.
        """
        if self.right is None:
            raise ProgramDomainError(
                "A reproduction requires the interpretation under test."
            )

        return build_reproduction(self.right, label=self.label)

    def minimal_reproduction(self):
        """Return a self-contained snippet that reproduces the failing run.

        The snippet embeds the SSA program, the launch shape, the symbol values,
        the input arrays with their shapes and dtypes, and the recorded random
        seed, so it can be pasted into a test without the original arrangement and
        application.
        """
        return self.reproduction().render()


def compare_interpretations(
    left,
    right,
    *,
    rtol=DEFAULT_RTOL,
    atol=DEFAULT_ATOL,
    compare_trace=True,
    label="interpretation",
):
    """Compare two :class:`~ninetoothed.interpret.Interpretation` results.

    :param left: The reference interpretation.
    :param right: The interpretation under test.
    :param rtol: Relative tolerance for floating point outputs.
    :param atol: Absolute tolerance for floating point outputs.
    :param compare_trace: Also compare the recorded traces, when present.
    :param label: A human-readable description of the comparison.
    :return: A :class:`ProgramDiff`.
    """
    outputs = {}
    names = list(dict.fromkeys((*left.outputs, *right.outputs)))

    for name in names:
        if name not in left.outputs:
            outputs[name] = OutputDiff(
                name=name,
                matches=False,
                reason="missing from the reference interpretation",
            )

            continue

        if name not in right.outputs:
            outputs[name] = OutputDiff(
                name=name,
                matches=False,
                reason="missing from the interpretation under test",
            )

            continue

        expected = left.outputs[name]
        actual = right.outputs[name]
        matches, mismatches, total, max_error, first = _compare_arrays(
            actual, expected, rtol=rtol, atol=atol
        )
        outputs[name] = OutputDiff(
            name=name,
            matches=matches,
            mismatches=mismatches,
            total=total,
            max_error=max_error,
            shape=tuple(np.asarray(expected).shape),
            dtype=str(np.asarray(expected).dtype),
            first_mismatches=first,
        )

    trace = None

    if compare_trace and (left.trace or right.trace):
        trace = compare_traces(left.trace, right.trace)

    return ProgramDiff(
        outputs=outputs, trace=trace, left=left, right=right, label=label
    )


def compare_traces(left_events, right_events):
    """Compare two traces event by event.

    Only the shape of the execution is compared: the opcode sequence, the program
    instance, the region depth and the applied mask.  Recorded values are
    reported for the first divergence only, which is usually enough to see where
    the two runs parted ways.

    :param left_events: The reference trace events.
    :param right_events: The trace events under test.
    :return: A :class:`TraceDiff`.
    """
    left_events = tuple(left_events)
    right_events = tuple(right_events)
    divergences = []
    count = max(len(left_events), len(right_events))

    for index in range(count):
        left = left_events[index] if index < len(left_events) else None
        right = right_events[index] if index < len(right_events) else None

        if left is None or right is None:
            present = left if left is not None else right
            side = "left" if left is not None else "right"
            divergences.append(
                TraceDivergence(
                    sequence=index,
                    program_id=present.program_id,
                    opcode=present.opcode,
                    reason=f"event only present in the {side} trace",
                    left=_render_event(left),
                    right=_render_event(right),
                )
            )

            break

        if (left.program_id, left.opcode, left.depth) != (
            right.program_id,
            right.opcode,
            right.depth,
        ):
            divergences.append(
                TraceDivergence(
                    sequence=index,
                    program_id=left.program_id,
                    opcode=left.opcode,
                    reason=(
                        "different operation: left is "
                        f"p{left.program_id}/{left.opcode}@d{left.depth}, right is "
                        f"p{right.program_id}/{right.opcode}@d{right.depth}"
                    ),
                    left=_render_event(left),
                    right=_render_event(right),
                )
            )

            break

        if left.mask != right.mask:
            divergences.append(
                TraceDivergence(
                    sequence=index,
                    program_id=left.program_id,
                    opcode=left.opcode,
                    reason=f"different access mask: {left.mask!r} vs {right.mask!r}",
                    left=_render_event(left),
                    right=_render_event(right),
                )
            )

            break

    return TraceDiff(
        comparable=True,
        divergences=tuple(divergences),
        left_events=len(left_events),
        right_events=len(right_events),
    )


def _render_event(event: TraceEvent | None):
    if event is None:
        return None

    return (
        f"p{event.program_id} {event.opcode} "
        f"({', '.join(item.summary for item in event.operands)})"
        + (f" mask={event.mask}" if event.mask else "")
    )


def compare_pipeline(
    arrangement,
    application,
    tensors=(),
    inputs=None,
    *,
    pipeline,
    symbols=None,
    rtol=DEFAULT_RTOL,
    atol=DEFAULT_ATOL,
    trace=False,
    **options,
):
    """Interpret a program before and after a pass pipeline, then compare.

    Running the same program with and without the pipeline checks that the
    pipeline preserves semantics, without a GPU.

    :param arrangement: The arrangement function.
    :param application: The application function.
    :param tensors: Symbolic tensors or runtime arrays.
    :param inputs: Runtime arrays.
    :param pipeline: The SSA pass pipeline to validate.
    :param symbols: Values for symbolic dimensions and constexpr parameters.
    :param rtol: Relative tolerance for floating point outputs.
    :param atol: Absolute tolerance for floating point outputs.
    :param trace: Record traces for both runs.
    :param options: Extra keyword arguments forwarded to
        :func:`~ninetoothed.interpret.interpret`.
    :return: A :class:`ProgramDiff` whose ``left`` is the unpipelined run and
        whose ``right`` is the pipelined run.
    """
    from .trace import Tracer

    reference = interpret(
        arrangement,
        application,
        tensors=tensors,
        inputs=inputs,
        symbols=symbols,
        tracer=Tracer() if trace else None,
        **options,
    )
    pipelined = interpret(
        arrangement,
        application,
        tensors=tensors,
        inputs=inputs,
        symbols=symbols,
        pipeline=pipeline,
        tracer=Tracer() if trace else None,
        **options,
    )

    return compare_interpretations(
        reference,
        pipelined,
        rtol=rtol,
        atol=atol,
        label=f"pipeline {pipeline!r}",
    )


# -- per-pass bisection ----------------------------------------------------


@dataclass
class PassStage:
    """One cumulative pass prefix, and what it did to the program.

    Stage ``0`` is the program exactly as the Python frontend produced it; stage
    ``i`` is the program after the first ``i`` passes of the pipeline.

    :param index: The position of the stage in the sequence.
    :param name: The pass that produced this stage (``""`` for the frontend).
    :param interpretation: The interpretation of this stage, when it ran.
    :param diff: The comparison against the baseline stage.
    :param diff_previous: The comparison against the previous stage.
    :param error: Why the stage could not be interpreted, when it could not.
    :param pipeline: The pass prefix that produced this stage.
    """

    index: int
    name: str = ""
    interpretation: Interpretation | None = None
    diff: ProgramDiff | None = None
    diff_previous: ProgramDiff | None = None
    error: str | None = None
    pipeline: Any = None

    @property
    def executed(self):
        """Return whether the stage produced an interpretation."""
        return self.interpretation is not None

    @property
    def matches_baseline(self):
        """Return whether the stage still agrees with the baseline."""
        return self.diff is not None and self.diff.matches

    @property
    def label(self):
        """Return the human-readable name of the stage."""
        return self.name or "<frontend>"

    def to_dict(self):
        """Return a JSON-friendly representation."""
        return {
            "index": self.index,
            "name": self.name,
            "label": self.label,
            "executed": self.executed,
            "matches_baseline": self.matches_baseline,
            "error": self.error,
            "diff": None if self.diff is None else self.diff.to_dict(),
        }

    def render(self):
        """Render this stage as one line of text."""
        if not self.executed:
            return f"  [skip ] {self.index:>2} {self.label}: {self.error}"

        if self.diff is None:
            return f"  [base ] {self.index:>2} {self.label}"

        if self.diff.matches:
            return f"  [match] {self.index:>2} {self.label}"

        names = ", ".join(self.diff.mismatching_outputs) or "outputs"

        return f"  [DIFF ] {self.index:>2} {self.label} -> {names} differ"


@dataclass
class Localization:
    """A semantic difference pinned on one program instance and one operation.

    :param pass_name: The pass that first changed the observable behaviour.
    :param output: The output tensor that differs.
    :param index: The first differing index inside that output.
    :param expected: The value the baseline stage produced there.
    :param actual: The value the diverging stage produced there.
    :param program_id: The program instance that writes that element.
    :param operations: ``((opcode, location), ...)`` for the stores that write the
        output in the diverging program.
    """

    pass_name: str
    output: str
    index: tuple
    expected: Any
    actual: Any
    program_id: int | None = None
    operations: tuple = ()

    def to_dict(self):
        """Return a JSON-friendly representation."""
        return {
            "pass": self.pass_name,
            "output": self.output,
            "index": list(self.index),
            "expected": _jsonable(self.expected),
            "actual": _jsonable(self.actual),
            "program_id": self.program_id,
            "operations": [list(item) for item in self.operations],
        }

    def render(self):
        """Render the localization as readable text."""
        lines = [
            f"first semantic difference introduced by `{self.pass_name}`",
            f"  output      : {self.output}{tuple(self.index)}",
            f"  expected    : {self.expected!r}",
            f"  actual      : {self.actual!r}",
        ]

        if self.program_id is not None:
            lines.append(f"  program id  : {self.program_id}")

        if self.operations:
            lines.append("  stores      :")

            for _opcode, location in self.operations:
                lines.append(f"    {location}")

        return "\n".join(lines)


@dataclass
class PipelineDiff:
    """The stage-by-stage comparison of a pass pipeline against the frontend.

    :param stages: The :class:`PassStage` sequence, baseline first.
    :param pass_names: The pass names the pipeline was expanded to.
    :param label: A human-readable description of the comparison.
    :param rtol: The relative tolerance used for the comparisons.
    :param atol: The absolute tolerance used for the comparisons.
    """

    stages: tuple = ()
    pass_names: tuple = ()
    label: str = "pass pipeline"
    rtol: float = DEFAULT_RTOL
    atol: float = DEFAULT_ATOL

    @property
    def baseline(self):
        """Return the frontend stage."""
        return self.stages[0] if self.stages else None

    @property
    def executed(self):
        """Return whether every stage could be interpreted."""
        return bool(self.stages) and all(stage.executed for stage in self.stages)

    @property
    def matches(self):
        """Return whether every executed stage agrees with the baseline.

        An unexecutable stage is unknown, not agreement, so a scan whose baseline
        never ran reports no match.
        """
        baseline = self.baseline

        if baseline is None or not baseline.executed:
            return False

        return all(
            stage.diff is None or stage.diff.matches
            for stage in self.stages
            if stage.executed
        )

    @property
    def first_divergence(self):
        """Return the first stage that changed the observable behaviour."""
        for stage in self.stages:
            if stage.diff is not None and not stage.diff.matches:
                return stage

        return None

    @property
    def diverging_passes(self):
        """Return the names of every pass that changes the observable behaviour."""
        return tuple(
            stage.name
            for stage in self.stages
            if stage.diff is not None and not stage.diff.matches
        )

    def to_dict(self):
        """Return a JSON-friendly representation."""
        return {
            "label": self.label,
            "matches": self.matches,
            "executed": self.executed,
            "pass_names": list(self.pass_names),
            "first_divergence": None
            if self.first_divergence is None
            else self.first_divergence.name,
            "diverging_passes": list(self.diverging_passes),
            "stages": [stage.to_dict() for stage in self.stages],
        }

    def to_json(self, *, indent=2):
        """Return the comparison serialized as JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def render(self):
        """Render the stage-by-stage comparison as readable text."""
        if not self.executed:
            outcome = "UNKNOWN (at least one stage could not be interpreted)"
        elif self.matches:
            outcome = "MATCH"
        else:
            outcome = "DIFF"

        lines = [
            f"pipeline diff: {self.label}",
            f"passes: {', '.join(self.pass_names) or '<none>'}",
            f"result: {outcome}",
            "stages:",
        ]
        lines.extend(stage.render() for stage in self.stages)

        divergence = self.first_divergence

        if divergence is not None:
            lines.append(
                f"first diverging pass: `{divergence.name}` "
                f"(stage {divergence.index} of {len(self.pass_names)})"
            )

        return "\n".join(lines)

    def localize(self):
        """Pin the first divergence on a program instance and an SSA operation.

        :return: A :class:`Localization`, or ``None`` when the pipeline agrees
            with the frontend or the difference cannot be narrowed down.
        """
        divergence = self.first_divergence

        if divergence is None or self.baseline is None:
            return None

        return localize_divergence(self.baseline, divergence)

    def reproduction(self):
        """Return a :class:`Reproduction` for the first diverging stage.

        :raises ProgramDomainError: When no stage diverged.
        """
        divergence = self.first_divergence

        if divergence is None or divergence.interpretation is None:
            raise ProgramDomainError(
                "A reproduction requires a stage that diverged from the baseline."
            )

        return build_reproduction(
            divergence.interpretation,
            label=f"{self.label}: {divergence.name}",
        )


def localize_divergence(baseline, divergence):
    """Pin the difference between two stages on an instance and an operation.

    The first differing output element is mapped back to the program instance
    that writes it using the same access map the interpreter executes with, and
    the ``mem.store`` operations that produce that output are reported with their
    SSA locations.

    :param baseline: The last :class:`PassStage` that matched the frontend.
    :param divergence: The first :class:`PassStage` that diverged.
    :return: A :class:`Localization`, or ``None`` when it cannot be narrowed down.
    """
    if baseline.interpretation is None or divergence.interpretation is None:
        return None

    if divergence.diff is None:
        return None

    for name, output in divergence.diff.outputs.items():
        if output.matches or not output.first_mismatches:
            continue

        index = output.first_mismatches[0]
        expected = _element(baseline.interpretation, name, index)
        actual = _element(divergence.interpretation, name, index)

        return Localization(
            pass_name=divergence.name,
            output=name,
            index=index,
            expected=expected,
            actual=actual,
            program_id=_instance_writing(divergence.interpretation, name, index),
            operations=_stores_of(divergence.interpretation.program, name),
        )

    return None


def compare_passes(
    arrangement,
    application,
    tensors=(),
    inputs=None,
    *,
    pipeline=None,
    backend=None,
    platform=None,
    compute_arch=None,
    pass_options=None,
    pass_registry=None,
    symbols=None,
    rtol=DEFAULT_RTOL,
    atol=DEFAULT_ATOL,
    label=None,
    **options,
):
    """Interpret a program after each cumulative prefix of a pass pipeline.

    Every stage must agree with the program the frontend produced.  Running the
    interpreter at each prefix shows which pass, and which SSA operation, first
    changed the result.

    Prefixes are applied cumulatively, so the reported pass is the first one
    whose own rewrite changed behaviour.  A stage the interpreter cannot execute
    (an unsupported opcode, for instance) is recorded with its error instead of
    aborting the whole scan.

    :param arrangement: The arrangement function.
    :param application: The application function.
    :param tensors: Symbolic tensors or runtime arrays.
    :param inputs: Runtime arrays.
    :param pipeline: The pipeline to bisect.  A sequence of pass names, or ``None``
        for the default pipeline of ``backend``.
    :param backend: The backend whose default pipeline and passes are used.
    :param platform: The platform used when resolving the target profile.
    :param compute_arch: The compute architecture used for the target profile.
    :param pass_options: Per-pass options for the pipeline.
    :param pass_registry: A pass registry that extends the built-in one.
    :param symbols: Values for symbolic dimensions and constexpr parameters.
    :param rtol: Relative tolerance for floating point outputs.
    :param atol: Absolute tolerance for floating point outputs.
    :param label: A human-readable description of the comparison.
    :param options: Extra keyword arguments forwarded to
        :func:`~ninetoothed.interpret.interpret`.
    :return: A :class:`PipelineDiff`.
    """
    from ninetoothed.compiler.passes import PipelineSpec, default_spec

    if pipeline is None:
        pass_names = tuple(default_spec(backend).passes)
    else:
        pass_names = tuple(str(name) for name in pipeline)

    stages = []
    previous = None

    for index in range(len(pass_names) + 1):
        prefix = pass_names[:index]
        name = "" if index == 0 else pass_names[index - 1]
        stage_pipeline = (
            None
            if not prefix
            else PipelineSpec(
                passes=prefix,
                mode="custom",
                reason=f"prefix of {index} pass(es)",
            )
        )
        stage = PassStage(index=index, name=name, pipeline=stage_pipeline)

        try:
            stage.interpretation = interpret(
                arrangement,
                application,
                tensors=tensors,
                inputs=inputs,
                symbols=symbols,
                backend=backend,
                platform=platform,
                compute_arch=compute_arch,
                pipeline=stage_pipeline,
                pass_options=pass_options,
                pass_registry=pass_registry,
                **options,
            )
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            stage.error = f"{type(error).__name__}: {error}"
            stages.append(stage)

            continue

        if stages and stages[0].interpretation is not None:
            stage.diff = compare_interpretations(
                stages[0].interpretation,
                stage.interpretation,
                rtol=rtol,
                atol=atol,
                compare_trace=False,
                label=f"{name or '<frontend>'} vs <frontend>",
            )

        if previous is not None:
            stage.diff_previous = compare_interpretations(
                previous,
                stage.interpretation,
                rtol=rtol,
                atol=atol,
                compare_trace=False,
                label=f"{name} vs previous stage",
            )

        previous = stage.interpretation
        stages.append(stage)

    return PipelineDiff(
        stages=tuple(stages),
        pass_names=pass_names,
        label=label or f"{getattr(application, '__name__', 'application')} pipeline",
        rtol=rtol,
        atol=atol,
    )


def _element(interpretation, name, index):
    """Return one element of an output, or ``None`` when it is not available."""
    array = interpretation.outputs.get(name)

    if array is None:
        return None

    try:
        return np.asarray(array)[tuple(index)]
    except (IndexError, ValueError):  # pragma: no cover - defensive
        return None


def _instance_writing(interpretation, name, index):
    """Return the program instance that writes one output element."""
    from .interpreter import domain_size
    from .memory import tile_access_map

    tensor = interpretation.memory.tensors.get(name)

    if tensor is None:
        return None

    if not tensor.is_tiled:
        return 0

    shape = tuple(int(dim) for dim in tensor.source_shape)

    try:
        flat = int(np.ravel_multi_index(tuple(index), shape))
    except ValueError:  # pragma: no cover - defensive
        return None

    total = domain_size(interpretation.launch_shape)

    for program_id in range(total):
        context = interpretation.memory.context(program_id)
        offsets, mask = tile_access_map(tensor, context)
        active = np.asarray(mask).astype(bool) & (np.asarray(offsets) == flat)

        if active.any():
            return program_id

    return None


def _stores_of(program, name):
    """Return ``((opcode, location), ...)`` for the stores that write ``name``.

    The locations use the same ``path:index:opcode`` scheme the interpreter
    records in a trace, so they can be matched against a trace event directly.
    """
    found = []

    def visit(block, path):
        for index, operation in enumerate(block.operations):
            location = f"{path}:{index}:{operation.opcode}"

            if (
                operation.opcode.startswith("mem.store")
                and len(operation.operands) == 2
            ):
                target = operation.operands[1]

                if target == name or target.startswith(f"{name}."):
                    found.append((operation.opcode, location))

            for region in operation.regions:
                visit(region, f"{location}/{region.name or 'region'}")

    for block in program.blocks:
        visit(block, block.name or "entry")

    return tuple(found)


def _jsonable(value):
    """Return a JSON-friendly scalar for a NumPy value."""
    if isinstance(value, np.generic):
        return value.item()

    return value


@dataclass
class BufferSpec:
    """One buffer of a reproduction: its name, shape, dtype and data.

    :param name: The tensor name.
    :param shape: The shape of the tensor as the arrangement sees it.
    :param dtype: The NumPy dtype name.
    :param data: The buffer contents.
    :param parameter: Whether the buffer is an input parameter of the application.
    """

    name: str
    shape: tuple
    dtype: str
    data: Any
    parameter: bool = True

    def to_dict(self):
        """Return a JSON-friendly representation."""
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "parameter": self.parameter,
            "data": np.asarray(self.data).tolist(),
        }


@dataclass
class Reproduction:
    """Everything needed to rebuild one failing interpretation.

    :param ssa: The executed SSA program, rendered as text.
    :param buffers: The :class:`BufferSpec` of every tensor in CPU memory.
    :param symbols: The resolved symbol values.
    :param launch_shape: The shape of the program-instance grid.
    :param seed: The random seed recorded for the inputs, when one was.
    :param program_id: The program instance to highlight.
    :param label: A description of what is being reproduced.
    :param passes: The pass names the program was lowered with.
    :param arrangement_source: The arrangement source, when it is recoverable.
    :param application_source: The application source, when it is recoverable.
    """

    ssa: str = ""
    buffers: tuple = ()
    symbols: dict = field(default_factory=dict)
    launch_shape: tuple = ()
    seed: int | None = None
    program_id: int = 0
    label: str = "interpretation"
    passes: tuple = ()
    arrangement_source: str | None = None
    application_source: str | None = None

    def to_dict(self):
        """Return the reproduction as JSON-friendly data."""
        return {
            "label": self.label,
            "ssa": self.ssa,
            "launch_shape": list(self.launch_shape),
            "program_id": self.program_id,
            "seed": self.seed,
            "passes": list(self.passes),
            "symbols": {name: _jsonable(value) for name, value in self.symbols.items()},
            "inputs": [buffer.to_dict() for buffer in self.buffers if buffer.parameter],
            "outputs": [
                buffer.to_dict() for buffer in self.buffers if not buffer.parameter
            ],
        }

    def to_json(self, *, indent=2):
        """Return the reproduction serialized as JSON."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def render(self):
        """Render the reproduction as a runnable Python snippet."""
        inputs = tuple(buffer for buffer in self.buffers if buffer.parameter)
        outputs = tuple(buffer for buffer in self.buffers if not buffer.parameter)
        lines = [
            "# Minimal reproduction generated by ninetoothed.interpret.diff",
            f"# {self.label}",
            "#",
            f"# launch shape : {tuple(self.launch_shape)}",
            f"# program id   : {self.program_id}",
            f"# passes       : {', '.join(self.passes) or '<frontend program>'}",
            f"# random seed  : {self.seed!r}",
            "#",
            "# Lowered SSA:",
        ]
        lines.extend(f"#   {line}" for line in self.ssa.rstrip().splitlines())
        lines.extend(
            [
                "",
                "import numpy as np",
                "",
                "from ninetoothed import Tensor, block_size  # noqa: F401",
                "from ninetoothed.interpret import Tracer, interpret, interpret_program",
                "",
                "# Recorded when the failing run was produced.  Regenerate the inputs",
                "# with `random_inputs(BUFFER_SPECS, seed=SEED)` to rebuild them from",
                "# scratch; the literal arrays below are authoritative.",
                f"SEED = {self.seed!r}",
                "",
                "# {name: (shape, dtype)} for every input.",
                "BUFFER_SPECS = {",
            ]
        )

        for buffer in inputs:
            lines.append(f"    {buffer.name!r}: ({buffer.shape!r}, {buffer.dtype!r}),")

        lines.extend(["}", "", "BUFFERS = {"])

        for buffer in inputs:
            array = np.asarray(buffer.data)
            lines.extend(
                [
                    f"    # shape {buffer.shape}, dtype {buffer.dtype}",
                    f"    {buffer.name!r}: np.array(",
                    f"        {array.tolist()!r},",
                    f"        dtype=np.{array.dtype},",
                    "    ),",
                ]
            )

        lines.extend(["}", "", "OUTPUTS = {"])

        for buffer in outputs:
            array = np.asarray(buffer.data)
            lines.extend(
                [
                    f"    # shape {buffer.shape}, dtype {buffer.dtype}",
                    f"    {buffer.name!r}: np.array(",
                    f"        {array.tolist()!r},",
                    f"        dtype=np.{array.dtype},",
                    "    ),",
                ]
            )

        lines.extend(["}", "", "SYMBOLS = {"])

        for name, value in sorted(self.symbols.items()):
            lines.append(f"    {name!r}: {_jsonable(value)!r},")

        lines.extend(["}", ""])

        if self.application_source is not None:
            if self.arrangement_source is not None:
                lines.extend(
                    [
                        "# --- arrangement (as originally written) ---",
                        self.arrangement_source,
                        "",
                    ]
                )

            lines.extend(
                [
                    "# --- application (as originally written) ---",
                    self.application_source,
                    "",
                    "# `interpret` re-lowers the program from the two functions above.",
                    "# Pass the original Tensor descriptors as `tensors` when the",
                    "# arrangement needs their rank.",
                    "if __name__ == '__main__':",
                    "    result = interpret(",
                    "        arrangement,",
                    "        application,",
                    "        tensors=tuple(",
                    "            Tensor(np.asarray(item).ndim) for item in BUFFERS.values()",
                    "        ),",
                    "        inputs=tuple(BUFFERS.values()),",
                    "        symbols=SYMBOLS,",
                    "        seed=SEED,",
                    "        tracer=Tracer(),",
                    "    )",
                    "",
                    "    print(result.render_trace(limit=40))",
                ]
            )
        else:
            lines.extend(
                [
                    "# The original functions are not importable from here, so the",
                    "# snippet rebuilds the buffers only. Re-lower the program with the",
                    "# original arrangement/application and pass it to `interpret_program`.",
                    "if __name__ == '__main__':",
                    "    print('SEED =', SEED)",
                    "    print('SYMBOLS =', SYMBOLS)",
                    "    for name, (shape, dtype) in BUFFER_SPECS.items():",
                    "        print(name, shape, dtype)",
                ]
            )

        return "\n".join(lines)


def build_reproduction(interpretation, *, label="interpretation", program_id=0):
    """Collect everything needed to rebuild one interpretation.

    :param interpretation: The :class:`~ninetoothed.interpret.Interpretation` to
        reproduce.
    :param label: A description of what is being reproduced.
    :param program_id: The program instance to highlight.
    :return: A :class:`Reproduction`.
    """
    from ninetoothed.ir import ssa

    stored = set()
    program = interpretation.program

    if program is not None:
        stored.update(_stored_names(program))

    buffers = []

    for name, tensor in interpretation.memory.tensors.items():
        shape = tuple(int(dim) for dim in tensor.source_shape)
        array = np.asarray(tensor.buffer).reshape(shape)
        buffers.append(
            BufferSpec(
                name=name,
                shape=shape,
                dtype=str(array.dtype),
                data=array,
                parameter=name not in stored,
            )
        )

    return Reproduction(
        ssa=ssa.render(program) if program is not None else "",
        buffers=tuple(buffers),
        symbols=dict(interpretation.symbols),
        launch_shape=tuple(interpretation.launch_shape),
        seed=interpretation.seed,
        program_id=program_id,
        label=label,
        passes=tuple(_pass_names_of(interpretation.pipeline)),
        arrangement_source=_function_source(interpretation.arrangement),
        application_source=_function_source(interpretation.application),
    )


def render_reproduction(interpretation, *, label="interpretation", program_id=0):
    """Render a self-contained reproduction snippet for one interpretation.

    The snippet embeds the executed SSA (as a comment), the input arrays with
    their shapes and dtypes, the resolved symbol values, the recorded random seed
    and, when the original functions are still importable, their source.

    :param interpretation: The :class:`~ninetoothed.interpret.Interpretation` to
        reproduce.
    :param label: A comment describing the reproduction.
    :param program_id: The program instance to highlight.
    :return: A Python source snippet as a string.
    """
    return build_reproduction(
        interpretation, label=label, program_id=program_id
    ).render()


def _stored_names(program):
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

    return names


def _pass_names_of(pipeline):
    """Return the pass names of a pipeline spec, when it carries any."""
    if pipeline is None:
        return ()

    passes = getattr(pipeline, "passes", None)

    if passes is None:
        return ()

    return tuple(str(name) for name in passes)


def _function_source(function):
    """Return the source of ``function`` when it can be recovered."""
    import inspect

    if function is None:
        return None

    try:
        return inspect.getsource(function).rstrip()
    except (OSError, TypeError):  # pragma: no cover - interactive or builtin
        return None


__all__ = [
    "DEFAULT_ATOL",
    "DEFAULT_RTOL",
    "BufferSpec",
    "Localization",
    "OutputDiff",
    "PassStage",
    "PipelineDiff",
    "ProgramDiff",
    "Reproduction",
    "TraceDiff",
    "TraceDivergence",
    "build_reproduction",
    "compare_interpretations",
    "compare_passes",
    "compare_pipeline",
    "compare_traces",
    "localize_divergence",
    "render_reproduction",
]
