"""Pass-pipeline differential diagnosis for the CPU reference interpreter.

A backend pipeline is a chain of rewrites, and a rewrite that changes the meaning
of a program is hard to spot by reading the SSA alone. ``compare_passes`` runs
the low-level SSA program and then every prefix of the pipeline through the
interpreter, compares the outputs, and reports the first prefix whose result
differs from the front end's.
"""

from dataclasses import dataclass, field

import numpy as np

from ninetoothed.interpreter.lowering import Interpretation, lower


@dataclass(frozen=True, kw_only=True)
class PassStep:
    """The interpreter result after one prefix of the SSA pass pipeline.

    :param pass_name: The pass the prefix added, or ``None`` for the front end.
    :param passes: The passes that ran for this step, in order.
    :param matches: Whether the step reproduces the front-end result.
    :param max_difference: The largest absolute element difference, if any.
    :param interpretation: The lowered interpretation of this step.
    """

    pass_name: str | None
    passes: tuple[str, ...]
    matches: bool
    max_difference: float | None
    interpretation: Interpretation = field(repr=False, compare=False)

    def format(self) -> str:
        """Render the step as one readable line."""
        label = self.pass_name or "front end"
        status = "matches" if self.matches else "differs"

        if self.max_difference is None:
            return f"{label}: {status} (no output to compare)"

        return f"{label}: {status} (max |difference| = {self.max_difference:.3e})"


@dataclass(frozen=True, kw_only=True)
class PassComparison:
    """The outcome of comparing every pass-pipeline prefix.

    :param steps: One step per pipeline prefix, starting with the front end.
    """

    steps: tuple[PassStep, ...]

    @property
    def first_mismatch(self) -> PassStep | None:
        """Return the first step that diverges from the front-end result."""
        for step in self.steps:
            if not step.matches:
                return step

        return None

    def format(self) -> str:
        """Render every step as a readable multi-line string."""
        lines = [step.format() for step in self.steps]
        mismatch = self.first_mismatch

        if mismatch is None:
            lines.append("Every pass preserved the front-end semantics.")

            return "\n".join(lines)

        lines.append(
            f"The first pass that changed the semantics is `{mismatch.pass_name}`."
        )

        return "\n".join(lines)


def compare_passes(
    arrangement,
    application,
    tensors,
    arguments,
    *,
    kernel_name=None,
    meta=None,
    pipeline=None,
    rtol=1e-3,
    atol=1e-3,
):
    """Interpret a program after every pass-pipeline prefix and compare.

    :param arrangement: The arrangement function, or ``None`` for annotations.
    :param application: The application function.
    :param tensors: The declared symbolic tensors, in parameter order.
    :param arguments: A callable returning a fresh argument tuple per run, so
        every prefix starts from clean buffers.
    :param kernel_name: The SSA program name.
    :param meta: Explicit values for the meta and constexpr symbols.
    :param pipeline: An explicit pipeline whose prefixes are compared, or
        ``None`` for the backend's default pipeline.
    :param rtol: The relative tolerance used to compare two outputs.
    :param atol: The absolute tolerance used to compare two outputs.
    :return: The comparison, one step per pipeline prefix.
    """
    from ninetoothed.interpreter.kernel import Interpreter

    full = lower(
        arrangement,
        application,
        tensors,
        kernel_name=kernel_name,
        pipeline=pipeline,
        run_passes=True,
    )
    passes = tuple(full.pass_trace)
    steps = []
    reference = None

    for count in range(len(passes) + 1):
        interpretation = lower(
            arrangement,
            application,
            tensors,
            kernel_name=kernel_name,
            pipeline=list(passes[:count]) if count else None,
            run_passes=bool(count),
        )
        values = tuple(arguments())
        output = Interpreter(interpretation, meta=meta)(*values)

        if reference is None:
            reference = None if output is None else np.array(output, copy=True)
            steps.append(
                PassStep(
                    pass_name=None,
                    passes=(),
                    matches=True,
                    max_difference=0.0 if reference is not None else None,
                    interpretation=interpretation,
                )
            )

            continue

        matches, difference = _compare(reference, output, rtol=rtol, atol=atol)

        steps.append(
            PassStep(
                pass_name=passes[count - 1],
                passes=passes[:count],
                matches=matches,
                max_difference=difference,
                interpretation=interpretation,
            )
        )

    return PassComparison(steps=tuple(steps))


def _compare(reference, actual, *, rtol, atol):
    if reference is None or actual is None:
        return reference is actual, None

    actual = np.asarray(actual)

    if actual.shape != reference.shape:
        return False, float("inf")

    if reference.dtype == np.dtype(np.bool_) or not np.issubdtype(
        reference.dtype, np.floating
    ):
        equal = np.array_equal(reference, actual)

        return bool(equal), 0.0 if equal else float("inf")

    difference = float(
        np.abs(actual.astype(np.float64) - reference.astype(np.float64)).max()
    )
    matches = bool(np.allclose(actual, reference, rtol=rtol, atol=atol))

    return matches, difference


__all__ = ["PassComparison", "PassStep", "compare_passes"]
