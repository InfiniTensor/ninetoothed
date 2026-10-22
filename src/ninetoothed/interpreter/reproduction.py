"""Export a minimal reproduction of a mismatching differential case.

A differential run compares the CPU interpreter against a reference, such as
NumPy, PyTorch, or the production backend. When the two disagree, the useful
artefact is a directory holding everything needed to reproduce the disagreement:
the SSA of the case, the inputs, the shape and dtype of every argument, the seed,
and a script that re-runs the case.

``export_reproduction`` writes that directory. The script it generates embeds the
source of the arrangement and of the application, because the frontend lowers
source: a reproduction that only stored the SSA could not be re-run.

.. code-block:: python

    from ninetoothed.interpreter import export_reproduction

    if not torch.allclose(expected, actual, rtol=rtol, atol=atol):
        export_reproduction(
            "repro",
            name="masked",
            arrangement=arrangement,
            application=application,
            tensors=tensors,
            arguments=arguments,
            meta=meta,
            expected=expected,
            actual=actual,
            tolerance=(rtol, atol),
            seed=20260916,
            reason="the backend result differs from the interpreter result",
        )
"""

import inspect
import json
import math
import pathlib
import platform
import textwrap
from dataclasses import dataclass

import numpy as np

from ninetoothed.interpreter.lowering import lower
from ninetoothed.naming import is_constexpr, is_meta, remove_prefixes

_UNEXPORTED = "# NOTE: `{name}` is used by the source below but could not be exported."


@dataclass(frozen=True, kw_only=True)
class Reproduction:
    """The directory written for one mismatching case.

    :param directory: The directory that holds the reproduction.
    :param files: The files that were written, in name order.
    :param reason: Why the reproduction was exported.
    :param unresolved: The global names the script could not export.
    """

    directory: pathlib.Path
    files: tuple = ()
    reason: str = ""
    unresolved: tuple = ()

    def format(self) -> str:
        """Render the reproduction as a readable multi-line string."""
        lines = [f"reproduction of `{self.directory.name}` in {self.directory}"]

        if self.reason:
            lines.append(f"  reason: {self.reason}")

        lines.extend(f"  {path.name}" for path in self.files)

        if self.unresolved:
            lines.append(f"  unresolved names: {', '.join(self.unresolved)}")

        return "\n".join(lines)


def export_reproduction(
    directory,
    *,
    name,
    arrangement,
    application,
    tensors=(),
    arguments=(),
    meta=None,
    expected=None,
    actual=None,
    tolerance=(1e-3, 1e-3),
    seed=None,
    reason="",
    kernel_name=None,
) -> Reproduction:
    """Write everything needed to reproduce one differential case.

    :param directory: The directory to write, created when it does not exist.
    :param name: The name of the case.
    :param arrangement: The arrangement function of the case.
    :param application: The application function of the case.
    :param tensors: The declared symbolic tensors, in parameter order.
    :param arguments: The arguments the case ran with.
    :param meta: The meta and constexpr values of the case.
    :param expected: The result the interpreter produced, if any.
    :param actual: The result the reference produced, if any.
    :param tolerance: The relative and absolute tolerance of the comparison.
    :param seed: The seed the inputs were generated with.
    :param reason: Why the reproduction is being exported.
    :param kernel_name: The SSA program name, defaulting to ``name``.
    :return: The written reproduction.
    """
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    arguments = list(arguments)
    meta = dict(meta or {})
    rtol, atol = tolerance
    kernel_name = kernel_name or name

    interpretation = lower(arrangement, application, tensors, kernel_name=kernel_name)

    values = [_to_numpy(argument) for argument in arguments]
    expected_array = None if expected is None else _to_numpy(expected)
    actual_array = None if actual is None else _to_numpy(actual)

    written = []

    np.savez(
        directory / "inputs.npz",
        **{f"argument_{index}": value for index, value in enumerate(values)},
    )
    written.append(directory / "inputs.npz")

    if expected_array is not None:
        np.save(directory / "expected.npy", expected_array)
        written.append(directory / "expected.npy")

    if actual_array is not None:
        np.save(directory / "actual.npy", actual_array)
        written.append(directory / "actual.npy")

    written.append(
        _write(directory / "ssa_frontend.txt", str(interpretation.raw_program))
    )
    written.append(_write(directory / "ssa_optimized.txt", str(interpretation.program)))
    written.append(
        _write(directory / "passes.txt", "\n".join(interpretation.pass_trace))
    )

    script, unresolved = _render_script(
        name=name,
        kernel_name=kernel_name,
        arrangement=arrangement,
        application=application,
        tensors=tensors,
        meta=meta,
        seed=seed,
        tolerance=(rtol, atol),
    )

    written.append(_write(directory / "reproduce.py", script))

    case = {
        "case": name,
        "kernel_name": kernel_name,
        "reason": reason,
        "seed": seed,
        "meta": {key: str(value) for key, value in meta.items()},
        "tolerance": {"rtol": rtol, "atol": atol},
        "tensors": [_describe_tensor(tensor) for tensor in tensors],
        "arguments": [
            dict(_describe_array(value), name=f"argument_{index}")
            for index, value in enumerate(values)
        ],
        "expected": None if expected_array is None else _describe_array(expected_array),
        "actual": None if actual_array is None else _describe_array(actual_array),
        "passes": [str(pass_name) for pass_name in interpretation.pass_trace],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "unresolved_names": unresolved,
    }

    case.update(_compare(expected_array, actual_array, rtol, atol))

    written.append(
        _write(
            directory / "case.json",
            json.dumps(case, indent=2, ensure_ascii=False) + "\n",
        )
    )

    return Reproduction(
        directory=directory,
        files=tuple(sorted(written)),
        reason=reason,
        unresolved=tuple(unresolved),
    )


def _compare(expected, actual, rtol, atol) -> dict:
    """Return the recorded comparison of two results."""
    if expected is None or actual is None:
        return {"max_absolute_difference": None, "mismatched_elements": None}

    left = np.asarray(expected, dtype=np.float64)
    right = np.asarray(actual, dtype=np.float64)

    if left.shape != right.shape:
        return {
            "max_absolute_difference": None,
            "mismatched_elements": None,
            "shape_mismatch": [list(left.shape), list(right.shape)],
        }

    close = np.isclose(left, right, rtol=rtol, atol=atol)

    return {
        "max_absolute_difference": float(np.abs(left - right).max())
        if left.size
        else 0.0,
        "mismatched_elements": int((~close).sum()),
    }


def _to_numpy(value):
    """Return a NumPy view of a NumPy array or a CPU tensor."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def _describe_array(value) -> dict:
    """Return the shape, dtype and element count of one argument."""
    array = np.asarray(value)

    return {
        "shape": [int(dimension) for dimension in array.shape],
        "dtype": str(array.dtype),
        "elements": int(array.size),
    }


def _describe_tensor(tensor) -> dict:
    """Return the declared properties of one symbolic tensor."""
    return {
        "ndim": int(tensor.ndim),
        "dtype": None if tensor.dtype is None else str(tensor.dtype),
        "other": _literal(tensor.other),
    }


def _write(path: pathlib.Path, text: str) -> pathlib.Path:
    """Write one text file and return its path."""
    if not text.endswith("\n"):
        text = f"{text}\n"

    path.write_text(text, encoding="utf-8")

    return path


def _literal(value):
    """Return a Python literal for a value, or ``None`` when it has none."""
    if value is None:
        return "None"

    if isinstance(value, (bool, int, str)):
        return repr(value)

    if isinstance(value, float):
        if math.isnan(value):
            return 'float("nan")'

        if math.isinf(value):
            return 'float("inf")' if value > 0 else 'float("-inf")'

        return repr(value)

    return None


def _symbol_literal(value):
    """Return a ``Symbol(...)`` expression that rebuilds a symbol, or ``None``.

    The interpreter receives meta and constexpr values per name, so a script that
    lost the symbols of its source would not reproduce anything. The mangled name
    of a symbol records the options it was built with, which is enough to rebuild
    it exactly.

    :param value: The value to rebuild.
    :return: The expression, or ``None`` when the value is not a symbol.
    """
    if type(value).__name__ != "Symbol" or not hasattr(value, "lower_bound"):
        return None

    text = str(value)
    options = []

    if is_meta(text):
        options.append("meta=True")
    elif is_constexpr(text):
        options.append("constexpr=True")

    listed = "".join(f", {option}" for option in options)

    return f"Symbol({remove_prefixes(text)!r}{listed})"


def _used_names(*functions) -> dict:
    """Return the global names the functions read, keyed by name."""
    found = {}

    for function in functions:
        try:
            variables = inspect.getclosurevars(function)
        except TypeError:
            continue

        for name, value in variables.globals.items():
            found.setdefault(name, value)

    return found


def _tensor_literals(tensors) -> list:
    """Return one ``Tensor(...)`` expression per symbolic tensor."""
    expressions = []

    for tensor in tensors:
        parts = [repr(int(tensor.ndim))]

        if tensor.dtype is not None:
            parts.append(f"dtype={_literal(tensor.dtype)}")

        if tensor.other is not None:
            parts.append(f"other={_literal(tensor.other)}")

        expressions.append(f"Tensor({', '.join(parts)})")

    return expressions


def _render_script(
    *, name, kernel_name, arrangement, application, tensors, meta, seed, tolerance
):
    """Return the reproduction script and the names it could not export."""
    rtol, atol = tolerance
    used = _used_names(arrangement, application)

    lines = [
        '"""Minimal reproduction of the `%s` differential case.' % name,
        "",
        "This script and the files next to it were written by",
        "`ninetoothed.interpreter.export_reproduction`. It runs the CPU interpreter on",
        "the recorded inputs, so it needs no device:",
        "",
        "    python reproduce.py",
        "",
        "Pass `--backend` to also run the production backend, which needs a CUDA",
        "device, and to compare it against the recorded backend result.",
        '"""',
        "",
        "import argparse",
        "import pathlib",
        "",
        "import numpy as np",
        "",
        "import ninetoothed",
        "from ninetoothed import Symbol, Tensor",
    ]

    imports = []

    for global_name in sorted(used):
        value = used[global_name]

        if inspect.ismodule(value):
            module = getattr(value, "__name__", global_name)
            imports.append(
                f"import {module}"
                if global_name == module
                else f"import {module} as {global_name}"
            )

    if imports:
        lines.extend(["", *imports])

    constants, unresolved = _render_constants(used)

    lines.extend(
        [
            "",
            "# The seed the recorded inputs were generated with.",
            f"SEED = {_literal(seed)}",
            "",
            f"META = {json.dumps(meta, ensure_ascii=False)}",
            f"RTOL = {_literal(rtol)}",
            f"ATOL = {_literal(atol)}",
        ]
    )

    if constants:
        lines.extend(["", *constants])

    lines.extend(
        [
            "",
            "",
            "def tensors():",
            '    """Return the symbolic tensors of the case."""',
        ]
    )

    expressions = _tensor_literals(tensors)

    if expressions:
        lines.append("    return (")
        lines.extend(f"        {expression}," for expression in expressions)
        lines.append("    )")
    else:
        lines.append("    return ()")

    lines.extend(_render_function(arrangement, "arrangement"))
    lines.extend(_render_function(application, "application"))
    lines.extend(["", "", _render_main(name, kernel_name, rtol, atol)])

    return "\n".join(lines) + "\n", unresolved


def _render_constants(used) -> tuple:
    """Return the exported global values and the names that could not be exported."""
    lines = []
    unresolved = []

    for name in sorted(used):
        value = used[name]

        if inspect.ismodule(value):
            continue

        literal = _symbol_literal(value)

        if literal is None:
            literal = _literal(value)

        if literal is None:
            unresolved.append(name)
            lines.append(_UNEXPORTED.format(name=name))

            continue

        lines.append(f"{name} = {literal}")

    return lines, unresolved


def _render_function(function, alias: str) -> list:
    """Return the source of one function, named as the script expects it.

    The recorded functions do not have to be called ``arrangement`` and
    ``application``, so the source is followed by an alias when the names differ.

    :param function: The function to embed.
    :param alias: The name the generated script uses.
    :return: The lines of the definition, and of the alias when one is needed.
    """
    lines = ["", "", textwrap.dedent(inspect.getsource(function)).strip()]
    name = getattr(function, "__name__", alias)

    if name != alias:
        lines.extend(["", "", f"{alias} = {name}"])

    return lines


def _render_main(name, kernel_name, rtol, atol) -> str:
    """Return the ``main`` function of the reproduction script."""
    lines = [
        "def main():",
        '    """Re-run the case and report whether it still reproduces."""',
        "    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])",
        "    parser.add_argument(",
        '        "--backend",',
        '        action="store_true",',
        '        help="Also run the production backend (needs a CUDA device).",',
        "    )",
        "    arguments = parser.parse_args()",
        "",
        "    directory = pathlib.Path(__file__).resolve().parent",
        "",
        '    with np.load(directory / "inputs.npz") as recorded:',
        "        values = [",
        '            recorded[f"argument_{index}"].copy()',
        "            for index in range(len(recorded.files))",
        "        ]",
        "",
        "    kernel = ninetoothed.interpret(",
        "        arrangement,",
        "        application,",
        "        tensors(),",
        f"        kernel_name={kernel_name!r},",
        "        meta=META,",
        "    )",
        "",
        "    outputs = [value.copy() for value in values]",
        "",
        "    kernel(*outputs)",
        "",
        "    result = outputs[-1]",
        "",
        '    print(f"seed: {SEED}")',
        '    print(f"tolerance: rtol={RTOL}, atol={ATOL}")',
        "",
        '    recorded = directory / "expected.npy"',
        "",
        "    if recorded.exists():",
        "        expected = np.load(recorded)",
        "",
        '        print(f"interpreter matches the recorded result: '
        '{np.allclose(result, expected, rtol=RTOL, atol=ATOL)}")',
        '        print(f"max |difference|: {np.abs(result - expected).max()}")',
        "    else:",
        '        print(f"interpreter output: {result}")',
        "",
        "    if not arguments.backend:",
        '        print("pass --backend to also run the production backend.")',
        "",
        "        return",
        "",
        "    _run_backend(directory, values, result)",
    ]

    lines.extend(
        [
            "",
            "",
            "def _run_backend(directory, values, result):",
            '    """Run the production backend, which needs a CUDA device."""',
            "    import torch",
            "",
            '    if not torch.cuda.is_available() or not hasattr(ninetoothed, "make"):',
            '        print("no CUDA device is available for the production backend.")',
            "",
            "        return",
            "",
            "    backend_arguments = [torch.as_tensor(value).cuda() for value in values]",
            "    backend_kernel = ninetoothed.make(",
            "        arrangement,",
            "        application,",
            "        tensors(),",
            f"        kernel_name={kernel_name!r},",
            "        max_num_configs=1,",
            "    )",
            "    backend_kernel(*backend_arguments, **META)",
            "",
            "    actual = backend_arguments[-1].cpu().numpy()",
            "",
            '    print(f"backend matches the interpreter: '
            '{np.allclose(actual, result, rtol=RTOL, atol=ATOL)}")',
            "",
            '    recorded = directory / "actual.npy"',
            "",
            "    if recorded.exists():",
            "        recorded_actual = np.load(recorded)",
            "",
            '        print(f"backend matches the recorded result: '
            '{np.allclose(actual, recorded_actual, rtol=RTOL, atol=ATOL)}")',
            "",
            "",
            'if __name__ == "__main__":',
            "    main()",
        ]
    )

    return "\n".join(lines)
