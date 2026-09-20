"""Lowering of a NineToothed application to an interpretable SSA program.

The interpreter reuses the existing frontend and SSA pipeline: ``arrangement``
produces the layouts and the index mappings, the Python frontend produces an
``ssa.Program``, and the SSA pass pipeline produces exactly the program a
backend would receive. The interpreter then executes that program on the CPU.
"""

import copy
import inspect
from dataclasses import dataclass, field

from ninetoothed.compiler.driver import _meta_defaults
from ninetoothed.compiler.passes import lower_for_target
from ninetoothed.frontend.layout import tensor_specs
from ninetoothed.frontend.python import from_application
from ninetoothed.ir import TensorSpec, ssa
from ninetoothed.naming import is_constexpr, is_meta, remove_prefixes


@dataclass(frozen=True, kw_only=True)
class Interpretation:
    """A NineToothed application lowered for the CPU interpreter.

    :param application: The application function that was lowered.
    :param parameters: The application parameter names, in order.
    :param tensors: The declared symbolic tensors.
    :param arranged: The arranged tensor views produced by the arrangement.
    :param specs: The tensor specs of the arranged views.
    :param program: The lowered SSA program the interpreter executes.
    :param raw_program: The SSA program before the pass pipeline.
    :param pass_trace: The names of the passes that ran.
    :param meta_defaults: The default values of the meta symbols.
    :param symbol_names: The user-facing name of every bounded symbol.
    :param tensor_parameters: The parameters that are tensors.
    :param scalar_parameters: The parameters that are scalars.
    """

    application: object
    parameters: tuple
    tensors: tuple
    arranged: tuple
    specs: tuple
    program: ssa.Program
    raw_program: ssa.Program
    pass_trace: tuple = ()
    meta_defaults: dict = field(default_factory=dict)
    symbol_names: dict = field(default_factory=dict)
    tensor_parameters: tuple = ()
    scalar_parameters: tuple = ()

    @property
    def required_symbols(self) -> tuple[str, ...]:
        """Return the constexpr symbols the caller has to provide."""
        return tuple(
            name for name in self.symbol_names if name not in self.meta_symbol_names
        )

    @property
    def meta_symbol_names(self) -> tuple[str, ...]:
        """Return the user-facing names of the meta symbols."""
        return tuple(
            sorted(
                name for name in self.symbol_names if is_meta(self.symbol_names[name])
            )
        )


def _require_readable_source(application, name):
    """Fail early when the frontend cannot read the application's source.

    The Python frontend lowers source text, not bytecode, so a function defined in
    a REPL, through ``exec``, or through ``python -c`` on an interpreter whose
    ``inspect`` cannot recover that source cannot be interpreted. The frontend
    reports that only by returning ``None``, so the reason is spelled out here.
    """
    try:
        inspect.getsource(application)
    except (OSError, TypeError) as error:
        raise ValueError(
            f"Cannot interpret `{name}`: the frontend lowers Python source, but the "
            "source of the application is not retrievable. Define the application in "
            "a file or an importable module rather than in a REPL, in `exec`, or in "
            f"`python -c`. The underlying error is: {error}."
        ) from None


def lower(
    arrangement,
    application,
    tensors=(),
    *,
    backend=None,
    kernel_name=None,
    pipeline=None,
    pass_options=None,
    run_passes=True,
) -> Interpretation:
    """Lower an application to the SSA program the interpreter executes.

    :param arrangement: The arrangement function, or ``None`` to use the tensor
        annotations of ``application``.
    :param application: The application function.
    :param tensors: The declared symbolic tensors, in parameter order.
    :param backend: The backend whose SSA shape the interpreter mirrors.
    :param kernel_name: The SSA program name.
    :param pipeline: An optional SSA pass pipeline specification.
    :param pass_options: Optional per-pass options.
    :param run_passes: Whether to run the SSA pass pipeline.
    :return: The lowered interpretation.
    """
    parameters = tuple(inspect.signature(application).parameters)
    symbolic_tensors = tuple(copy.deepcopy(tensor) for tensor in tensors)
    arranged = _arrange(arrangement, application, parameters, symbolic_tensors)

    if len(arranged) != len(parameters):
        raise ValueError(
            f"Cannot interpret `{getattr(application, '__name__', application)}`: the "
            f"arrangement returned {len(arranged)} values for {len(parameters)} "
            "parameters."
        )

    specs = tensor_specs(parameters, arranged)
    name = kernel_name or getattr(application, "__name__", "kernel")

    _require_readable_source(application, name)

    program = from_application(application, specs, kind=name, strict=True)

    if program is None:
        raise ValueError(
            f"Cannot interpret `{name}`: the frontend did not produce an `ssa.Program`."
        )

    lowered = program

    if run_passes:
        lowered = lower_for_target(
            program,
            backend=backend,
            tensors=specs,
            pass_pipeline=pipeline,
            pass_options=pass_options,
        )

    return Interpretation(
        application=application,
        parameters=parameters,
        tensors=symbolic_tensors,
        arranged=arranged,
        specs=specs,
        program=lowered,
        raw_program=program,
        pass_trace=tuple(lowered.metadata.get("pass_trace", ())),
        meta_defaults=_meta_defaults(arranged),
        symbol_names=_symbol_names(arranged),
        tensor_parameters=_tensor_parameters(specs),
        scalar_parameters=_scalar_parameters(specs),
    )


def _arrange(arrangement, application, parameters, symbolic_tensors):
    if arrangement is not None:
        arranged = arrangement(*symbolic_tensors)

        return arranged if isinstance(arranged, tuple) else (arranged,)

    annotations = inspect.get_annotations(application, eval_str=False)

    try:
        return tuple(copy.deepcopy(annotations[name]) for name in parameters)
    except KeyError as exc:
        raise ValueError(
            f"Cannot interpret `{application.__name__}`: parameter "
            f"`{exc.args[0]}` has no tensor annotation."
        ) from exc


def _symbol_names(arranged) -> dict:
    names = {}

    for tensor in arranged:
        for symbol in tensor.names():
            text = str(symbol)

            if not hasattr(symbol, "lower_bound") or not is_constexpr(text):
                continue

            names.setdefault(remove_prefixes(text), text)

    return names


def _tensor_parameters(specs) -> tuple:
    return tuple(
        spec.name for spec in specs if isinstance(spec, TensorSpec) and spec.ndim > 0
    )


def _scalar_parameters(specs) -> tuple:
    return tuple(
        spec.name for spec in specs if isinstance(spec, TensorSpec) and spec.ndim == 0
    )


__all__ = ["Interpretation", "lower"]
