"""Python adapter that freezes a neutral module exactly once."""

import copy
import inspect
import textwrap
from dataclasses import replace

from ninetoothed.errors import LoweringError
from ninetoothed.frontend.layout import tensor_specs
from ninetoothed.frontend.python import from_application
from ninetoothed.ir import IndexExpr
from ninetoothed.ir.module import FrontendModule, verify_module
from ninetoothed.naming import is_meta


def freeze_application(request):
    """Capture all Python-dependent semantics before entering the shared compiler."""
    application = request.application
    params = tuple(inspect.signature(application).parameters)

    if request.arrangement is None:
        annotations = inspect.get_annotations(application, eval_str=False)

        try:
            arranged = tuple(copy.deepcopy(annotations[name]) for name in params)
        except KeyError as exc:
            raise LoweringError(
                f"Cannot lower `{application.__name__}`: parameter `{exc.args[0]}` "
                "does not have a Tensor annotation."
            ) from exc
    else:
        value = request.arrangement(*copy.deepcopy(request.tensors))
        arranged = value if isinstance(value, tuple) else (value,)

    if len(arranged) != len(params):
        raise LoweringError(
            f"Cannot lower `{application.__name__}`: arrangement returned "
            f"{len(arranged)} values for {len(params)} parameters."
        )

    for name, tensor in zip(params, arranged):
        dtype = (request.tensor_dtypes or {}).get(name)

        if dtype is not None:
            getattr(tensor, "source", tensor).dtype = dtype

    named = None
    function = None

    if any(getattr(tensor, "_named_layout", None) is not None for tensor in arranged):
        from ninetoothed.frontend.named_layout import normalize, validate_scope
        from ninetoothed.frontend.python import prepare_application

        function, scope = prepare_application(application)

        validate_scope(function, scope)
        arranged, named = normalize(function, params, arranged)

    specs = tensor_specs(params, arranged)

    if named is not None:
        specs = tuple(
            replace(
                spec,
                attrs=dict(spec.attrs)
                | {"named_layout": named["parameters"][spec.name]},
            )
            if spec.name in named["parameters"]
            else spec
            for spec in specs
        )

    name = request.kernel_name or application.__name__

    try:
        if function is None:
            program = from_application(application, specs, kind=name, strict=True)
        else:
            from ninetoothed.frontend.python import _from_function

            program = _from_function(
                function, specs, kind=name, strict=True, named=True
            )
    except LoweringError as exc:
        raise LoweringError(
            f"Cannot lower `{application.__name__}` through the SSA backend path: {exc}."
        ) from exc

    if program is None:
        raise LoweringError(
            f"Cannot lower `{application.__name__}` through the SSA backend path: "
            "source inspection did not produce ssa.Program."
        )

    domains = {}
    defaults = {}
    symbols = set(program.metadata.get("symbols", ()))

    for tensor in arranged:
        for symbol in tensor.names():
            key = str(symbol)
            symbols.add(key)

            if not hasattr(symbol, "lower_bound"):
                continue

            low, high = int(symbol.lower_bound), int(symbol.upper_bound)
            default = min(max(256, low), high)
            power = bool(getattr(symbol, "power_of_two", False))

            if power:
                default = 1 << max(0, default.bit_length() - 1)

                if default < low:
                    default = 1 << max(0, (low - 1).bit_length())

                if default > high:
                    raise LoweringError(
                        f"Meta domain {key!r} contains no legal power-of-two value."
                    )

            defaults[key] = default

            if not is_meta(key):
                continue

            domains[key] = {
                "lower": low,
                "upper": high,
                "power_of_two": power,
                "default": default,
                "scope": "hard_layout" if named is not None else "triton_tuning",
            }

    roles = {}

    def declare(expression, role, *, override=False):
        if expression.op == "symbol":
            symbols.add(expression.value)

            if override:
                roles[expression.value] = role
            else:
                roles.setdefault(expression.value, role)

        for operand in expression.operands:
            declare(operand, role, override=override)

    for spec in specs:
        for expression in (
            *spec.attrs.get("source_shape", ()),
            *spec.attrs.get("source_strides", ()),
        ):
            declare(
                IndexExpr.parse(expression),
                "extent"
                if expression in spec.attrs.get("source_shape", ())
                else "stride",
            )

        symbols.update(
            str(value)
            for key, value in spec.attrs.items()
            if key.startswith("jagged_") and key.endswith("_param")
        )

        if spec.jagged_dim is not None:
            for field in ("values_numel", "offsets_numel", "max_seq_len", "seq_len"):
                roles[spec.attrs[f"jagged_{field}_param"]] = "extent"

    # Legacy constexpr tile symbols have no tuning domain, but still describe
    # extents. Infer their role from layout use rather than their spelling.
    for spec in specs:
        for text in (
            *spec.shape,
            *spec.attrs.get("application_shape", ()),
            *(dim for shape in spec.attrs.get("dtype_shapes", ()) for dim in shape),
        ):
            declare(IndexExpr.parse(text), "extent", override=True)

    # Scalar inputs are SSA values, not additional index-symbol definitions.
    symbols.difference_update(params)

    def external_calls(operations):
        for operation in operations:
            if operation.opcode.startswith("call."):
                yield (
                    operation.opcode,
                    operation.attrs.get("callee"),
                    len(operation.operands),
                )

            for region in operation.regions:
                yield from external_calls(region.operations)

    program = replace(
        program,
        metadata=dict(program.metadata)
        | {
            "shape_validation": "strict",
            "external_calls": tuple(external_calls(program.blocks[0].operations)),
            "symbols": tuple(
                name
                for name in program.metadata.get("symbols", ())
                if name not in params
            ),
        },
    )

    try:
        source = textwrap.dedent(inspect.getsource(application))
    except OSError:
        source = ""

    return verify_module(
        FrontendModule(
            name=name,
            tensors=specs,
            program=program,
            symbols=tuple(sorted(symbols)),
            symbol_roles={
                key: "meta" if key in domains else roles.get(key, "index")
                for key in symbols
            },
            meta_domains=domains,
            parameter_defaults=defaults,
            named_layout=named,
            source=source,
            constexpr_values={
                name: getattr(getattr(tensor, "source", tensor), "value", None)
                for name, tensor in zip(params, arranged)
            },
        )
    )


__all__ = ["freeze_application"]
