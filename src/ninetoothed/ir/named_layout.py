"""Versioned named-layout obligations, independent of Python and all backends."""

from collections.abc import Mapping
from dataclasses import replace

from ninetoothed.errors import LoweringError
from ninetoothed.ir import IndexExpr


def specialize_contract(contract, values):
    if contract is None:
        return None

    return dict(contract) | {
        "parameters": {
            name: specialize_binding(binding, values)
            for name, binding in contract["parameters"].items()
        }
    }


def specialize_binding(binding, values):
    from ninetoothed.ir.module import evaluate_index_expression

    def bind(expression):
        expression = IndexExpr.parse(expression)

        if expression.op == "symbol" and expression.value in values:
            return IndexExpr.parse(values[expression.value])

        expression = replace(
            expression, operands=tuple(bind(item) for item in expression.operands)
        )

        if expression.operands and all(
            item.op == "constant" for item in expression.operands
        ):
            return IndexExpr.parse(evaluate_index_expression(expression, {}))

        return expression

    return dict(binding) | {
        "tile": tuple(bind(size).render() for size in binding["tile"])
    }


def verify_named_layout(module):
    """Validate semantic identity and computation before any backend pass."""
    contract = module.named_layout
    marked = {spec.name for spec in module.tensors if "named_layout" in spec.attrs}

    if contract is None:
        if marked:
            raise LoweringError("Named layouts cannot discard their semantic contract.")

        return

    if (
        not isinstance(contract, Mapping)
        or set(contract)
        != {"version", "parameters", "output", "streams", "bound_stream"}
        or contract["version"] != 1
    ):
        raise LoweringError("Unsupported named layout contract.")

    specs = {spec.name: spec for spec in module.tensors}
    parameters = contract["parameters"]
    positive = {name for name, spec in specs.items() if spec.ndim}

    if (
        not isinstance(parameters, Mapping)
        or set(parameters) != marked
        or marked != positive
    ):
        raise LoweringError("Named contract must describe every data parameter.")

    if contract["output"] not in parameters:
        raise LoweringError("Named contract requires one positive-rank output.")

    groups = {}
    value_axes = {}

    for name, binding in parameters.items():
        spec = specs[name]

        if not isinstance(binding, Mapping) or set(binding) != {
            "axes",
            "order",
            "tile",
        }:
            raise LoweringError("Malformed named parameter binding.")

        axes, order, tiles = (binding[key] for key in ("axes", "order", "tile"))

        if (
            len(axes) not in (1, 2)
            or len(axes) != len(spec.attrs["source_shape"])
            or any(type(axis) is not str or not axis.strip() for axis in axes)
            or len(set(axes)) != len(axes)
            or any(type(dim) is not int for dim in order)
            or sorted(order) != list(range(len(axes)))
            or len(tiles) != len(axes)
            or dict(spec.attrs["named_layout"]) != dict(binding)
        ):
            raise LoweringError(f"Invalid named axes or provenance for {name!r}.")

        names = tuple(axes[i] for i in order)
        value_axes[name] = names

        for axis, dim, tile in zip(names, order, tiles):
            expression = IndexExpr.parse(tile)

            if expression.op == "constant":
                if type(expression.value) is not int or expression.value <= 0:
                    raise LoweringError("Named tile sizes must be positive integers.")
            elif (
                expression.op != "symbol"
                or expression.value not in module.parameter_defaults
            ):
                raise LoweringError(
                    "Named tile must be a bounded compile-time parameter."
                )

            extent = IndexExpr.parse(spec.attrs["source_shape"][dim])

            if extent.op == "constant" and (
                type(extent.value) is not int or extent.value <= 0
            ):
                raise LoweringError(f"Named axis {axis!r} requires positive extents.")

            groups.setdefault(axis, []).append((name, dim, expression, extent))

    for axis, bindings in groups.items():
        first = bindings[0]

        for name, dim, tile, extent in bindings[1:]:
            if tile != first[2] or (
                extent.op == first[3].op == "constant"
                and extent.value != first[3].value
            ):
                raise LoweringError(
                    f"Named axis {axis!r} conflicts between {first[0]}[{first[1]}] and {name}[{dim}]."
                )

    output = contract["output"]
    streams = tuple(contract["streams"])
    output_axes = value_axes[output]

    if len(set(streams)) != len(streams) or any(
        name not in parameters or name == output for name in streams
    ):
        raise LoweringError("Invalid paired named streams.")

    if streams:
        extras = [set(value_axes[name]) - set(output_axes) for name in streams]

        if (
            len(streams) != 2
            or contract["bound_stream"] not in streams
            or len(output_axes) != 2
            or len(extras[0]) != 1
            or extras[0] != extras[1]
        ):
            raise LoweringError("Named S1 requires one axis and two paired streams.")
    elif contract["bound_stream"] is not None:
        raise LoweringError("A named stream bound requires paired inputs.")

    for name in parameters:
        if name not in streams:
            if set(value_axes[name]) - set(output_axes):
                raise LoweringError("Named source-only axes require block iteration.")

            value_axes[name] = output_axes

    _verify_program(module.program, contract, value_axes)


def _verify_program(program, contract, value_axes):
    output = contract["output"]
    streams = set(contract["streams"])
    top = program.blocks[0].operations
    loops = [op for op in top if op.opcode == "scf.for"]
    definitions = {value.name: op for op in top for value in op.results}

    if len(loops) != bool(streams) or not top or top[-1].opcode != "mem.store":
        raise LoweringError(
            "Named program requires one final store and its paired loop."
        )

    if tuple(value.name for value in program.outputs) != (output,):
        raise LoweringError("Named output effects differ from the contract.")

    induction = None

    if loops:
        loop = loops[0]

        if len(loop.results) != 1:
            raise LoweringError("Named S1 requires exactly one loop-carried value.")

        start, bound, step = (definitions.get(name) for name in loop.operands[:3])

        if (
            start is None
            or start.opcode != "arith.constant"
            or start.attrs["value"] != 0
            or step is None
            or step.opcode != "arith.constant"
            or step.attrs["value"] != 1
            or bound is None
            or bound.opcode != "shape.dim"
            or bound.operands != (contract["bound_stream"],)
            or bound.attrs.get("source")
            or bound.attrs.get("dim") != 0
        ):
            raise LoweringError(
                "Named loop must traverse complete paired block streams."
            )

        induction = loop.regions[0].args[0].name

    extracted = set()
    stores = []

    def merge(domains):
        rank = max(map(len, domains), default=0)
        result = [None] * rank

        for domain in domains:
            for i, axis in enumerate((None,) * (rank - len(domain)) + domain):
                if axis is not None:
                    if result[i] not in (None, axis):
                        raise LoweringError("Named value axes do not align.")

                    result[i] = axis

        return tuple(result)

    def walk(operations, domains, in_loop=False):
        domains = dict(domains)

        for op in operations:
            code = op.opcode
            args = [domains.get(name, ()) for name in op.operands]

            for i, operand in enumerate(op.operands):
                if (
                    operand == output
                    and code not in {"shape.dim", "dtype.of", "tensor.stride"}
                    and not (code == "mem.store" and i == 1)
                ):
                    raise LoweringError(
                        "Named output cannot be read before its final store."
                    )

                if operand in streams and code not in {
                    "shape.dim",
                    "dtype.of",
                    "tensor.stride",
                    "tensor.extract",
                }:
                    raise LoweringError(
                        "Named streams must be read through paired extraction."
                    )

            if code == "scf.for":
                if in_loop or op not in loops:
                    raise LoweringError("Named nested control flow is unsupported.")

                body = op.regions[0]
                nested = dict(domains) | {body.args[0].name: ()}
                nested.update(
                    (arg.name, domain) for arg, domain in zip(body.args[1:], args[3:])
                )
                yielded = walk(body.operations, nested, True)

                if yielded != tuple(args[3:]):
                    raise LoweringError("Named loop-carried axes must stay unchanged.")

                for result, domain in zip(op.results, args[3:]):
                    domains[result.name] = domain

                continue

            if code == "scf.yield":
                return tuple(args)

            if code == "tensor.extract":
                if (
                    not in_loop
                    or op.operands[0] not in streams
                    or op.operands != (op.operands[0], induction)
                    or op.attrs.get("source")
                ):
                    raise LoweringError(
                        "Named streams require the same unmodified block index."
                    )

                extracted.add(op.operands[0])
                domain = value_axes[op.operands[0]]
            elif code in {"linalg.dot", "linalg.matmul"}:
                a, b = args

                if (
                    len(a) != 2
                    or len(b) != 2
                    or a[-1] != b[0]
                    or a[-1] in (None, *value_axes[output])
                ):
                    raise LoweringError("Dot axes must match the named contraction.")

                domain = (a[0], b[1])
            elif code == "linalg.transpose":
                domain = tuple(reversed(args[0]))
            elif code in {"tensor.zeros", "tensor.full", "tensor.empty"}:
                origins = op.attrs.get("shape_origins", ())
                inputs = {value.name: value for value in program.inputs}

                if len(origins) != len(op.attrs["shape_values"]) or any(
                    name not in inputs
                    or name in streams
                    or type(dim) is not int
                    or not 0 <= dim < len(inputs[name].type.shape)
                    or str(inputs[name].type.shape[dim]) != str(size)
                    for (name, dim), size in zip(origins, op.attrs["shape_values"])
                ):
                    raise LoweringError(
                        "Named constructor shapes require checked block origins."
                    )

                domain = tuple(domains[name][dim] for name, dim in origins)
            elif code == "tensor.cast":
                domain = args[0]
            elif code.startswith(("shape.", "dtype.")) or code in {
                "tensor.stride",
                "tuple.construct",
            }:
                domain = ()
            elif code.startswith(("arith.", "math.", "cmp.", "select.")):
                domain = merge(args)
            elif code == "mem.store":
                if (
                    in_loop
                    or op is not top[-1]
                    or op.operands[1] != output
                    or op.attrs.get("indices")
                    or op.attrs.get("subscript")
                ):
                    raise LoweringError(
                        "Named output requires a final full-tile store."
                    )

                if merge((args[0], value_axes[output])) != value_axes[output]:
                    raise LoweringError("Named store axes differ from the output.")

                stores.append(op)
                domain = ()
            else:
                raise LoweringError(f"Operation {code!r} is not supported in named S1.")

            for result in op.results:
                if result.type.kind == "pointer":
                    raise LoweringError("Named S1 does not support raw pointers.")

                if (
                    result.type.kind == "tensor"
                    and code != "shape.dim"
                    and len(domain) != len(result.type.shape)
                ):
                    raise LoweringError(f"Named domain rank differs at {code}.")

                domains[result.name] = domain

        return ()

    walk(top, dict(value_axes))

    if extracted != streams or len(stores) != 1:
        raise LoweringError("Named program must consume every stream and write once.")
