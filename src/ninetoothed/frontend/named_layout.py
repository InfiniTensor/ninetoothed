"""S1 named layouts: checked logical axes to existing anonymous Tensor views."""

import ast
import builtins
import copy
import types

from ninetoothed.errors import LoweringError
from ninetoothed.ir import IndexExpr
from ninetoothed.naming import is_constexpr, is_meta
from ninetoothed.symbol import Symbol


def transform_axes(tensor, operation, args, kwargs):
    """Keep the public named subset closed; internal normalization is anonymous."""
    layout = dict(tensor._named_layout)

    if layout["tile"] or operation not in {"permute", "tile"}:
        raise ValueError("Named layouts require axes -> optional permute -> one tile.")

    if len(args) != 1 or kwargs:
        raise ValueError("Named permute/tile requires one explicit positional tuple.")

    values = tuple(args[0])

    if operation == "permute":
        if (
            layout["permuted"]
            or any(type(i) is not int for i in values)
            or sorted(values) != list(range(tensor.ndim))
        ):
            raise ValueError(
                "Named permute requires one permutation of all dimensions."
            )

        layout["order"] = tuple(layout["order"][i] for i in values)
        layout["permuted"] = True
    else:
        if len(values) != tensor.ndim or any(not _tile_size(v) for v in values):
            raise ValueError("Named tile sizes must be positive compile-time integers.")

        layout["tile"] = tuple(str(value) for value in values)

    return layout


def _tile_size(value):
    return (type(value) is int and value > 0) or (
        isinstance(value, Symbol)
        and Symbol.is_name(value)
        and (is_constexpr(str(value)) or is_meta(str(value)))
        and getattr(value, "lower_bound", 0) > 0
        and getattr(value, "upper_bound", 0) >= value.lower_bound
    )


def _names(layout):
    return tuple(layout["axes"][i] for i in layout["order"])


def validate_scope(function, scope):
    """Require intrinsic calls to resolve to their real language namespace."""
    assigned = {arg.arg for arg in function.args.args} | {
        node.id
        for node in ast.walk(function)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }

    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue

        if isinstance(node.func, ast.Name):
            if (
                node.func.id != "range"
                or "range" in assigned
                or scope.get("range", builtins.range) is not builtins.range
            ):
                raise LoweringError(
                    "Named calls require language intrinsics or inlineable helpers."
                )

            continue

        if not isinstance(node.func, ast.Attribute):
            raise LoweringError("Indirect named calls are unsupported.")

        receiver = node.func.value

        if isinstance(receiver, ast.Name) and receiver.id in assigned:
            if node.func.attr != "to":
                raise LoweringError(
                    "Named tensor methods only support explicit dtype conversion."
                )

            continue

        attributes = []

        while isinstance(receiver, ast.Attribute):
            attributes.append(receiver.attr)
            receiver = receiver.value

        if not isinstance(receiver, ast.Name):
            continue

        value = scope.get(receiver.id)

        for attribute in reversed(attributes):
            value = (
                vars(value).get(attribute)
                if isinstance(value, types.ModuleType)
                else None
            )

        if not isinstance(value, types.ModuleType) or value.__name__ not in {
            "ninetoothed.language",
            "triton.language",
            "math",
        }:
            raise LoweringError("Named intrinsic namespace is missing or shadowed.")


def normalize(function, params, tensors):
    """Run only after helper expansion, before ordinary syntax-directed SSA."""
    positive = {name: t for name, t in zip(params, tensors) if t.ndim}

    if any(t._named_layout is None for t in positive.values()):
        raise LoweringError("Every positive-rank parameter needs axes() in named mode.")

    identities = [t.source._source_identity for t in positive.values()]
    source_names = [t.source.name for t in positive.values()]

    if len(set(identities)) != len(identities) or len(set(source_names)) != len(
        source_names
    ):
        raise LoweringError(
            "Named parameters require distinct root Tensor specs and source names."
        )

    bindings = {}

    for name, tensor in positive.items():
        layout = tensor._named_layout

        if tensor.source.ndim not in (1, 2) or tensor.source.jagged_dim is not None:
            raise LoweringError("Named S1 supports non-jagged rank-1/rank-2 tensors.")

        if not layout["tile"] or len(tensor._history) != 1 + layout["permuted"]:
            raise LoweringError("Named parameters need one unmodified tiled hierarchy.")

        if (
            tensor.dtype.source._source_identity is not tensor.source._source_identity
            or tensor.dtype._history
            or tuple(IndexExpr.parse(size) for size in tensor.dtype.shape)
            != tuple(IndexExpr.parse(size) for size in layout["tile"])
        ):
            raise LoweringError("Named tile values cannot be manually reshaped.")

        bindings[name] = {
            "axes": layout["axes"],
            "order": layout["order"],
            "tile": layout["tile"],
        }

    # A parameter assignment is an effect, not an arbitrary output-name heuristic.
    writes = []

    for node in ast.walk(function):
        if isinstance(node, (ast.If, ast.While, ast.Break, ast.Continue, ast.Return)):
            raise LoweringError(
                "Named S1 supports straight-line code and one plain loop."
            )

        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)

            for target in targets:
                root = target.value if isinstance(target, ast.Subscript) else target

                if isinstance(root, ast.Name) and root.id in params:
                    writes.append((node, target, root.id))

    if len(writes) != 1:
        raise LoweringError("Named S1 requires one final full-tile output assignment.")

    statement, target, output = writes[0]

    if (
        statement is not function.body[-1]
        or not isinstance(statement, ast.Assign)
        or len(statement.targets) != 1
        or output not in positive
        or (
            isinstance(target, ast.Subscript)
            and not (
                isinstance(target.slice, ast.Constant)
                and target.slice.value is Ellipsis
            )
        )
    ):
        raise LoweringError("Named output must be the final full-tile assignment.")

    statement.targets[0] = ast.copy_location(
        ast.Name(id=output, ctx=ast.Store()), target
    )
    output_axes = _names(bindings[output])
    loops = [node for node in ast.walk(function) if isinstance(node, ast.For)]
    streams = ()
    bound_stream = None

    if len(loops) > 1 or (loops and loops[0] not in function.body):
        raise LoweringError("Named S1 allows one non-nested top-level block loop.")

    if loops:
        loop = loops[0]
        call = loop.iter

        if (
            not isinstance(loop.target, ast.Name)
            or loop.orelse
            or not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Name)
            or call.func.id != "range"
            or call.keywords
            or len(call.args) not in (1, 2, 3)
        ):
            raise LoweringError("Named loops require range(stream.shape[0]).")

        for expr, value in ((call.args[0], 0),) if len(call.args) > 1 else ():
            if (
                not isinstance(expr, ast.Constant)
                or type(expr.value) is not int
                or expr.value != value
            ):
                raise LoweringError("Named block loops must start at zero.")

        if len(call.args) == 3 and not (
            isinstance(call.args[2], ast.Constant)
            and type(call.args[2].value) is int
            and call.args[2].value == 1
        ):
            raise LoweringError("Named block loops require unit steps.")

        bound = call.args[0 if len(call.args) == 1 else 1]

        if not (
            isinstance(bound, ast.Subscript)
            and isinstance(bound.value, ast.Attribute)
            and bound.value.attr == "shape"
            and isinstance(bound.value.value, ast.Name)
            and isinstance(bound.slice, ast.Constant)
            and type(bound.slice.value) is int
            and bound.slice.value == 0
        ):
            raise LoweringError("Named loop bounds must be stream.shape[0].")

        bound_stream = bound.value.value.id
        streams = tuple(
            name
            for name, layout in bindings.items()
            if set(_names(layout)) - set(output_axes)
        )

        if len(streams) != 2 or bound_stream not in streams:
            raise LoweringError("Named loops require two paired input streams.")

        extras = [set(_names(bindings[name])) - set(output_axes) for name in streams]

        if len(extras[0]) != 1 or extras[0] != extras[1]:
            raise LoweringError("Named streams need the same single source-only axis.")

        if len(output_axes) != 2:
            raise LoweringError("Named block loops require a matrix output.")

        for node in ast.walk(loop):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Store)
                and node.id == loop.target.id
                and node is not loop.target
            ):
                raise LoweringError("The named block index cannot be reassigned.")

    elif any(set(_names(layout)) - set(output_axes) for layout in bindings.values()):
        raise LoweringError("Source-only named axes require a paired block loop.")

    normalized = []

    for name, tensor in zip(params, tensors):
        if not tensor.ndim:
            normalized.append(tensor)
            continue

        tensor = copy.deepcopy(tensor)
        tensor._named_layout = None
        names = _names(bindings[name])

        if name in streams:
            # Exactly the old tile/expand/squeeze projection, without reordering
            # block-internal axes. A dot is never an implicit matrix transpose.
            extra = next(iter(set(names) - set(output_axes)))
            slots = [
                names.index(axis) if axis in names else names.index(extra)
                for axis in output_axes
            ]

            if len(set(slots)) != len(names):
                raise LoweringError("S1 streams must share one output axis each.")

            if slots != list(range(len(names))):
                tensor = tensor.permute(tuple(slots))

            ordered = tuple(names[i] for i in slots)
            tensor = tensor.tile(tuple(-1 if axis == extra else 1 for axis in ordered))
            tensor = tensor.expand(
                tuple(
                    size if axis == extra else -1
                    for axis, size in zip(ordered, positive[output].shape)
                )
            )
            tensor.dtype = tensor.dtype.squeeze(
                tuple(i for i, axis in enumerate(ordered) if axis != extra)
            )
        elif name != output:
            order = tuple(names.index(axis) for axis in output_axes if axis in names)

            if order != tuple(range(len(names))):
                tensor = tensor.permute(order)
                tensor.dtype = tensor.dtype.permute(order)

            for dim, axis in enumerate(output_axes):
                if axis not in names:
                    tensor = tensor.unsqueeze(dim)
                    tensor.dtype = tensor.dtype.unsqueeze(dim)

            if set(names) != set(output_axes):
                tensor = tensor.expand(
                    tuple(
                        -1 if axis in names else size
                        for axis, size in zip(output_axes, positive[output].shape)
                    )
                )
                tensor.dtype = tensor.dtype.expand(
                    tuple(
                        -1 if axis in names else size
                        for axis, size in zip(output_axes, positive[output].dtype.shape)
                    )
                )

        normalized.append(tensor)

    return tuple(normalized), {
        "version": 1,
        "parameters": bindings,
        "output": output,
        "streams": streams,
        "bound_stream": bound_stream,
    }
