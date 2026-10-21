"""Bound the live vector size of direct Ascend load reductions."""

import ast
import copy


def _language_member(node, member):
    return (
        isinstance(node, ast.Attribute)
        and node.attr == member
        and ast.unparse(node.value) == "triton.language"
    )


def _empty_slice_bound(node):
    return node is None or isinstance(node, ast.Constant) and node.value is None


def _loaded_value(node):
    while (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "to"
    ):
        node = node.func.value

    return isinstance(node, ast.Call) and _language_member(node.func, "load")


def _reduction_axis(node):
    axis = next((kw.value for kw in node.keywords if kw.arg == "axis"), None)

    if axis is None and len(node.args) > 1:
        axis = node.args[1]

    if isinstance(axis, ast.Constant) and isinstance(axis.value, int):
        return axis.value

    return None


def _arange_extent(node):
    if (
        isinstance(node, ast.Call)
        and _language_member(node.func, "arange")
        and len(node.args) == 2
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == 0
    ):
        return node.args[1]

    return None


class _OffsetArange(ast.NodeTransformer):
    def __init__(self, extent, offset, chunk_size):
        self.extent = ast.dump(extent)
        self.offset = offset
        self.chunk_size = chunk_size

    def visit_Call(self, node):
        extent = _arange_extent(node)

        if extent is not None and ast.dump(extent) == self.extent:
            node.args[1] = ast.Constant(value=self.chunk_size)

            return ast.BinOp(left=copy.deepcopy(self.offset), op=ast.Add(), right=node)

        return self.generic_visit(node)


class _ExtractReductions(ast.NodeTransformer):
    def __init__(self, names, constexprs, max_elements, chunk_size):
        self.names = names
        self.constexprs = constexprs
        self.max_elements = max_elements
        self.chunk_size = chunk_size
        self.prelude = []

    def _fresh(self, suffix):
        index = 0

        while (name := f"ninetoothed_reduction_{index}_{suffix}") in self.names:
            index += 1

        self.names.add(name)

        return name

    def visit_IfExp(self, node):
        return node

    def visit_BoolOp(self, node):
        return node

    def visit_Lambda(self, node):
        return node

    def _extent(self, value, axis):
        parents = {
            child: parent
            for parent in ast.walk(value)
            for child in ast.iter_child_nodes(parent)
        }
        candidates = {}

        for node in ast.walk(value):
            if isinstance(node, ast.Call) and not (
                _language_member(node.func, "load")
                or _language_member(node.func, "arange")
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "to")
                or (isinstance(node.func, ast.Name) and node.func.id == "float")
            ):
                return None

            extent = _arange_extent(node)

            if extent is None:
                continue

            if isinstance(extent, ast.Constant):
                if (
                    not isinstance(extent.value, int)
                    or extent.value <= self.max_elements
                ):
                    continue
            elif not (isinstance(extent, ast.Name) and extent.id in self.constexprs):
                continue

            parent = parents.get(node)

            if isinstance(parent, ast.Subscript) and parent.value is node:
                dims = (
                    parent.slice.elts
                    if isinstance(parent.slice, ast.Tuple)
                    else [parent.slice]
                )
                varying_axes = [
                    i for i, dim in enumerate(dims) if isinstance(dim, ast.Slice)
                ]

                if any(
                    not (
                        isinstance(dim, ast.Constant)
                        and dim.value is None
                        or isinstance(dim, ast.Slice)
                        and all(
                            _empty_slice_bound(bound)
                            for bound in (dim.lower, dim.upper, dim.step)
                        )
                    )
                    for dim in dims
                ):
                    return None

                if varying_axes != [axis % len(dims)]:
                    return None
            elif axis not in (0, -1):
                return None

            candidates[ast.dump(extent)] = extent

        if len(candidates) == 1:
            return next(iter(candidates.values()))

        return None

    def visit_Call(self, node):
        if not (
            any(_language_member(node.func, member) for member in ("sum", "min", "max"))
            and node.args
            and _loaded_value(node.args[0])
        ):
            return self.generic_visit(node)

        axis = _reduction_axis(node)

        if axis is None:
            return node

        extent = self._extent(node.args[0], axis)

        if extent is None:
            return node

        accumulator = self._fresh("value")
        offset = self._fresh("offset")

        def part(start):
            return _OffsetArange(extent, start, self.chunk_size).visit(
                copy.deepcopy(node)
            )

        first = ast.Assign(
            targets=[ast.Name(id=accumulator, ctx=ast.Store())],
            value=part(ast.Constant(value=0)),
        )
        partial = part(ast.Name(id=offset, ctx=ast.Load()))
        current = ast.Name(id=accumulator, ctx=ast.Load())

        if node.func.attr == "sum":
            combined = ast.BinOp(left=current, op=ast.Add(), right=partial)
        else:
            combined = ast.Call(
                func=ast.Attribute(
                    value=copy.deepcopy(node.func.value),
                    attr="minimum" if node.func.attr == "min" else "maximum",
                    ctx=ast.Load(),
                ),
                args=[current, partial],
                keywords=[],
            )

        loop = ast.For(
            target=ast.Name(id=offset, ctx=ast.Store()),
            iter=ast.Call(
                func=ast.Name(id="range", ctx=ast.Load()),
                args=[
                    ast.Constant(value=self.chunk_size),
                    copy.deepcopy(extent),
                    ast.Constant(value=self.chunk_size),
                ],
                keywords=[],
            ),
            body=[
                ast.Assign(
                    targets=[ast.Name(id=accumulator, ctx=ast.Store())], value=combined
                )
            ],
            orelse=[],
        )
        statements = [first, loop]

        if not isinstance(extent, ast.Constant):
            statements = [
                ast.If(
                    test=ast.Compare(
                        left=copy.deepcopy(extent),
                        ops=[ast.Gt()],
                        comparators=[ast.Constant(value=self.max_elements)],
                    ),
                    body=statements,
                    orelse=[
                        ast.Assign(
                            targets=[ast.Name(id=accumulator, ctx=ast.Store())],
                            value=copy.deepcopy(node),
                        )
                    ],
                )
            ]

        self.prelude.extend(statements)

        return ast.Name(id=accumulator, ctx=ast.Load())


def split_large_load_reductions(module, *, max_elements=32768, chunk_size=2048):
    """Split direct load reductions while preserving the original load masks.

    The first chunk initializes the accumulator with the reduction's own
    result dtype. Values already materialized in a local variable and
    reductions across ambiguous multidimensional layouts are left unchanged.
    """
    if chunk_size <= 0 or chunk_size & (chunk_size - 1) or max_elements < chunk_size:
        raise ValueError("Reduction chunk size must be a bounded power of two.")

    for function in module.body:
        if not isinstance(function, ast.FunctionDef):
            continue

        names = {node.id for node in ast.walk(function) if isinstance(node, ast.Name)}
        names.update(arg.arg for arg in function.args.args)
        constexprs = {
            arg.arg
            for arg in function.args.args
            if arg.annotation is not None
            and _language_member(arg.annotation, "constexpr")
        }
        body = []

        for statement in function.body:
            if isinstance(statement, ast.Assign) or (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and _language_member(statement.value.func, "store")
            ):
                extractor = _ExtractReductions(
                    names, constexprs, max_elements, chunk_size
                )
                statement = extractor.visit(statement)
                body.extend(extractor.prelude)

            body.append(statement)

        function.body = body

    return ast.fix_missing_locations(module)
