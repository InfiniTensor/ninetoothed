"""SSA lowering from NineToothed application Python AST.

This module deliberately lowers computation structure instead of recognizing
whole operators.  Complex fused kernels are represented with ``scf`` regions,
tensor operations, reductions, masks, scalar math, and stores rather than with
operator-level SSA opcodes.
"""

from __future__ import annotations

import ast
import inspect
import math
import textwrap
from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

from ninetoothed.ir import (
    SSABlockIR,
    SSAOperationIR,
    SSAProgramIR,
    SSATypeIR,
    SSAValueIR,
    TensorTypeIR,
)


class SSALoweringError(ValueError):
    """Raised when a Python construct is outside this SSA lowerer's subset."""


def application_to_ssa(
    application: Any,
    tensor_irs: tuple[TensorTypeIR, ...] = (),
    *,
    kind: str | None = None,
) -> SSAProgramIR | None:
    """Lower a NineToothed application function to generic SSA.

    The pass is syntax-directed and target-neutral.  It is intentionally not a
    semantic pattern matcher for named operators; it only sees Python AST and
    emits generic operations.
    """
    try:
        source = inspect.getsource(application)
    except OSError:
        return None

    return source_to_ssa(
        source,
        tensor_irs=tensor_irs,
        kind=kind or getattr(application, "__name__", "application"),
        globalns=getattr(application, "__globals__", None),
    )


def source_to_ssa(
    source: str,
    tensor_irs: tuple[TensorTypeIR, ...] = (),
    *,
    kind: str = "application",
    globalns: Mapping[str, Any] | None = None,
) -> SSAProgramIR | None:
    tree = ast.parse(textwrap.dedent(source))
    func = next((node for node in tree.body if isinstance(node, ast.FunctionDef)), None)

    if func is None:
        return None

    if globalns is not None:
        func = _InlineHelperCalls(globalns).inline(func)

    builder = _ApplicationSSABuilder(func, tensor_irs, kind)
    builder.lower()

    return builder.finish()


class _InlineHelperCalls:
    """Inline user-defined helper calls before syntax-directed SSA lowering."""

    def __init__(self, globalns: Mapping[str, Any]):
        self.globalns = globalns
        self.counter = 0
        self.stack: set[Any] = set()

    def inline(self, func: ast.FunctionDef) -> ast.FunctionDef:
        func = deepcopy(func)
        func.body = self._inline_statements(func.body)
        ast.fix_missing_locations(func)

        return func

    def _inline_statements(self, statements: Iterable[ast.stmt]) -> list[ast.stmt]:
        result: list[ast.stmt] = []

        for stmt in statements:
            result.extend(self._inline_statement(deepcopy(stmt)))
        return result

    def _inline_statement(self, stmt: ast.stmt) -> list[ast.stmt]:
        if isinstance(stmt, ast.Assign):
            value, prefix = self._inline_expr(stmt.value)
            stmt.value = value

            return [*prefix, stmt]

        if isinstance(stmt, ast.AnnAssign):
            if stmt.value is None:
                return [stmt]

            value, prefix = self._inline_expr(stmt.value)
            stmt.value = value

            return [*prefix, stmt]

        if isinstance(stmt, ast.AugAssign):
            value, prefix = self._inline_expr(stmt.value)
            stmt.value = value

            return [*prefix, stmt]

        if isinstance(stmt, ast.Return):
            if stmt.value is None:
                return [stmt]

            value, prefix = self._inline_expr(stmt.value)
            stmt.value = value

            return [*prefix, stmt]

        if isinstance(stmt, ast.Expr):
            value, prefix = self._inline_expr(stmt.value)
            stmt.value = value

            if isinstance(value, ast.Constant) and value.value is None:
                return prefix
            return [*prefix, stmt]

        if isinstance(stmt, ast.If):
            test, prefix = self._inline_expr(stmt.test)
            stmt.test = test
            stmt.body = self._inline_statements(stmt.body)
            stmt.orelse = self._inline_statements(stmt.orelse)

            return [*prefix, stmt]

        if isinstance(stmt, ast.For):
            iter_expr, prefix = self._inline_expr(stmt.iter)
            stmt.iter = iter_expr
            stmt.body = self._inline_statements(stmt.body)
            stmt.orelse = self._inline_statements(stmt.orelse)

            return [*prefix, stmt]
        return [stmt]

    def _inline_expr(self, expr: ast.AST) -> tuple[ast.AST, list[ast.stmt]]:
        prefix: list[ast.stmt] = []

        if isinstance(expr, ast.Call):
            func, func_prefix = self._inline_expr(expr.func)
            prefix.extend(func_prefix)
            args = []

            for arg in expr.args:
                lowered, arg_prefix = self._inline_expr(arg)
                prefix.extend(arg_prefix)
                args.append(lowered)

            keywords = []

            for keyword in expr.keywords:
                if keyword.arg is None:
                    return expr, prefix

                value, keyword_prefix = self._inline_expr(keyword.value)
                prefix.extend(keyword_prefix)
                keywords.append(ast.keyword(arg=keyword.arg, value=value))

            expr = ast.Call(func=func, args=args, keywords=keywords)
            inlined = self._inline_call(expr)

            if inlined is not None:
                value, statements = inlined

                return value, [*prefix, *statements]
            return expr, prefix

        for field, value in ast.iter_fields(expr):
            if isinstance(value, ast.AST):
                lowered, field_prefix = self._inline_expr(value)
                prefix.extend(field_prefix)
                setattr(expr, field, lowered)
            elif isinstance(value, list):
                items = []

                for item in value:
                    if isinstance(item, ast.AST):
                        lowered, item_prefix = self._inline_expr(item)
                        prefix.extend(item_prefix)
                        items.append(lowered)
                    else:
                        items.append(item)

                setattr(expr, field, items)

        return expr, prefix

    def _inline_call(self, node: ast.Call) -> tuple[ast.AST, list[ast.stmt]] | None:
        func = self._resolve_user_function(node.func)

        if func is None or func in self.stack:
            return None

        source = _function_source(func)

        if source is None:
            return None

        helper = _find_function_def(source, getattr(func, "__name__", ""))

        if helper is None:
            return None

        binding = _bind_call_arguments(helper, node)

        if binding is None:
            return None

        self.stack.add(func)

        try:
            body = deepcopy(helper.body)
            local_names = set(_assigned_names(body)) - set(binding)
            prefix = f"__nt_inline_{self.counter}_"
            self.counter += 1
            body = [_RenameLocals(local_names, prefix).visit(stmt) for stmt in body]
            body = [_ReplaceParameters(binding).visit(stmt) for stmt in body]
            inlined_body = self._inline_statements(body)
        finally:
            self.stack.remove(func)

        if not inlined_body or not isinstance(inlined_body[-1], ast.Return):
            return ast.Constant(value=None), inlined_body

        ret = inlined_body.pop()
        value = ret.value or ast.Constant(value=None)
        temp_name = f"{prefix}return"
        assignment = ast.Assign(
            targets=[ast.Name(id=temp_name, ctx=ast.Store())],
            value=value,
        )
        inlined_body.append(assignment)

        return ast.Name(id=temp_name, ctx=ast.Load()), inlined_body

    def _resolve_user_function(self, node: ast.AST) -> Any | None:
        obj: Any | None

        if isinstance(node, ast.Name):
            obj = self.globalns.get(node.id)
        elif isinstance(node, ast.Attribute):
            base = self._resolve_object(node.value)
            obj = None if base is None else getattr(base, node.attr, None)
        else:
            return None

        if not inspect.isfunction(obj):
            return None

        module = inspect.getmodule(obj)
        module_name = "" if module is None else module.__name__

        if (
            module_name.startswith(("ninetoothed", "torch", "triton"))
            or module_name == "math"
        ):
            return None
        return obj

    def _resolve_object(self, node: ast.AST) -> Any | None:
        if isinstance(node, ast.Name):
            return self.globalns.get(node.id)

        if isinstance(node, ast.Attribute):
            base = self._resolve_object(node.value)

            return None if base is None else getattr(base, node.attr, None)
        return None


class _RenameLocals(ast.NodeTransformer):
    def __init__(self, names: set[str], prefix: str):
        self.names = names
        self.prefix = prefix

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.names:
            return ast.copy_location(
                ast.Name(id=f"{self.prefix}{node.id}", ctx=node.ctx),
                node,
            )
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        return node


class _ReplaceParameters(ast.NodeTransformer):
    def __init__(self, binding: Mapping[str, ast.AST]):
        self.binding = binding

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self.binding:
            replacement = deepcopy(self.binding[node.id])

            return ast.copy_location(replacement, node)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        return node


def _function_source(func: Any) -> str | None:
    try:
        return inspect.getsource(func)
    except (OSError, TypeError):
        return None


def _find_function_def(source: str, name: str) -> ast.FunctionDef | None:
    tree = ast.parse(textwrap.dedent(source))

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _bind_call_arguments(
    func: ast.FunctionDef, call: ast.Call
) -> dict[str, ast.AST] | None:
    if any(keyword.arg is None for keyword in call.keywords):
        return None

    params = [arg.arg for arg in func.args.args]

    if len(call.args) > len(params):
        return None

    defaults = list(func.args.defaults)
    default_by_param = dict(zip(params[len(params) - len(defaults) :], defaults))
    binding: dict[str, ast.AST] = {}

    for name, arg in zip(params, call.args):
        binding[name] = arg

    for keyword in call.keywords:
        if keyword.arg not in params or keyword.arg in binding:
            return None

        binding[keyword.arg] = keyword.value

    for name in params:
        if name not in binding:
            if name not in default_by_param:
                return None

            binding[name] = default_by_param[name]
    return binding


def render_ssa_program(program: SSAProgramIR | None) -> str:
    """Render SSA IR as a readable textual form, not JSON."""
    if program is None:
        return "<not-available>"

    lines = [f"ssa @{program.kind} {{"]

    if program.inputs:
        lines.append("  inputs:")

        for value in program.inputs:
            lines.append(f"    {value.name} : {_format_type(value.type)}")

    if program.outputs:
        lines.append("  outputs:")

        for value in program.outputs:
            lines.append(f"    {value.name} : {_format_type(value.type)}")

    for block in program.blocks:
        _render_block(block, lines, indent=2)

    lines.append("}")

    return "\n".join(lines)


class _ApplicationSSABuilder:
    def __init__(
        self,
        func: ast.FunctionDef,
        tensor_irs: tuple[TensorTypeIR, ...],
        kind: str,
    ):
        self.func = func
        self.kind = kind
        self.param_names = tuple(arg.arg for arg in func.args.args)
        self.tensor_types = {
            tensor.name: SSATypeIR(
                "tensor" if tensor.ndim != 0 else "scalar",
                dtype=tensor.dtype,
                shape=tuple(
                    str(dim)
                    for dim in tensor.attrs.get("application_shape", tensor.shape)
                ),
                attrs={
                    "ndim": tensor.ndim,
                    "constexpr": tensor.constexpr,
                    "jagged_dim": tensor.jagged_dim,
                    "dtype_level": 0,
                }
                | dict(tensor.attrs),
            )
            for tensor in tensor_irs
        }
        self.values: dict[str, SSAValueIR] = {}
        self.outputs: list[SSAValueIR] = []
        self.operations: list[SSAOperationIR] = []
        self.env: dict[str, SSAValueIR] = {}
        self.temp_index = 0

        for name in self.param_names:
            value = self._named_value(name, self.tensor_types.get(name))
            self.env[name] = value

    def lower(self) -> None:
        self._lower_statements(self.func.body, self.operations, self.env)

    def finish(self) -> SSAProgramIR:
        metadata = {
            "source": "application_ast",
            "function": self.func.name,
            "coarse_operator_nodes": False,
            "ssa_operation_count": _count_operations(self.operations),
        }

        return SSAProgramIR(
            kind=self.kind,
            inputs=tuple(
                self.values[name] for name in self.param_names if name in self.values
            ),
            outputs=tuple(self.outputs),
            blocks=(SSABlockIR(operations=tuple(self.operations)),),
            metadata=metadata,
        )

    def _lower_statements(
        self,
        statements: Iterable[ast.stmt],
        operations: list[SSAOperationIR],
        env: dict[str, SSAValueIR],
    ) -> None:
        for stmt in statements:
            if isinstance(stmt, ast.Assign):
                self._lower_assign(stmt, operations, env)
                continue

            if isinstance(stmt, ast.AnnAssign):
                self._lower_annassign(stmt, operations, env)
                continue

            if isinstance(stmt, ast.AugAssign):
                self._lower_augassign(stmt, operations, env)
                continue

            if isinstance(stmt, ast.For):
                self._lower_for(stmt, operations, env)
                continue

            if isinstance(stmt, ast.If):
                self._lower_if(stmt, operations, env)
                continue

            if isinstance(stmt, ast.Expr):
                self._lower_expr(stmt.value, operations, env)
                continue

            if isinstance(stmt, ast.Return):
                if stmt.value is not None:
                    self._lower_expr(stmt.value, operations, env)

                continue

            if isinstance(stmt, ast.Pass):
                continue

            raise SSALoweringError(f"Unsupported statement: {ast.dump(stmt)}.")

    def _lower_assign(
        self,
        stmt: ast.Assign,
        operations: list[SSAOperationIR],
        env: dict[str, SSAValueIR],
    ) -> None:
        if len(stmt.targets) != 1:
            raise SSALoweringError("Only single-target assignments are supported.")

        target = stmt.targets[0]
        value = self._lower_expr(stmt.value, operations, env)

        if isinstance(target, ast.Name):
            if target.id in self.param_names:
                output = self._named_value(target.id)

                if output not in self.outputs:
                    self.outputs.append(output)

                operations.append(
                    SSAOperationIR(
                        "mem.store",
                        operands=(value.name, output.name),
                        attrs={"target": target.id},
                    )
                )
            else:
                env[target.id] = value
            return

        if isinstance(target, ast.Subscript):
            destination = self._lower_tensor_ref(target.value, operations, env)

            if destination not in self.outputs:
                self.outputs.append(destination)

            index_values = tuple(
                value.name
                for value in self._lower_subscript_values(target.slice, operations, env)
            )
            operations.append(
                SSAOperationIR(
                    "mem.store",
                    operands=(value.name, destination.name),
                    attrs={
                        "subscript": _unparse(target.slice),
                        "indices": index_values,
                    },
                )
            )

            return

        raise SSALoweringError(f"Unsupported assignment target: {ast.dump(target)}.")

    def _lower_annassign(
        self,
        stmt: ast.AnnAssign,
        operations: list[SSAOperationIR],
        env: dict[str, SSAValueIR],
    ) -> None:
        if stmt.value is None:
            return

        self._lower_assign(
            ast.Assign(targets=[stmt.target], value=stmt.value),
            operations,
            env,
        )

    def _lower_augassign(
        self,
        stmt: ast.AugAssign,
        operations: list[SSAOperationIR],
        env: dict[str, SSAValueIR],
    ) -> None:
        if isinstance(stmt.target, ast.Subscript):
            destination = self._lower_tensor_ref(stmt.target.value, operations, env)

            if destination not in self.outputs:
                self.outputs.append(destination)

            index_values = tuple(
                value.name
                for value in self._lower_subscript_values(
                    stmt.target.slice, operations, env
                )
            )
            lhs = self._emit(
                operations,
                "tensor.extract" if index_values else "tensor.view",
                operands=(destination.name, *index_values),
                attrs={"subscript": _unparse(stmt.target.slice)},
                result_type=destination.type,
            )
            rhs = self._lower_expr(stmt.value, operations, env)
            result = self._emit(
                operations,
                f"arith.{_binop_name(stmt.op)}",
                operands=(lhs.name, rhs.name),
                result_type=lhs.type,
                attrs={
                    "python": f"{_unparse(stmt.target)} {_augop_symbol(stmt.op)}= ..."
                },
            )
            operations.append(
                SSAOperationIR(
                    "mem.store",
                    operands=(result.name, destination.name),
                    attrs={
                        "subscript": _unparse(stmt.target.slice),
                        "indices": index_values,
                    },
                )
            )

            return

        if not isinstance(stmt.target, ast.Name):
            raise SSALoweringError(
                "Only name and tensor subscript AugAssign targets are supported."
            )

        lhs = env.get(stmt.target.id)

        if lhs is None:
            raise SSALoweringError(f"Unknown AugAssign target {stmt.target.id!r}.")

        rhs = self._lower_expr(stmt.value, operations, env)
        result = self._emit(
            operations,
            f"arith.{_binop_name(stmt.op)}",
            operands=(lhs.name, rhs.name),
            result_type=lhs.type,
            attrs={"python": f"{stmt.target.id} {_augop_symbol(stmt.op)}= ..."},
        )
        env[stmt.target.id] = result

    def _lower_for(
        self,
        stmt: ast.For,
        operations: list[SSAOperationIR],
        env: dict[str, SSAValueIR],
    ) -> None:
        if not isinstance(stmt.target, ast.Name):
            raise SSALoweringError("Only simple induction variables are supported.")

        lower_bound, upper_bound, step = self._range_bounds(stmt.iter, operations, env)
        assigned = _assigned_names(stmt.body)
        carried = tuple(name for name in assigned if name in env)

        induction = SSAValueIR(f"%{stmt.target.id}", SSATypeIR("index"))
        block_args = [induction]
        loop_env = dict(env)
        loop_env[stmt.target.id] = induction
        iter_arg_attrs = []

        for name in carried:
            current = env[name]
            arg = SSAValueIR(f"%{name}_iter", current.type)
            block_args.append(arg)
            loop_env[name] = arg
            iter_arg_attrs.append(
                {"name": name, "initial": current.name, "block_arg": arg.name}
            )

        loop_operations: list[SSAOperationIR] = []
        self._lower_statements(stmt.body, loop_operations, loop_env)
        yield_values = tuple(loop_env[name].name for name in carried)
        loop_operations.append(SSAOperationIR("scf.yield", operands=yield_values))

        results = tuple(self._temp(env[name].type, hint=name) for name in carried)
        operations.append(
            SSAOperationIR(
                "scf.for",
                operands=(
                    lower_bound.name,
                    upper_bound.name,
                    step.name,
                    *(env[name].name for name in carried),
                ),
                results=results,
                attrs={
                    "induction": induction.name,
                    "iter_args": tuple(iter_arg_attrs),
                    "python_target": stmt.target.id,
                },
                regions=(
                    SSABlockIR(
                        name="loop",
                        args=tuple(block_args),
                        operations=tuple(loop_operations),
                    ),
                ),
            )
        )

        for name, result in zip(carried, results):
            env[name] = result

    def _lower_if(
        self,
        stmt: ast.If,
        operations: list[SSAOperationIR],
        env: dict[str, SSAValueIR],
    ) -> None:
        condition = self._ensure_bool_condition(
            self._lower_expr(stmt.test, operations, env), env
        )
        assigned = tuple(
            name for name in _assigned_names(stmt.body + stmt.orelse) if name in env
        )

        if not assigned:
            body_ops: list[SSAOperationIR] = []
            self._lower_statements(stmt.body, body_ops, dict(env))
            regions = [SSABlockIR(name="then", operations=tuple(body_ops))]

            if stmt.orelse:
                else_ops: list[SSAOperationIR] = []
                self._lower_statements(stmt.orelse, else_ops, dict(env))
                regions.append(SSABlockIR(name="else", operations=tuple(else_ops)))

            operations.append(
                SSAOperationIR(
                    "scf.if",
                    operands=(condition.name,),
                    attrs={"has_results": False},
                    regions=tuple(regions),
                )
            )

            return

        then_env = dict(env)
        then_ops: list[SSAOperationIR] = []
        self._lower_statements(stmt.body, then_ops, then_env)
        then_ops.append(
            SSAOperationIR(
                "scf.yield", operands=tuple(then_env[name].name for name in assigned)
            )
        )

        else_env = dict(env)
        else_ops: list[SSAOperationIR] = []

        if stmt.orelse:
            self._lower_statements(stmt.orelse, else_ops, else_env)

        else_ops.append(
            SSAOperationIR(
                "scf.yield", operands=tuple(else_env[name].name for name in assigned)
            )
        )

        results = tuple(self._temp(env[name].type, hint=name) for name in assigned)
        operations.append(
            SSAOperationIR(
                "scf.if",
                operands=(condition.name,),
                results=results,
                attrs={"assigned": assigned},
                regions=(
                    SSABlockIR(name="then", operations=tuple(then_ops)),
                    SSABlockIR(name="else", operations=tuple(else_ops)),
                ),
            )
        )

        for name, result in zip(assigned, results):
            env[name] = result

    def _range_bounds(
        self,
        node: ast.AST,
        operations: list[SSAOperationIR],
        env: dict[str, SSAValueIR],
    ) -> tuple[SSAValueIR, SSAValueIR, SSAValueIR]:
        if not isinstance(node, ast.Call) or _call_leaf_name(node.func) != "range":
            raise SSALoweringError("Only for ... in range(...) loops are supported.")

        args = node.args

        if len(args) == 1:
            lower = self._constant(operations, 0)
            upper = self._lower_expr(args[0], operations, env)
            step = self._constant(operations, 1)

            return lower, upper, step

        if len(args) == 2:
            lower = self._lower_expr(args[0], operations, env)
            upper = self._lower_expr(args[1], operations, env)
            step = self._constant(operations, 1)

            return lower, upper, step

        if len(args) == 3:
            return tuple(self._lower_expr(arg, operations, env) for arg in args)  # type: ignore[return-value]

        raise SSALoweringError(
            "Calls to `range()` with more than three arguments are unsupported."
        )

    def _lower_expr(
        self,
        node: ast.AST,
        operations: list[SSAOperationIR],
        env: dict[str, SSAValueIR],
    ) -> SSAValueIR:
        if isinstance(node, ast.Constant):
            return self._constant(operations, node.value)

        if isinstance(node, ast.Name):
            return env.get(node.id) or self._named_value(node.id)

        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
                value = node.operand.value

                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return self._constant(operations, -value)

            operand = self._lower_expr(node.operand, operations, env)

            return self._emit(
                operations,
                f"arith.{_unaryop_name(node.op)}",
                operands=(operand.name,),
                result_type=operand.type,
            )

        if isinstance(node, ast.BinOp):
            lhs = self._lower_expr(node.left, operations, env)
            rhs = self._lower_expr(node.right, operations, env)
            opcode = (
                "linalg.matmul"
                if isinstance(node.op, ast.MatMult)
                else f"arith.{_binop_name(node.op)}"
            )

            return self._emit(
                operations,
                opcode,
                operands=(lhs.name, rhs.name),
                result_type=_common_type(lhs, rhs),
            )

        if isinstance(node, ast.BoolOp):
            values = [self._lower_expr(value, operations, env) for value in node.values]

            if not values:
                raise SSALoweringError("Empty BoolOp is unsupported.")

            result = values[0]

            for rhs in values[1:]:
                result = self._emit(
                    operations,
                    f"arith.{_boolop_name(node.op)}",
                    operands=(result.name, rhs.name),
                    result_type=SSATypeIR("tensor", dtype="bool"),
                )
            return result

        if isinstance(node, ast.Compare):
            lhs = self._lower_expr(node.left, operations, env)
            comparisons: list[SSAValueIR] = []

            for operator, comparator in zip(node.ops, node.comparators):
                rhs = self._lower_expr(comparator, operations, env)
                comparisons.append(
                    self._emit(
                        operations,
                        f"cmp.{_cmpop_name(operator)}",
                        operands=(lhs.name, rhs.name),
                        result_type=_bool_type(lhs, rhs),
                    )
                )
                lhs = rhs

            if not comparisons:
                raise SSALoweringError("Empty comparison is unsupported.")

            result = comparisons[0]

            for rhs in comparisons[1:]:
                result = self._emit(
                    operations,
                    "arith.and",
                    operands=(result.name, rhs.name),
                    result_type=_bool_type(result, rhs),
                )
            return result

        if isinstance(node, ast.IfExp):
            condition = self._lower_expr(node.test, operations, env)
            body = self._lower_expr(node.body, operations, env)
            orelse = self._lower_expr(node.orelse, operations, env)

            return self._emit(
                operations,
                "select.where",
                operands=(condition.name, body.name, orelse.name),
                result_type=body.type,
            )

        if isinstance(node, ast.Subscript):
            shape_dim = self._lower_shape_dim(node, operations, env)

            if shape_dim is not None:
                return shape_dim

            base = self._lower_expr(node.value, operations, env)
            index_values = tuple(
                value.name
                for value in self._lower_subscript_values(node.slice, operations, env)
            )

            return self._emit(
                operations,
                "tensor.extract" if index_values else "tensor.view",
                operands=(base.name, *index_values),
                attrs={"subscript": _unparse(node.slice)},
                result_type=_subscript_type(base.type, node.slice),
            )

        if isinstance(node, ast.Attribute):
            if node.attr == "T":
                value = self._lower_expr(node.value, operations, env)

                return self._emit(
                    operations,
                    "linalg.transpose",
                    operands=(value.name,),
                    attrs={"python": _unparse(node)},
                    result_type=_transpose_type(value.type),
                )
            return self._emit(
                operations,
                "symbol.attr",
                attrs={"expr": _unparse(node)},
                result_type=SSATypeIR("symbol"),
            )

        if isinstance(node, ast.Call):
            return self._lower_call(node, operations, env)

        if isinstance(node, (ast.Tuple, ast.List)):
            items = tuple(self._lower_expr(item, operations, env) for item in node.elts)

            return self._emit(
                operations,
                "tuple.construct",
                operands=tuple(item.name for item in items),
                attrs={"items": tuple(_unparse(item) for item in node.elts)},
                result_type=SSATypeIR("tuple"),
            )

        raise SSALoweringError(f"Unsupported expression: {ast.dump(node)}.")

    def _lower_call(
        self,
        node: ast.Call,
        operations: list[SSAOperationIR],
        env: dict[str, SSAValueIR],
    ) -> SSAValueIR:
        if _call_leaf_name(node.func) == "float" and len(node.args) == 1:
            literal = _literal_value(node.args[0])

            if literal == "-inf":
                return self._constant(operations, float("-inf"))

            if literal == "inf":
                return self._constant(operations, float("inf"))

        if isinstance(node.func, ast.Attribute) and not _is_namespace_ref(
            node.func.value
        ):
            method = node.func.attr
            receiver = self._lower_tensor_ref(node.func.value, operations, env)

            if method == "to":
                return self._emit(
                    operations,
                    "tensor.cast",
                    operands=(receiver.name,),
                    attrs={"dtype": _unparse(node.args[0]) if node.args else None},
                    result_type=receiver.type,
                )

            if method == "offsets":
                dim = _literal_value(node.args[0]) if node.args else None

                return self._emit(
                    operations,
                    "index.offset",
                    operands=(receiver.name,),
                    attrs={"dim": dim},
                    result_type=_offset_type(receiver.type, dim),
                )

            if method == "stride":
                dim = _literal_value(node.args[0]) if node.args else 0

                return self._emit(
                    operations,
                    "tensor.stride",
                    operands=(receiver.name,),
                    attrs={"dim": dim},
                    result_type=SSATypeIR("index"),
                )

            if method == "data_ptr":
                return self._emit(
                    operations,
                    "mem.data_ptr",
                    operands=(receiver.name,),
                    result_type=SSATypeIR("pointer", dtype=receiver.type.dtype),
                )

            if method in {"sum", "max", "min"}:
                axis = _axis_from_call(node, positional_index=0)
                opcode = f"reduce.{method}"

                return self._emit(
                    operations,
                    opcode,
                    operands=(receiver.name,),
                    attrs={"axis": axis},
                    result_type=_reduce_type(receiver.type, axis),
                )

            if method in _SUPPORTED_MATH_CALLS:
                args = tuple(
                    self._lower_expr(arg, operations, env) for arg in node.args
                )

                return self._emit(
                    operations,
                    f"math.{method}",
                    operands=(receiver.name, *(arg.name for arg in args)),
                    result_type=receiver.type,
                )

        name = _call_leaf_name(node.func)

        if name in {"zeros", "empty"}:
            shape = (
                _shape_tuple_from_ast(node.args[0], operations, env, self)
                if node.args
                else ()
            )

            return self._emit(
                operations,
                "tensor.zeros",
                attrs={
                    "shape": _unparse(node.args[0]) if node.args else None,
                    "dtype": _keyword_text(node, "dtype"),
                },
                result_type=SSATypeIR(
                    "tensor", dtype=_keyword_text(node, "dtype"), shape=shape
                ),
            )

        if name == "full":
            operands = tuple(
                self._lower_expr(arg, operations, env) for arg in node.args[1:]
            )
            shape = (
                _shape_tuple_from_ast(node.args[0], operations, env, self)
                if node.args
                else ()
            )

            return self._emit(
                operations,
                "tensor.full",
                operands=tuple(value.name for value in operands),
                attrs={
                    "shape": _unparse(node.args[0]) if node.args else None,
                    "value": _literal_value(node.args[1])
                    if len(node.args) > 1
                    else None,
                    "dtype": _keyword_text(node, "dtype"),
                },
                result_type=SSATypeIR(
                    "tensor", dtype=_keyword_text(node, "dtype"), shape=shape
                ),
            )

        operands = tuple(self._lower_expr(arg, operations, env) for arg in node.args)

        if name == "fill":
            if len(operands) == 1:
                return operands[0]

            if len(operands) >= 2:
                destination, value = operands[0], operands[1]
                self._store_intrinsic_result(operations, destination, value, name)

                return destination

        if name == "copy" and len(operands) >= 2:
            source, destination = operands[0], operands[1]
            self._store_intrinsic_result(operations, destination, source, name)

            return destination

        if name.startswith("reduce_") and operands:
            operator = name[len("reduce_") :]

            if operator in {"sum", "max", "min"}:
                axis = _axis_from_call(node, positional_index=2)
                reduced = self._emit(
                    operations,
                    f"reduce.{operator}",
                    operands=(operands[0].name,),
                    attrs={"axis": axis},
                    result_type=_reduce_type(operands[0].type, axis),
                )

                if len(operands) > 1:
                    self._store_intrinsic_result(operations, operands[1], reduced, name)

                    return operands[1]
                return reduced

        if name in {"matmul", "dot"} and len(operands) >= 3:
            result = self._emit(
                operations,
                "linalg.matmul" if name == "matmul" else "linalg.dot",
                operands=(operands[0].name, operands[1].name),
                result_type=operands[2].type,
            )
            self._store_intrinsic_result(operations, operands[2], result, name)

            return operands[2]

        if name == "atomic_add":
            return self._emit(
                operations,
                "mem.atomic_add",
                operands=tuple(value.name for value in operands),
                result_type=SSATypeIR(
                    "scalar",
                    dtype=operands[1].type.dtype if len(operands) > 1 else "float32",
                ),
            )

        if name in {"dot", "matmul"}:
            result_type = (
                _matmul_type(operands[0].type, operands[1].type)
                if len(operands) >= 2
                else SSATypeIR("tensor")
            )

            return self._emit(
                operations,
                "linalg.dot" if name == "dot" else "linalg.matmul",
                operands=tuple(value.name for value in operands),
                result_type=result_type,
            )

        if name in {"trans", "transpose"} and len(operands) >= 2:
            result = self._emit(
                operations,
                "linalg.transpose",
                operands=(operands[0].name,),
                result_type=operands[1].type,
            )
            self._store_intrinsic_result(operations, operands[1], result, name)

            return operands[1]

        if name in {"trans", "transpose"}:
            return self._emit(
                operations,
                "linalg.transpose",
                operands=tuple(value.name for value in operands),
                result_type=_transpose_type(operands[0].type)
                if operands
                else SSATypeIR("tensor"),
            )

        if name == "where":
            return self._emit(
                operations,
                "select.where",
                operands=tuple(value.name for value in operands),
                result_type=operands[1].type
                if len(operands) > 1
                else SSATypeIR("tensor"),
            )

        if name in {"sum", "max", "min"}:
            axis = _axis_from_call(node, positional_index=1)

            return self._emit(
                operations,
                f"reduce.{name}",
                operands=(operands[0].name,) if operands else (),
                attrs={"axis": axis},
                result_type=_reduce_type(operands[0].type, axis)
                if operands
                else SSATypeIR("tensor"),
            )

        if name in {"maximum", "minimum"}:
            return self._emit(
                operations,
                f"arith.{name}",
                operands=tuple(value.name for value in operands),
                result_type=operands[0].type if operands else SSATypeIR("tensor"),
            )

        if name in _SUPPORTED_MATH_CALLS:
            return self._emit(
                operations,
                f"math.{name}",
                operands=tuple(value.name for value in operands),
                result_type=operands[0].type if operands else SSATypeIR("tensor"),
            )

        return self._emit(
            operations,
            f"call.{name}",
            operands=tuple(value.name for value in operands),
            attrs={"callee": _unparse(node.func)},
            result_type=operands[0].type if operands else SSATypeIR("tensor"),
        )

    def _store_intrinsic_result(
        self,
        operations: list[SSAOperationIR],
        destination: SSAValueIR,
        value: SSAValueIR,
        intrinsic: str,
    ) -> None:
        if destination not in self.outputs:
            self.outputs.append(destination)

        operations.append(
            SSAOperationIR(
                "mem.store",
                operands=(value.name, destination.name),
                attrs={"target": destination.name, "intrinsic": intrinsic},
            )
        )

    def _lower_tensor_ref(
        self,
        node: ast.AST,
        operations: list[SSAOperationIR],
        env: dict[str, SSAValueIR],
    ) -> SSAValueIR:
        shape_dim = self._lower_shape_dim(node, operations, env)

        if shape_dim is not None:
            return shape_dim

        if isinstance(node, ast.Name):
            return env.get(node.id) or self._named_value(node.id)

        if isinstance(node, ast.Subscript):
            return self._lower_expr(node, operations, env)

        if isinstance(node, ast.Attribute):
            if node.attr == "source":
                return self._lower_tensor_ref(node.value, operations, env)
            return self._emit(
                operations,
                "symbol.attr",
                attrs={"expr": _unparse(node)},
                result_type=SSATypeIR("symbol"),
            )
        return self._lower_expr(node, operations, env)

    def _lower_shape_dim(
        self,
        node: ast.AST,
        operations: list[SSAOperationIR],
        env: dict[str, SSAValueIR],
    ) -> SSAValueIR | None:
        if not isinstance(node, ast.Subscript):
            return None

        value = node.value

        if not isinstance(value, ast.Attribute) or value.attr != "shape":
            return None

        tensor_node = value.value
        source = False

        if isinstance(tensor_node, ast.Attribute) and tensor_node.attr == "source":
            tensor_node = tensor_node.value
            source = True

        tensor = self._lower_tensor_ref(tensor_node, operations, env)

        return self._emit(
            operations,
            "shape.dim",
            operands=(tensor.name,),
            attrs={"dim": _literal_value(node.slice), "source": source},
            result_type=SSATypeIR("index"),
        )

    def _lower_subscript_values(
        self,
        node: ast.AST,
        operations: list[SSAOperationIR],
        env: dict[str, SSAValueIR],
    ) -> tuple[SSAValueIR, ...]:
        if isinstance(node, ast.Tuple):
            values = []

            for elt in node.elts:
                if isinstance(elt, ast.Slice) or (
                    isinstance(elt, ast.Constant) and elt.value is None
                ):
                    continue

                values.append(self._lower_expr(elt, operations, env))
            return tuple(values)

        if isinstance(node, ast.Slice):
            return ()

        if isinstance(node, ast.Constant) and node.value is None:
            return ()
        return (self._lower_expr(node, operations, env),)

    def _constant(self, operations: list[SSAOperationIR], value: Any) -> SSAValueIR:
        if isinstance(value, bool):
            dtype = "bool"
        elif isinstance(value, int):
            dtype = "int64"
        elif isinstance(value, float):
            dtype = "float32"
        elif value is None:
            dtype = "none"
        else:
            dtype = "symbol"

        attr_value: Any = value

        if isinstance(value, float) and not math.isfinite(value):
            attr_value = "-inf" if value < 0 else "inf"
        return self._emit(
            operations,
            "arith.constant",
            attrs={"value": attr_value},
            result_type=SSATypeIR("scalar", dtype=dtype),
        )

    def _emit(
        self,
        operations: list[SSAOperationIR],
        opcode: str,
        *,
        operands: tuple[str, ...] = (),
        attrs: Mapping[str, Any] | None = None,
        result_type: SSATypeIR | None = None,
    ) -> SSAValueIR:
        result = self._temp(result_type or SSATypeIR("tensor"))
        operations.append(
            SSAOperationIR(
                opcode,
                operands=operands,
                results=(result,),
                attrs=dict(attrs or {}),
            )
        )

        return result

    def _temp(self, type_: SSATypeIR, *, hint: str | None = None) -> SSAValueIR:
        name = f"%{self.temp_index}" if hint is None else f"%{hint}_{self.temp_index}"
        self.temp_index += 1
        value = SSAValueIR(name, type_)
        self.values[name] = value

        return value

    def _named_value(self, name: str, type_: SSATypeIR | None = None) -> SSAValueIR:
        if name not in self.values:
            self.values[name] = SSAValueIR(
                name, type_ or self.tensor_types.get(name, SSATypeIR("tensor"))
            )
        return self.values[name]

    def _ensure_bool_condition(
        self, value: SSAValueIR, env: dict[str, SSAValueIR]
    ) -> SSAValueIR:
        if value.type.dtype == "bool":
            return value

        if value.type.kind != "scalar" or value.type.dtype not in {None, "symbol"}:
            return value

        replacement = SSAValueIR(
            value.name,
            SSATypeIR(
                "scalar",
                dtype="bool",
                shape=value.type.shape,
                attrs=dict(value.type.attrs),
            ),
        )
        self.values[value.name] = replacement

        for name, current in tuple(env.items()):
            if current.name == value.name:
                env[name] = replacement
        return replacement


def _assigned_names(statements: Iterable[ast.stmt]) -> tuple[str, ...]:
    names: list[str] = []

    class Visitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> None:
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id not in names:
                    names.append(target.id)

            self.generic_visit(node.value)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if isinstance(node.target, ast.Name) and node.target.id not in names:
                names.append(node.target.id)

            if node.value is not None:
                self.visit(node.value)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:
            if isinstance(node.target, ast.Name) and node.target.id not in names:
                names.append(node.target.id)

            self.generic_visit(node.value)

        def visit_For(self, node: ast.For) -> None:
            if isinstance(node.target, ast.Name) and node.target.id not in names:
                names.append(node.target.id)

            for stmt in node.body + node.orelse:
                self.visit(stmt)

    visitor = Visitor()

    for stmt in statements:
        visitor.visit(stmt)
    return tuple(names)


def _count_operations(operations: Iterable[SSAOperationIR]) -> int:
    total = 0

    for op in operations:
        total += 1

        for region in op.regions:
            total += _count_operations(region.operations)
    return total


def _render_block(block: SSABlockIR, lines: list[str], *, indent: int) -> None:
    prefix = " " * indent
    args = ""

    if block.args:
        args = (
            "("
            + ", ".join(f"{arg.name}: {_format_type(arg.type)}" for arg in block.args)
            + ")"
        )

    lines.append(f"{prefix}^{block.name}{args}:")

    for operation in block.operations:
        _render_operation(operation, lines, indent=indent + 2)


def _render_operation(
    operation: SSAOperationIR, lines: list[str], *, indent: int
) -> None:
    prefix = " " * indent
    results = ", ".join(result.name for result in operation.results)
    operands = ", ".join(operation.operands)
    lhs = f"{results} = " if results else ""
    attrs = _format_attrs(operation.attrs)
    suffix = f" {attrs}" if attrs else ""
    operand_text = f" {operands}" if operands else ""
    lines.append(f"{prefix}{lhs}{operation.opcode}{operand_text}{suffix}".rstrip())

    for region in operation.regions:
        _render_block(region, lines, indent=indent + 2)


def _format_type(type_: SSATypeIR) -> str:
    shape = ""

    if type_.shape:
        shape = "<" + "x".join(type_.shape) + ">"

    dtype = f"x{type_.dtype}" if type_.dtype else ""

    return f"{type_.kind}{shape}{dtype}"


def _format_attrs(attrs: Mapping[str, Any]) -> str:
    cleaned = {
        key: value for key, value in attrs.items() if value is not None and value != ()
    }

    if not cleaned:
        return ""
    return (
        "{"
        + ", ".join(f"{key}={_format_attr(value)}" for key, value in cleaned.items())
        + "}"
    )


def _format_attr(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)

    if isinstance(value, tuple):
        return "(" + ", ".join(_format_attr(item) for item in value) + ")"

    if isinstance(value, list):
        return "[" + ", ".join(_format_attr(item) for item in value) + "]"

    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(f"{key}: {_format_attr(item)}" for key, item in value.items())
            + "}"
        )
    return repr(value)


def _shape_tuple_from_ast(
    node: ast.AST,
    operations: list[SSAOperationIR],
    env: dict[str, SSAValueIR],
    builder: _ApplicationSSABuilder,
) -> tuple[str, ...]:
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(
            _shape_text_from_ast(item, operations, env, builder) for item in node.elts
        )

    text = _shape_text_from_ast(node, operations, env, builder)

    return () if text in {"", "None"} else (text,)


def _shape_text_from_ast(
    node: ast.AST,
    operations: list[SSAOperationIR],
    env: dict[str, SSAValueIR],
    builder: _ApplicationSSABuilder,
) -> str:
    resolved = _shape_dim_text_from_ast(node, env, builder)

    if resolved is not None:
        return resolved

    literal = _literal_value(node)

    if isinstance(literal, (int, float)) and not isinstance(literal, bool):
        return str(literal)

    if isinstance(literal, str) and literal != _unparse(node):
        return literal

    value = builder._lower_expr(node, operations, env)

    if value.name.startswith("%"):
        return value.name
    return str(value.name)


def _shape_dim_text_from_ast(
    node: ast.AST,
    env: dict[str, SSAValueIR],
    builder: _ApplicationSSABuilder,
) -> str | None:
    if not isinstance(node, ast.Subscript):
        return None

    value = node.value

    if not isinstance(value, ast.Attribute) or value.attr != "shape":
        return None

    tensor_node = value.value
    source = False

    if isinstance(tensor_node, ast.Attribute) and tensor_node.attr == "source":
        tensor_node = tensor_node.value
        source = True

    tensor = _value_for_shape_node(tensor_node, env, builder)

    if tensor is None:
        return None

    dim = _literal_value(node.slice)

    return _shape_dim_from_type(tensor.type, dim, source=source)


def _value_for_shape_node(
    node: ast.AST,
    env: dict[str, SSAValueIR],
    builder: _ApplicationSSABuilder,
) -> SSAValueIR | None:
    if isinstance(node, ast.Name):
        return env.get(node.id) or builder.values.get(node.id)

    if isinstance(node, ast.Subscript):
        base = _value_for_shape_node(node.value, env, builder)

        if base is None:
            return None
        return SSAValueIR("<shape-proxy>", _subscript_type(base.type, node.slice))
    return None


def _shape_dim_from_type(
    type_: SSATypeIR, dim: Any, *, source: bool = False
) -> str | None:
    if source:
        shape = tuple(str(item) for item in type_.attrs.get("source_shape", ()))
    else:
        shape = tuple(str(item) for item in type_.shape)

    if not shape:
        return None

    index = int(dim or 0)

    if index < 0:
        index += len(shape)

    if index < 0 or index >= len(shape):
        return None
    return shape[index]


def _subscript_type(type_: SSATypeIR, slice_node: ast.AST) -> SSATypeIR:
    if type_.kind != "tensor":
        return type_

    elements = (
        tuple(slice_node.elts) if isinstance(slice_node, ast.Tuple) else (slice_node,)
    )
    shape = tuple(str(dim) for dim in type_.shape)
    result_shape: list[str] = []
    position = 0
    consumed = 0

    for element in elements:
        if isinstance(element, ast.Constant) and element.value is None:
            result_shape.append("1")
            continue

        if isinstance(element, ast.Slice):
            if position < len(shape):
                result_shape.append(shape[position])
                position += 1

            continue

        if position < len(shape):
            position += 1
            consumed += 1

    result_shape.extend(shape[position:])
    attrs = dict(type_.attrs)

    if not result_shape:
        next_shape = _next_dtype_shape(type_)

        if next_shape is not None:
            level = int(attrs.get("dtype_level", 0)) + 1
            attrs["dtype_level"] = level

            return SSATypeIR("tensor", dtype=type_.dtype, shape=next_shape, attrs=attrs)
        return SSATypeIR("scalar", dtype=type_.dtype, attrs=attrs)

    if consumed:
        attrs["partial_indices"] = int(attrs.get("partial_indices", 0)) + consumed
    return SSATypeIR(
        "tensor", dtype=type_.dtype, shape=tuple(result_shape), attrs=attrs
    )


def _next_dtype_shape(type_: SSATypeIR) -> tuple[str, ...] | None:
    shapes = tuple(
        tuple(str(dim) for dim in shape)
        for shape in type_.attrs.get("dtype_shapes", ())
    )
    level = int(type_.attrs.get("dtype_level", 0))

    if level + 1 >= len(shapes):
        return None
    return shapes[level + 1]


def _reduce_type(type_: SSATypeIR, axis: Any) -> SSATypeIR:
    if type_.kind != "tensor":
        return type_

    shape = tuple(str(dim) for dim in type_.shape)

    if axis is None:
        return SSATypeIR("scalar", dtype=type_.dtype, attrs=dict(type_.attrs))

    index = int(axis)

    if index < 0:
        index += len(shape)

    if index < 0 or index >= len(shape):
        return type_

    result_shape = shape[:index] + shape[index + 1 :]

    if not result_shape:
        return SSATypeIR("scalar", dtype=type_.dtype, attrs=dict(type_.attrs))
    return SSATypeIR(
        "tensor", dtype=type_.dtype, shape=result_shape, attrs=dict(type_.attrs)
    )


def _offset_type(type_: SSATypeIR, dim: Any) -> SSATypeIR:
    if type_.kind != "tensor":
        return SSATypeIR("scalar", dtype="index")

    shape = tuple(str(item) for item in type_.shape)
    dtype_target_dims = tuple(
        tuple(None if item is None else str(item) for item in dims)
        for dims in type_.attrs.get("dtype_target_dims", ())
    )
    level = int(type_.attrs.get("dtype_level", 0))
    target_dims = dtype_target_dims[level] if level < len(dtype_target_dims) else ()

    if not target_dims:
        return SSATypeIR("tensor", dtype="index", shape=shape)

    source_ndim = int(type_.attrs.get("source_ndim", len(target_dims)))
    source_dim = int(dim or 0)

    if source_dim < 0:
        source_dim += source_ndim

    kept = tuple(
        axis
        for axis, target_dim in zip(shape, target_dims)
        if target_dim is not None and int(target_dim) == source_dim
    )

    if not kept:
        return SSATypeIR("scalar", dtype="index")
    return SSATypeIR("tensor", dtype="index", shape=kept)


def _matmul_type(lhs: SSATypeIR, rhs: SSATypeIR) -> SSATypeIR:
    lhs_shape = tuple(str(dim) for dim in lhs.shape)
    rhs_shape = tuple(str(dim) for dim in rhs.shape)
    dtype = lhs.dtype or rhs.dtype
    attrs = dict(lhs.attrs)

    if len(lhs_shape) >= 2 and len(rhs_shape) >= 2:
        return SSATypeIR(
            "tensor", dtype=dtype, shape=(lhs_shape[-2], rhs_shape[-1]), attrs=attrs
        )

    if len(lhs_shape) >= 2 and len(rhs_shape) == 1:
        return SSATypeIR("tensor", dtype=dtype, shape=(lhs_shape[-2],), attrs=attrs)

    if len(lhs_shape) == 1 and len(rhs_shape) >= 2:
        return SSATypeIR("tensor", dtype=dtype, shape=(rhs_shape[-1],), attrs=attrs)

    if len(lhs_shape) == 1 and len(rhs_shape) == 1:
        return SSATypeIR("scalar", dtype=dtype, attrs=attrs)
    return _broadcast_type(lhs, rhs)


def _common_type(lhs: SSAValueIR, rhs: SSAValueIR) -> SSATypeIR:
    return _broadcast_type(lhs.type, rhs.type)


def _broadcast_type(lhs: SSATypeIR, rhs: SSATypeIR) -> SSATypeIR:
    if lhs.kind != "tensor" and rhs.kind != "tensor":
        return lhs

    if lhs.kind == "tensor" and rhs.kind != "tensor":
        return lhs

    if rhs.kind == "tensor" and lhs.kind != "tensor":
        return rhs

    lhs_shape = tuple(str(dim) for dim in lhs.shape)
    rhs_shape = tuple(str(dim) for dim in rhs.shape)
    result: list[str] = []

    for lhs_dim, rhs_dim in zip(reversed(lhs_shape), reversed(rhs_shape)):
        if lhs_dim == rhs_dim or rhs_dim == "1":
            result.append(lhs_dim)
        elif lhs_dim == "1":
            result.append(rhs_dim)
        else:
            result.append(lhs_dim)

    longer = lhs_shape if len(lhs_shape) > len(rhs_shape) else rhs_shape
    prefix = longer[: abs(len(lhs_shape) - len(rhs_shape))]
    shape = tuple(prefix) + tuple(reversed(result))
    dtype = lhs.dtype or rhs.dtype
    attrs = dict(lhs.attrs if lhs.kind == "tensor" else rhs.attrs)

    return SSATypeIR("tensor", dtype=dtype, shape=shape, attrs=attrs)


def _bool_type(lhs: SSAValueIR, rhs: SSAValueIR | None = None) -> SSATypeIR:
    if rhs is not None and rhs.type.kind == "tensor":
        shape = _broadcast_type(lhs.type, rhs.type).shape

        return SSATypeIR("tensor", dtype="bool", shape=shape)

    if lhs.type.kind == "tensor":
        return SSATypeIR("tensor", dtype="bool", shape=lhs.type.shape)
    return SSATypeIR("scalar", dtype="bool")


_SUPPORTED_MATH_CALLS = {
    "abs",
    "acos",
    "asin",
    "atan",
    "atan2",
    "ceil",
    "cos",
    "cosh",
    "erf",
    "exp",
    "exp2",
    "expm1",
    "floor",
    "log",
    "log1p",
    "log2",
    "log10",
    "pow",
    "rsqrt",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
}


def _transpose_type(type_: SSATypeIR) -> SSATypeIR:
    if type_.kind != "tensor" or len(type_.shape) < 2:
        return type_
    return SSATypeIR(
        type_.kind,
        dtype=type_.dtype,
        shape=tuple(reversed(type_.shape)),
        attrs=dict(type_.attrs),
    )


def _axis_from_call(node: ast.Call, *, positional_index: int) -> Any:
    if len(node.args) > positional_index:
        return _literal_value(node.args[positional_index])

    for keyword in node.keywords:
        if keyword.arg in {"axis", "dim"}:
            return _literal_value(keyword.value)
    return None


def _keyword_text(node: ast.Call, name: str) -> str | None:
    for keyword in node.keywords:
        if keyword.arg == name:
            return _unparse(keyword.value)
    return None


def _literal_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant):
        value = node.operand.value

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None

        if isinstance(node.op, ast.USub):
            return -value

        if isinstance(node.op, ast.UAdd):
            return value
    return _unparse(node)


def _call_leaf_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        return node.attr
    return _unparse(node)


def _is_namespace_ref(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id in {
            "F",
            "libdevice",
            "math",
            "ninetoothed",
            "ntl",
            "tl",
            "torch",
            "triton",
        }

    if isinstance(node, ast.Attribute):
        return _is_namespace_ref(node.value)
    return False


def _unparse(node: ast.AST) -> str:
    return ast.unparse(node)


def _binop_name(node: ast.operator) -> str:
    mapping = {
        ast.Add: "add",
        ast.Sub: "sub",
        ast.Mult: "mul",
        ast.Div: "div",
        ast.FloorDiv: "floordiv",
        ast.Mod: "mod",
        ast.Pow: "pow",
        ast.LShift: "bitwise_left_shift",
        ast.RShift: "bitwise_right_shift",
        ast.BitAnd: "bitwise_and",
        ast.BitOr: "bitwise_or",
        ast.BitXor: "bitwise_xor",
        ast.MatMult: "matmul",
    }

    return mapping[type(node)]


def _augop_symbol(node: ast.operator) -> str:
    mapping = {
        ast.Add: "+",
        ast.Sub: "-",
        ast.Mult: "*",
        ast.Div: "/",
        ast.FloorDiv: "//",
        ast.Mod: "%",
    }

    return mapping.get(type(node), "?")


def _unaryop_name(node: ast.unaryop) -> str:
    mapping = {
        ast.USub: "neg",
        ast.UAdd: "pos",
        ast.Not: "not",
        ast.Invert: "invert",
    }

    return mapping[type(node)]


def _boolop_name(node: ast.boolop) -> str:
    return "and" if isinstance(node, ast.And) else "or"


def _cmpop_name(node: ast.cmpop) -> str:
    mapping = {
        ast.Eq: "eq",
        ast.NotEq: "ne",
        ast.Lt: "lt",
        ast.LtE: "le",
        ast.Gt: "gt",
        ast.GtE: "ge",
        ast.Is: "eq",
        ast.IsNot: "ne",
    }

    return mapping[type(node)]
