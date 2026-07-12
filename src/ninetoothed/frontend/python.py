"""SSA lowering from NineToothed application Python AST.

This module deliberately lowers computation structure instead of recognizing
whole operators.  Complex fused kernels are represented with ``scf`` regions,
tensor operations, reductions, masks, scalar math, and stores rather than with
operator-level SSA opcodes.
"""

import ast
import inspect
import math
import re
import textwrap
from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

from ninetoothed.frontend.errors import LoweringError
from ninetoothed.frontend.types import (
    _binary_type,
    _bool_type,
    _broadcast_type,
    _cast_type,
    _load_type,
    _math_result_type,
    _matmul_type,
    _offset_type,
    _reduce_type,
    _shape_dim_from_type,
    _subscript_type,
    _transpose_type,
)
from ninetoothed.ir import TensorSpec, ssa


def from_application(
    application: Any,
    tensor_irs: tuple[TensorSpec, ...] = (),
    *,
    kind: str | None = None,
    strict: bool = False,
) -> ssa.Program | None:
    """Lower a NineToothed application function to generic SSA.

    The pass is syntax-directed and target-neutral.  It is intentionally not a
    semantic pattern matcher for named operators; it only sees Python AST and
    emits generic operations.
    """
    try:
        source = inspect.getsource(application)
    except OSError:
        return None

    closure = inspect.getclosurevars(application)
    scope = dict(getattr(application, "__globals__", {}) or {})
    scope.update(closure.globals)
    scope.update(closure.nonlocals)

    return from_source(
        source,
        tensor_irs=tensor_irs,
        kind=kind or getattr(application, "__name__", "application"),
        globalns=scope,
        strict=strict,
    )


def from_source(
    source: str,
    tensor_irs: tuple[TensorSpec, ...] = (),
    *,
    kind: str = "application",
    globalns: Mapping[str, Any] | None = None,
    strict: bool = False,
) -> ssa.Program | None:
    tree = ast.parse(textwrap.dedent(source))
    func = next((node for node in tree.body if isinstance(node, ast.FunctionDef)), None)

    if func is None:
        return None

    if globalns is not None:
        func = _InlineHelperCalls(globalns).inline(func)

    builder = _ApplicationSSABuilder(func, tensor_irs, kind, strict=strict)
    builder.lower()
    program = builder.finish()
    ssa.verify_program(program)

    return program


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
        handler = {
            ast.Assign: self._inline_value_statement,
            ast.AnnAssign: self._inline_optional_value_statement,
            ast.AugAssign: self._inline_value_statement,
            ast.Return: self._inline_optional_value_statement,
            ast.Expr: self._inline_expression_statement,
            ast.If: self._inline_if_statement,
            ast.For: self._inline_for_statement,
        }.get(type(stmt))

        return [stmt] if handler is None else handler(stmt)

    def _inline_value_statement(self, stmt) -> list[ast.stmt]:
        value, prefix = self._inline_expr(stmt.value)
        stmt.value = value

        return [*prefix, stmt]

    def _inline_optional_value_statement(self, stmt) -> list[ast.stmt]:
        if stmt.value is None:
            return [stmt]
        return self._inline_value_statement(stmt)

    def _inline_expression_statement(self, stmt: ast.Expr) -> list[ast.stmt]:
        value, prefix = self._inline_expr(stmt.value)
        stmt.value = value

        if isinstance(value, ast.Constant) and value.value is None:
            return prefix
        return [*prefix, stmt]

    def _inline_if_statement(self, stmt: ast.If) -> list[ast.stmt]:
        test, prefix = self._inline_expr(stmt.test)
        stmt.test = test
        stmt.body = self._inline_statements(stmt.body)
        stmt.orelse = self._inline_statements(stmt.orelse)

        return [*prefix, stmt]

    def _inline_for_statement(self, stmt: ast.For) -> list[ast.stmt]:
        iterator, prefix = self._inline_expr(stmt.iter)
        stmt.iter = iterator
        stmt.body = self._inline_statements(stmt.body)
        stmt.orelse = self._inline_statements(stmt.orelse)

        return [*prefix, stmt]

    def _inline_expr(self, expr: ast.AST) -> tuple[ast.AST, list[ast.stmt]]:
        if isinstance(expr, ast.Call):
            return self._inline_call_expr(expr)
        return self._inline_expr_fields(expr)

    def _inline_call_expr(self, expr: ast.Call) -> tuple[ast.AST, list[ast.stmt]]:
        function, prefix = self._inline_expr(expr.func)
        args = []

        for argument in expr.args:
            lowered, argument_prefix = self._inline_expr(argument)
            prefix.extend(argument_prefix)
            args.append(lowered)

        keywords = []

        for keyword in expr.keywords:
            if keyword.arg is None:
                return expr, prefix

            value, keyword_prefix = self._inline_expr(keyword.value)
            prefix.extend(keyword_prefix)
            keywords.append(ast.keyword(arg=keyword.arg, value=value))

        call = ast.Call(func=function, args=args, keywords=keywords)
        inlined = self._inline_call(call)

        if inlined is None:
            return call, prefix

        value, statements = inlined

        return value, [*prefix, *statements]

    def _inline_expr_fields(self, expr: ast.AST) -> tuple[ast.AST, list[ast.stmt]]:
        prefix: list[ast.stmt] = []

        for field, value in ast.iter_fields(expr):
            if isinstance(value, ast.AST):
                lowered, field_prefix = self._inline_expr(value)
                prefix.extend(field_prefix)
                setattr(expr, field, lowered)
            elif isinstance(value, list):
                items, item_prefix = self._inline_expr_items(value)
                prefix.extend(item_prefix)
                setattr(expr, field, items)
        return expr, prefix

    def _inline_expr_items(self, values: list) -> tuple[list, list[ast.stmt]]:
        items = []
        prefix: list[ast.stmt] = []

        for item in values:
            if not isinstance(item, ast.AST):
                items.append(item)
                continue

            lowered, item_prefix = self._inline_expr(item)
            prefix.extend(item_prefix)
            items.append(lowered)
        return items, prefix

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
        if node.id in self.binding and isinstance(node.ctx, (ast.Load, ast.Store)):
            replacement = deepcopy(self.binding[node.id])

            if isinstance(node.ctx, ast.Store):
                if not isinstance(replacement, ast.Name):
                    return node

                replacement.ctx = ast.Store()

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


class _ApplicationSSABuilder:
    def __init__(
        self,
        func: ast.FunctionDef,
        tensor_irs: tuple[TensorSpec, ...],
        kind: str,
        *,
        strict: bool,
    ):
        self.func = func
        self.kind = kind
        self.strict = strict
        self.param_names = tuple(arg.arg for arg in func.args.args)
        self.tensor_types = {
            tensor.name: ssa.Type(
                kind="tensor" if tensor.ndim != 0 else "scalar",
                shape=tuple(
                    str(dim)
                    for dim in tensor.attrs.get("application_shape", tensor.shape)
                ),
                dtype=tensor.dtype,
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
        self.values: dict[str, ssa.Value] = {}
        self.outputs: list[ssa.Value] = []
        self.operations: list[ssa.Operation] = []
        self.env: dict[str, ssa.Value] = {}
        self.temp_index = 0
        self.symbol_names = {
            name
            for tensor in tensor_irs
            for text in (*tensor.shape, *tensor.attrs.get("source_shape", ()))
            for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(text))
        }

        for name in self.param_names:
            value = self._named_value(name, self.tensor_types.get(name))
            self.env[name] = value

    def lower(self) -> None:
        self._lower_statements(self.func.body, self.operations, self.env)

    def finish(self) -> ssa.Program:
        metadata = {
            "source": "application_ast",
            "function": self.func.name,
            "coarse_operator_nodes": False,
            "ssa_operation_count": _count_operations(self.operations),
            "symbols": tuple(sorted(self.symbol_names)),
        }

        return ssa.Program(
            kind=self.kind,
            inputs=tuple(
                self.values[name] for name in self.param_names if name in self.values
            ),
            outputs=tuple(self.outputs),
            blocks=(ssa.Block(operations=tuple(self.operations)),),
            metadata=metadata,
        )

    def _lower_statements(
        self,
        statements: Iterable[ast.stmt],
        operations: list[ssa.Operation],
        env: dict[str, ssa.Value],
    ) -> None:
        handlers = {
            ast.Assign: self._lower_assign,
            ast.AnnAssign: self._lower_annassign,
            ast.AugAssign: self._lower_augassign,
            ast.For: self._lower_for,
            ast.If: self._lower_if,
            ast.Expr: self._lower_expression_statement,
            ast.Return: self._lower_return_statement,
            ast.Pass: self._lower_pass_statement,
        }

        for statement in statements:
            try:
                handler = handlers[type(statement)]
            except KeyError as exc:
                raise LoweringError(
                    f"Unsupported statement: {ast.dump(statement)}."
                ) from exc

            handler(statement, operations, env)

    def _lower_expression_statement(self, statement, operations, env) -> None:
        self._lower_expr(statement.value, operations, env)

    def _lower_return_statement(self, statement, operations, env) -> None:
        if statement.value is not None:
            self._lower_expr(statement.value, operations, env)

    def _lower_pass_statement(self, statement, operations, env) -> None:
        del statement, operations, env

    def _lower_assign(
        self,
        stmt: ast.Assign,
        operations: list[ssa.Operation],
        env: dict[str, ssa.Value],
    ) -> None:
        if len(stmt.targets) != 1:
            raise LoweringError("Only single-target assignments are supported.")

        target = stmt.targets[0]
        value = self._lower_expr(stmt.value, operations, env)

        if isinstance(target, ast.Name):
            if target.id in self.param_names:
                output = self._named_value(target.id)

                if output not in self.outputs:
                    self.outputs.append(output)

                operations.append(
                    ssa.Operation(
                        opcode="mem.store",
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
            target_type = _subscript_type(destination.type, target.slice)
            operations.append(
                ssa.Operation(
                    opcode="mem.store",
                    operands=(value.name, destination.name),
                    attrs={
                        "subscript": _unparse(target.slice),
                        "indices": index_values,
                        "source": isinstance(target.value, ast.Attribute)
                        and target.value.attr == "source",
                        "target_shape": tuple(target_type.shape),
                        "target_dtype_level": int(
                            target_type.attrs.get("dtype_level", 0)
                        ),
                        "base_dtype_level": int(
                            destination.type.attrs.get("dtype_level", 0)
                        ),
                    },
                )
            )

            return

        raise LoweringError(f"Unsupported assignment target: {ast.dump(target)}.")

    def _lower_annassign(
        self,
        stmt: ast.AnnAssign,
        operations: list[ssa.Operation],
        env: dict[str, ssa.Value],
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
        operations: list[ssa.Operation],
        env: dict[str, ssa.Value],
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
            target_type = _subscript_type(destination.type, stmt.target.slice)
            lhs = self._emit(
                operations,
                "tensor.extract" if index_values else "tensor.view",
                operands=(destination.name, *index_values),
                attrs={"subscript": _unparse(stmt.target.slice)},
                result_type=target_type,
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
                ssa.Operation(
                    opcode="mem.store",
                    operands=(result.name, destination.name),
                    attrs={
                        "subscript": _unparse(stmt.target.slice),
                        "indices": index_values,
                        "target_shape": tuple(target_type.shape),
                        "target_dtype_level": int(
                            target_type.attrs.get("dtype_level", 0)
                        ),
                        "base_dtype_level": int(
                            destination.type.attrs.get("dtype_level", 0)
                        ),
                    },
                )
            )

            return

        if not isinstance(stmt.target, ast.Name):
            raise LoweringError(
                "Only name and tensor subscript AugAssign targets are supported."
            )

        lhs = env.get(stmt.target.id)

        if lhs is None:
            raise LoweringError(f"Unknown AugAssign target {stmt.target.id!r}.")

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
        operations: list[ssa.Operation],
        env: dict[str, ssa.Value],
    ) -> None:
        if not isinstance(stmt.target, ast.Name):
            raise LoweringError("Only simple induction variables are supported.")

        lower_bound, upper_bound, step = self._range_bounds(stmt.iter, operations, env)
        assigned = _assigned_names(stmt.body)
        carried = tuple(name for name in assigned if name in env)

        induction = ssa.Value(name=f"%{stmt.target.id}", type=ssa.Type(kind="index"))
        block_args = [induction]
        loop_env = dict(env)
        loop_env[stmt.target.id] = induction
        iter_arg_attrs = []

        for name in carried:
            current = env[name]
            arg = ssa.Value(name=f"%{name}_iter", type=current.type)
            block_args.append(arg)
            loop_env[name] = arg
            iter_arg_attrs.append(
                {"name": name, "initial": current.name, "block_arg": arg.name}
            )

        loop_operations: list[ssa.Operation] = []
        self._lower_statements(stmt.body, loop_operations, loop_env)
        yield_values = tuple(loop_env[name].name for name in carried)
        loop_operations.append(ssa.Operation(opcode="scf.yield", operands=yield_values))

        results = tuple(self._temp(env[name].type, hint=name) for name in carried)
        operations.append(
            ssa.Operation(
                opcode="scf.for",
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
                    ssa.Block(
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
        operations: list[ssa.Operation],
        env: dict[str, ssa.Value],
    ) -> None:
        condition = self._ensure_bool_condition(
            self._lower_expr(stmt.test, operations, env), env
        )
        assigned = tuple(
            name for name in _assigned_names(stmt.body + stmt.orelse) if name in env
        )

        if not assigned:
            body_ops: list[ssa.Operation] = []
            self._lower_statements(stmt.body, body_ops, dict(env))
            regions = [ssa.Block(name="then", operations=tuple(body_ops))]

            if stmt.orelse:
                else_ops: list[ssa.Operation] = []
                self._lower_statements(stmt.orelse, else_ops, dict(env))
                regions.append(ssa.Block(name="else", operations=tuple(else_ops)))

            operations.append(
                ssa.Operation(
                    opcode="scf.if",
                    operands=(condition.name,),
                    attrs={"has_results": False},
                    regions=tuple(regions),
                )
            )

            return

        then_env = dict(env)
        then_ops: list[ssa.Operation] = []
        self._lower_statements(stmt.body, then_ops, then_env)
        then_ops.append(
            ssa.Operation(
                opcode="scf.yield",
                operands=tuple(then_env[name].name for name in assigned),
            )
        )

        else_env = dict(env)
        else_ops: list[ssa.Operation] = []

        if stmt.orelse:
            self._lower_statements(stmt.orelse, else_ops, else_env)

        else_ops.append(
            ssa.Operation(
                opcode="scf.yield",
                operands=tuple(else_env[name].name for name in assigned),
            )
        )

        results = tuple(self._temp(env[name].type, hint=name) for name in assigned)
        operations.append(
            ssa.Operation(
                opcode="scf.if",
                operands=(condition.name,),
                results=results,
                attrs={"assigned": assigned},
                regions=(
                    ssa.Block(name="then", operations=tuple(then_ops)),
                    ssa.Block(name="else", operations=tuple(else_ops)),
                ),
            )
        )

        for name, result in zip(assigned, results):
            env[name] = result

    def _range_bounds(
        self,
        node: ast.AST,
        operations: list[ssa.Operation],
        env: dict[str, ssa.Value],
    ) -> tuple[ssa.Value, ssa.Value, ssa.Value]:
        if not isinstance(node, ast.Call) or _call_leaf_name(node.func) != "range":
            raise LoweringError("Only for ... in range(...) loops are supported.")

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

        raise LoweringError(
            "Calls to `range()` with more than three arguments are unsupported."
        )

    def _lower_expr(self, node, operations, env) -> ssa.Value:
        handlers = {
            ast.Constant: self._lower_constant_expr,
            ast.Name: self._lower_name_expr,
            ast.UnaryOp: self._lower_unary_expr,
            ast.BinOp: self._lower_binary_expr,
            ast.BoolOp: self._lower_bool_expr,
            ast.Compare: self._lower_compare_expr,
            ast.IfExp: self._lower_if_expr,
            ast.Subscript: self._lower_subscript_expr,
            ast.Attribute: self._lower_attribute_expr,
            ast.Call: self._lower_call,
            ast.Tuple: self._lower_sequence_expr,
            ast.List: self._lower_sequence_expr,
        }

        try:
            handler = handlers[type(node)]
        except KeyError as exc:
            raise LoweringError(f"Unsupported expression: {ast.dump(node)}.") from exc
        return handler(node, operations, env)

    def _lower_constant_expr(self, node, operations, env) -> ssa.Value:
        del env

        return self._constant(operations, node.value)

    def _lower_name_expr(self, node, operations, env) -> ssa.Value:
        del operations

        if node.id in env:
            return env[node.id]

        if node.id in self.symbol_names:
            return self._named_value(node.id, ssa.Type(kind="index", dtype="index"))

        raise _lowering_error(node, f"Unknown value `{node.id}`")

    def _lower_unary_expr(self, node, operations, env) -> ssa.Value:
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

    def _lower_binary_expr(self, node, operations, env) -> ssa.Value:
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
            result_type=_binary_type(node.op, lhs.type, rhs.type),
        )

    def _lower_bool_expr(self, node, operations, env) -> ssa.Value:
        values = [self._lower_expr(value, operations, env) for value in node.values]

        if not values:
            raise LoweringError("Empty BoolOp is unsupported.")

        result = values[0]

        for rhs in values[1:]:
            result = self._emit(
                operations,
                f"arith.{_boolop_name(node.op)}",
                operands=(result.name, rhs.name),
                result_type=ssa.Type(kind="tensor", dtype="bool"),
            )
        return result

    def _lower_compare_expr(self, node, operations, env) -> ssa.Value:
        lhs = self._lower_expr(node.left, operations, env)
        comparisons = []

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
            raise LoweringError("Empty comparison is unsupported.")

        result = comparisons[0]

        for rhs in comparisons[1:]:
            result = self._emit(
                operations,
                "arith.and",
                operands=(result.name, rhs.name),
                result_type=_bool_type(result, rhs),
            )
        return result

    def _lower_if_expr(self, node, operations, env) -> ssa.Value:
        condition = self._lower_expr(node.test, operations, env)
        body = self._lower_expr(node.body, operations, env)
        orelse = self._lower_expr(node.orelse, operations, env)

        return self._emit(
            operations,
            "select.where",
            operands=(condition.name, body.name, orelse.name),
            result_type=body.type,
        )

    def _lower_subscript_expr(self, node, operations, env) -> ssa.Value:
        shape_dim = self._lower_shape_dim(node, operations, env)

        if shape_dim is not None:
            return shape_dim

        base = self._lower_expr(node.value, operations, env)
        index_values = tuple(
            value.name
            for value in self._lower_subscript_values(node.slice, operations, env)
        )
        source = isinstance(node.value, ast.Attribute) and node.value.attr == "source"

        return self._emit(
            operations,
            "tensor.extract" if index_values else "tensor.view",
            operands=(base.name, *index_values),
            attrs={"subscript": _unparse(node.slice), "source": source},
            result_type=_subscript_type(base.type, node.slice, source=source),
        )

    def _lower_attribute_expr(self, node, operations, env) -> ssa.Value:
        if node.attr == "T":
            value = self._lower_expr(node.value, operations, env)

            return self._emit(
                operations,
                "linalg.transpose",
                operands=(value.name,),
                attrs={"python": _unparse(node)},
                result_type=_transpose_type(value.type),
            )

        if node.attr == "source":
            return self._lower_expr(node.value, operations, env)
        return self._emit(
            operations,
            "symbol.attr",
            attrs={"expr": _unparse(node)},
            result_type=ssa.Type(kind="symbol"),
        )

    def _lower_sequence_expr(self, node, operations, env) -> ssa.Value:
        items = tuple(self._lower_expr(item, operations, env) for item in node.elts)

        return self._emit(
            operations,
            "tuple.construct",
            operands=tuple(item.name for item in items),
            attrs={"items": tuple(_unparse(item) for item in node.elts)},
            result_type=ssa.Type(kind="tuple"),
        )

    def _lower_call(self, node, operations, env) -> ssa.Value:
        special = self._lower_float_literal_call(node, operations)

        if special is not None:
            return special

        method = self._lower_tensor_method_call(node, operations, env)

        if method is not None:
            return method

        name = _call_leaf_name(node.func)
        constructor = self._lower_constructor_call(name, node, operations, env)

        if constructor is not None:
            return constructor

        operands = tuple(self._lower_expr(arg, operations, env) for arg in node.args)
        handlers = (
            self._lower_memory_call,
            self._lower_reduction_call,
            self._lower_linalg_call,
            self._lower_elementwise_call,
        )

        for handler in handlers:
            result = handler(name, node, operands, operations)

            if result is not None:
                return result

        raise _lowering_error(
            node,
            f"Unsupported function call `{_unparse(node.func)}`; helper calls must "
            "be statically inlinable",
        )

    def _lower_float_literal_call(self, node, operations):
        if _call_leaf_name(node.func) != "float" or len(node.args) != 1:
            return None

        literal = _literal_value(node.args[0])

        if literal == "-inf":
            return self._constant(operations, float("-inf"))

        if literal == "inf":
            return self._constant(operations, float("inf"))
        return None

    def _lower_tensor_method_call(self, node, operations, env):
        if not isinstance(node.func, ast.Attribute) or _is_namespace_ref(
            node.func.value
        ):
            return None

        method = node.func.attr
        receiver = self._lower_tensor_ref(node.func.value, operations, env)

        if method == "to":
            if not node.args:
                raise _lowering_error(node, "`to()` requires a destination dtype")

            dtype = _unparse(node.args[0])

            return self._emit(
                operations,
                "tensor.cast",
                operands=(receiver.name,),
                attrs={"dtype": dtype},
                result_type=_cast_type(receiver.type, dtype),
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
            return self._lower_stride_method(node, receiver, operations)

        if method == "data_ptr":
            return self._emit(
                operations,
                "mem.data_ptr",
                operands=(receiver.name,),
                result_type=ssa.Type(kind="pointer", dtype=receiver.type.dtype),
            )

        if method in {"sum", "max", "min"}:
            return self._lower_reduce_method(method, node, receiver, operations)

        if method in _SUPPORTED_MATH_CALLS:
            args = tuple(self._lower_expr(arg, operations, env) for arg in node.args)

            return self._emit(
                operations,
                f"math.{method}",
                operands=(receiver.name, *(arg.name for arg in args)),
                result_type=receiver.type,
            )
        return None

    def _lower_stride_method(self, node, receiver, operations):
        dim = _literal_value(node.args[0]) if node.args else 0
        source = (
            isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "source"
        )

        return self._emit(
            operations,
            "tensor.stride",
            operands=(receiver.name,),
            attrs={"dim": dim, "source": source},
            result_type=ssa.Type(kind="index", dtype="index"),
        )

    def _lower_reduce_method(self, method, node, receiver, operations):
        axis = _axis_from_call(node, positional_index=0)

        return self._emit(
            operations,
            f"reduce.{method}",
            operands=(receiver.name,),
            attrs={"axis": axis},
            result_type=_reduce_type(receiver.type, axis, strict=self.strict),
        )

    def _lower_constructor_call(self, name, node, operations, env):
        if name not in {"zeros", "empty", "full"}:
            return None

        shape = (
            _shape_tuple_from_ast(node.args[0], operations, env, self)
            if node.args
            else ()
        )
        dtype = _keyword_text(node, "dtype")

        if name in {"zeros", "empty"}:
            return self._emit(
                operations,
                "tensor.zeros",
                attrs={
                    "shape": _unparse(node.args[0]) if node.args else None,
                    "dtype": dtype,
                },
                result_type=ssa.Type(kind="tensor", shape=shape, dtype=dtype),
            )

        operands = tuple(
            self._lower_expr(argument, operations, env) for argument in node.args[1:]
        )

        return self._emit(
            operations,
            "tensor.full",
            operands=tuple(value.name for value in operands),
            attrs={
                "shape": _unparse(node.args[0]) if node.args else None,
                "value": _literal_value(node.args[1]) if len(node.args) > 1 else None,
                "dtype": dtype,
            },
            result_type=ssa.Type(kind="tensor", shape=shape, dtype=dtype),
        )

    def _lower_memory_call(self, name, node, operands, operations):
        if name == "load":
            if len(operands) != 1 or operands[0].type.kind != "pointer":
                raise LoweringError(
                    "The `load()` operation requires exactly one pointer operand."
                )
            return self._emit(
                operations,
                "mem.load",
                operands=(operands[0].name,),
                result_type=_load_type(operands[0].type),
            )

        if name == "fill" and operands:
            if len(operands) == 1:
                return operands[0]

            destination, value = operands[0], operands[1]
            self._store_intrinsic_result(operations, destination, value, name)

            return destination

        if name == "copy" and len(operands) >= 2:
            source, destination = operands[0], operands[1]
            self._store_intrinsic_result(operations, destination, source, name)

            return destination

        if name == "atomic_add":
            dtype = operands[1].type.dtype if len(operands) > 1 else "float32"

            return self._emit(
                operations,
                "mem.atomic_add",
                operands=tuple(value.name for value in operands),
                result_type=ssa.Type(kind="scalar", dtype=dtype),
            )
        return None

    def _lower_reduction_call(self, name, node, operands, operations):
        if name.startswith("reduce_") and operands:
            operator = name.removeprefix("reduce_")

            if operator in {"sum", "max", "min"}:
                axis = _axis_from_call(node, positional_index=2)
                reduced = self._emit(
                    operations,
                    f"reduce.{operator}",
                    operands=(operands[0].name,),
                    attrs={"axis": axis},
                    result_type=_reduce_type(
                        operands[0].type,
                        axis,
                        strict=self.strict,
                    ),
                )

                if len(operands) > 1:
                    self._store_intrinsic_result(operations, operands[1], reduced, name)

                    return operands[1]
                return reduced

        if name not in {"sum", "max", "min"}:
            return None

        if not operands:
            raise _lowering_error(node, f"`{name}()` requires an input operand")

        axis = _axis_from_call(node, positional_index=1)

        return self._emit(
            operations,
            f"reduce.{name}",
            operands=(operands[0].name,),
            attrs={"axis": axis},
            result_type=_reduce_type(operands[0].type, axis, strict=self.strict),
        )

    def _lower_linalg_call(self, name, node, operands, operations):
        if name in {"matmul", "dot"} and len(operands) >= 3:
            result = self._emit(
                operations,
                "linalg.matmul" if name == "matmul" else "linalg.dot",
                operands=(operands[0].name, operands[1].name),
                result_type=operands[2].type,
            )
            self._store_intrinsic_result(operations, operands[2], result, name)

            return operands[2]

        if name in {"dot", "matmul"}:
            result_type = (
                _matmul_type(
                    operands[0].type,
                    operands[1].type,
                    strict=self.strict and name == "matmul",
                )
                if len(operands) >= 2
                else ssa.Type(kind="tensor")
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
            result_type = (
                _transpose_type(operands[0].type)
                if operands
                else ssa.Type(kind="tensor")
            )

            return self._emit(
                operations,
                "linalg.transpose",
                operands=tuple(value.name for value in operands),
                result_type=result_type,
            )
        return None

    def _lower_elementwise_call(self, name, node, operands, operations):
        if name == "where":
            if len(operands) != 3:
                raise _lowering_error(node, "`where()` requires three operands")
            return self._emit(
                operations,
                "select.where",
                operands=tuple(value.name for value in operands),
                result_type=_broadcast_type(operands[1].type, operands[2].type),
            )

        if name in {"maximum", "minimum"}:
            result_type = operands[0].type if operands else ssa.Type(kind="tensor")

            return self._emit(
                operations,
                f"arith.{name}",
                operands=tuple(value.name for value in operands),
                result_type=result_type,
            )

        if name in _SUPPORTED_MATH_CALLS:
            return self._emit(
                operations,
                f"math.{name}",
                operands=tuple(value.name for value in operands),
                attrs={"callee": _unparse(node.func)},
                result_type=_math_result_type(name, operands),
            )

        if isinstance(node.func, ast.Attribute) and _is_namespace_ref(node.func.value):
            result_type = operands[0].type if operands else ssa.Type(kind="tensor")

            return self._emit(
                operations,
                f"call.{name}",
                operands=tuple(value.name for value in operands),
                attrs={"callee": _unparse(node.func)},
                result_type=result_type,
            )
        return None

    def _store_intrinsic_result(
        self,
        operations: list[ssa.Operation],
        destination: ssa.Value,
        value: ssa.Value,
        intrinsic: str,
    ) -> None:
        if destination not in self.outputs:
            self.outputs.append(destination)

        operations.append(
            ssa.Operation(
                opcode="mem.store",
                operands=(value.name, destination.name),
                attrs={"target": destination.name, "intrinsic": intrinsic},
            )
        )

    def _lower_tensor_ref(
        self,
        node: ast.AST,
        operations: list[ssa.Operation],
        env: dict[str, ssa.Value],
    ) -> ssa.Value:
        shape_dim = self._lower_shape_dim(node, operations, env)

        if shape_dim is not None:
            return shape_dim

        if isinstance(node, ast.Name):
            if node.id in env:
                return env[node.id]

            if node.id in self.symbol_names:
                return self._named_value(node.id, ssa.Type(kind="index", dtype="index"))

            raise _lowering_error(node, f"Unknown tensor value `{node.id}`")

        if isinstance(node, ast.Subscript):
            return self._lower_expr(node, operations, env)

        if isinstance(node, ast.Attribute):
            if node.attr == "source":
                return self._lower_tensor_ref(node.value, operations, env)
            return self._emit(
                operations,
                "symbol.attr",
                attrs={"expr": _unparse(node)},
                result_type=ssa.Type(kind="symbol"),
            )
        return self._lower_expr(node, operations, env)

    def _lower_shape_dim(
        self,
        node: ast.AST,
        operations: list[ssa.Operation],
        env: dict[str, ssa.Value],
    ) -> ssa.Value | None:
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
            result_type=ssa.Type(kind="index"),
        )

    def _lower_subscript_values(
        self,
        node: ast.AST,
        operations: list[ssa.Operation],
        env: dict[str, ssa.Value],
    ) -> tuple[ssa.Value, ...]:
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

    def _constant(self, operations: list[ssa.Operation], value: Any) -> ssa.Value:
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
            result_type=ssa.Type(kind="scalar", dtype=dtype),
        )

    def _emit(
        self,
        operations: list[ssa.Operation],
        opcode: str,
        *,
        operands: tuple[str, ...] = (),
        attrs: Mapping[str, Any] | None = None,
        result_type: ssa.Type | None = None,
    ) -> ssa.Value:
        result = self._temp(result_type or ssa.Type(kind="tensor"))
        operations.append(
            ssa.Operation(
                opcode=opcode,
                operands=operands,
                results=(result,),
                attrs=dict(attrs or {}),
            )
        )

        return result

    def _temp(self, type_: ssa.Type, *, hint: str | None = None) -> ssa.Value:
        name = f"%{self.temp_index}" if hint is None else f"%{hint}_{self.temp_index}"
        self.temp_index += 1
        value = ssa.Value(name=name, type=type_)
        self.values[name] = value

        return value

    def _named_value(self, name: str, type_: ssa.Type | None = None) -> ssa.Value:
        if name not in self.values:
            self.values[name] = ssa.Value(
                name=name,
                type=type_ or self.tensor_types.get(name, ssa.Type(kind="tensor")),
            )
        return self.values[name]

    def _ensure_bool_condition(
        self, value: ssa.Value, env: dict[str, ssa.Value]
    ) -> ssa.Value:
        if value.type.dtype == "bool":
            return value

        if value.type.kind != "scalar" or value.type.dtype not in {None, "symbol"}:
            return value

        replacement = ssa.Value(
            name=value.name,
            type=ssa.Type(
                kind="scalar",
                shape=value.type.shape,
                dtype="bool",
                attrs=dict(value.type.attrs),
            ),
        )
        self.values[value.name] = replacement

        for name, current in tuple(env.items()):
            if current.name == value.name:
                env[name] = replacement
        return replacement


class _AssignedNameVisitor(ast.NodeVisitor):
    def __init__(self):
        self.names: list[str] = []

    def _record(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name) and target.id not in self.names:
            self.names.append(target.id)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._record(target)

        self.generic_visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._record(node.target)

        if node.value is not None:
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._record(node.target)
        self.generic_visit(node.value)

    def visit_For(self, node: ast.For) -> None:
        self._record(node.target)

        for statement in (*node.body, *node.orelse):
            self.visit(statement)


def _assigned_names(statements: Iterable[ast.stmt]) -> tuple[str, ...]:
    visitor = _AssignedNameVisitor()

    for statement in statements:
        visitor.visit(statement)
    return tuple(visitor.names)


def _count_operations(operations: Iterable[ssa.Operation]) -> int:
    total = 0

    for op in operations:
        total += 1

        for region in op.regions:
            total += _count_operations(region.operations)
    return total


def _shape_tuple_from_ast(
    node: ast.AST,
    operations: list[ssa.Operation],
    env: dict[str, ssa.Value],
    builder: _ApplicationSSABuilder,
) -> tuple[str, ...]:
    if isinstance(node, ast.Attribute) and node.attr == "shape":
        value = _value_for_shape_node(node.value, env, builder)

        if value is not None:
            return tuple(str(dim) for dim in value.type.shape)

    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(
            _shape_text_from_ast(item, operations, env, builder) for item in node.elts
        )

    text = _shape_text_from_ast(node, operations, env, builder)

    return () if text in {"", "None"} else (text,)


def _shape_text_from_ast(
    node: ast.AST,
    operations: list[ssa.Operation],
    env: dict[str, ssa.Value],
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
    env: dict[str, ssa.Value],
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
    env: dict[str, ssa.Value],
    builder: _ApplicationSSABuilder,
) -> ssa.Value | None:
    if isinstance(node, ast.Name):
        return env.get(node.id) or builder.values.get(node.id)

    if isinstance(node, ast.Subscript):
        base = _value_for_shape_node(node.value, env, builder)

        if base is None:
            return None
        return ssa.Value(
            name="<shape-proxy>", type=_subscript_type(base.type, node.slice)
        )
    return None


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
    "rand",
    "rsqrt",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
}


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


def _lowering_error(node: ast.AST, message: str) -> LoweringError:
    line = getattr(node, "lineno", None)
    column = getattr(node, "col_offset", None)
    location = (
        f" at line {line}, column {column + 1}"
        if line is not None and column is not None
        else ""
    )

    return LoweringError(f"{message}{location}.")


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
