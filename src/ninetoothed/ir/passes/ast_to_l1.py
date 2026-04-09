import ast
import copy

from ninetoothed.tensor import Tensor
from ninetoothed.ir.tensor_ir import (
    L1Assign,
    L1Attribute,
    L1BinOp,
    L1BoolOp,
    L1Call,
    L1Compare,
    L1Constant,
    L1DataPtr,
    L1DtypeAttr,
    L1ExprStmt,
    L1For,
    L1Function,
    L1If,
    L1IfExp,
    L1Name,
    L1Offsets,
    L1Return,
    L1Statement,
    L1Stride,
    L1Subscript,
    L1TensorAccess,
    L1TensorParam,
    L1Tuple,
    L1UnaryOp,
    TileOp,
)

# Map from AST operator types to string names
_BINOP_MAP = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.Pow: "**",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.BitAnd: "&",
}

_UNARYOP_MAP = {
    ast.UAdd: "+",
    ast.USub: "-",
    ast.Not: "not",
    ast.Invert: "~",
}

_CMPOP_MAP = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Is: "is",
    ast.IsNot: "is not",
    ast.In: "in",
    ast.NotIn: "not in",
}

_BOOLOP_MAP = {
    ast.And: "and",
    ast.Or: "or",
}


class ASTToL1Pass:
    """Transform Python AST (L0) to Tensor + Tiling IR (L1).

    Traverses the original AST and constructs L1 IR nodes,
    using Tensor object metadata (_history, _levels, etc.)
    to capture tiling information.
    """

    def __init__(self, context):
        """Initialize with the function's type annotations.

        :param context: dict mapping parameter names to Tensor objects
            (from inspect.get_annotations(func))
        """
        self._context = context

    def transform(self, tree):
        """Transform an AST Module into an L1Function.

        :param tree: ast.Module
        :return: L1Function
        """
        func_def = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                func_def = node
                break

        if func_def is None:
            raise ValueError("No FunctionDef found in AST module")

        return self._visit_FunctionDef(func_def)

    def _visit_FunctionDef(self, node):
        """Build L1Function from ast.FunctionDef."""
        params = []
        for arg in node.args.args:
            if arg.arg in self._context:
                params.append(self._make_param(arg.arg, self._context[arg.arg]))

        body = []
        for stmt in node.body:
            result = self._visit_stmt(stmt)
            if result is not None:
                if isinstance(result, list):
                    body.extend(result)
                else:
                    body.append(result)

        return L1Function(name=node.name, params=params, body=body)

    def _make_param(self, name, tensor):
        """Build L1TensorParam from a Tensor object."""
        tile_history = self._extract_tile_history(tensor)

        return L1TensorParam(
            name=name,
            tensor=tensor,
            tile_history=tile_history,
            ndim=tensor.ndim,
            dtype=tensor.dtype,
            other=tensor.other,
            jagged_dim=tensor.jagged_dim,
        )

    def _extract_tile_history(self, tensor):
        """Walk tensor._history to build TileOp list."""
        history = []
        for entry in tensor._history:
            if len(entry) == 3:
                func, args, kwargs = entry
                history.append(
                    TileOp(kind=func.__name__, args=args, kwargs=kwargs)
                )
        return history

    def _in_context(self, name):
        """Check if a name is a tensor parameter in context."""
        return name in self._context

    def _resolve_tensor(self, node):
        """Try to resolve a Tensor object from an AST node.

        Returns the Tensor if found, None otherwise.
        """
        if isinstance(node, ast.Name) and node.id in self._context:
            return self._context[node.id]
        if isinstance(node, ast.Subscript) and isinstance(
            node.value, ast.Name
        ):
            if node.value.id in self._context:
                return self._context[node.value.id]
        return None

    def _extract_param_name(self, node):
        """Extract the parameter name from an AST node.

        Handles both ast.Name (e.g. 'k') and ast.Subscript (e.g. 'k[i]').
        Returns the name string or None.
        """
        if isinstance(node, ast.Name) and node.id in self._context:
            return node.id
        if isinstance(node, ast.Subscript) and isinstance(
            node.value, ast.Name
        ):
            if node.value.id in self._context:
                return node.value.id
        return None

    # --- Statement visitors ---

    def _visit_stmt(self, node):
        """Dispatch to the appropriate statement visitor."""
        if isinstance(node, ast.Assign):
            return self._visit_Assign(node)
        if isinstance(node, ast.Expr):
            expr = self._visit_expr(node.value)
            if expr is not None:
                return L1ExprStmt(expr)
            return None
        if isinstance(node, ast.Return):
            value = self._visit_expr(node.value) if node.value else None
            return L1Return(value)
        if isinstance(node, ast.For):
            return self._visit_For(node)
        if isinstance(node, ast.If):
            return self._visit_If(node)
        if isinstance(node, ast.AugAssign):
            return self._visit_AugAssign(node)
        return None

    def _visit_Assign(self, node):
        """Visit an assignment statement."""
        if len(node.targets) != 1:
            return None

        target = node.targets[0]

        # Simple name target
        if isinstance(target, ast.Name) and self._in_context(target.id):
            value = self._visit_expr(node.value)
            tensor = self._context[target.id]
            return L1Assign(
                target=L1TensorAccess(param_name=target.id, tensor=tensor),
                value=value,
            )

        # Subscript target
        if isinstance(target, ast.Subscript) and isinstance(target.ctx, ast.Store):
            if isinstance(target.value, ast.Name) and self._in_context(target.value.id):
                indices = self._extract_indices(target.slice)
                value = self._visit_expr(node.value)
                tensor = self._context[target.value.id]
                return L1Assign(
                    target=L1TensorAccess(
                        param_name=target.value.id, tensor=tensor, indices=indices
                    ),
                    value=value,
                )

            # Handle tensor attribute chain: e.g., output.source[...]
            resolved = self._resolve_tensor(target.value)
            if resolved is not None:
                indices = self._extract_indices(target.slice)
                value = self._visit_expr(node.value)
                # Try to get the param name
                param_name = None
                if isinstance(target.value, ast.Name):
                    param_name = target.value.id
                return L1Assign(
                    target=L1TensorAccess(
                        param_name=param_name,
                        tensor=resolved,
                        indices=indices,
                    ),
                    value=value,
                )

        # Fallback: generic assignment
        target_expr = self._visit_expr(target)
        value = self._visit_expr(node.value)
        if target_expr is not None:
            return L1Assign(target=target_expr, value=value)

        return None

    def _visit_AugAssign(self, node):
        """Visit augmented assignment: target += value."""
        target_expr = self._visit_expr(node.target)
        value = self._visit_expr(ast.BinOp(left=node.target, op=node.op, right=node.value))
        if target_expr is not None and value is not None:
            return L1Assign(target=target_expr, value=value)
        return None

    def _visit_For(self, node):
        """Visit a for loop."""
        target = node.target.id if isinstance(node.target, ast.Name) else str(
            ast.unparse(node.target)
        )
        iter_expr = self._visit_expr(node.iter)
        body = []
        for stmt in node.body:
            result = self._visit_stmt(stmt)
            if result is not None:
                if isinstance(result, list):
                    body.extend(result)
                else:
                    body.append(result)
        return L1For(target=target, iter=iter_expr, body=body)

    def _visit_If(self, node):
        """Visit an if statement."""
        test = self._visit_expr(node.test)
        body = []
        for stmt in node.body:
            result = self._visit_stmt(stmt)
            if result is not None:
                if isinstance(result, list):
                    body.extend(result)
                else:
                    body.append(result)
        orelse = []
        for stmt in node.orelse:
            result = self._visit_stmt(stmt)
            if result is not None:
                if isinstance(result, list):
                    orelse.extend(result)
                else:
                    orelse.append(result)
        return L1If(test=test, body=body, orelse=orelse)

    # --- Expression visitors ---

    def _visit_expr(self, node):
        """Dispatch to the appropriate expression visitor."""
        if node is None:
            return None

        if isinstance(node, ast.BinOp):
            return self._visit_BinOp(node)
        if isinstance(node, ast.UnaryOp):
            return self._visit_UnaryOp(node)
        if isinstance(node, ast.Compare):
            return self._visit_Compare(node)
        if isinstance(node, ast.BoolOp):
            return self._visit_BoolOp(node)
        if isinstance(node, ast.IfExp):
            return self._visit_IfExp(node)
        if isinstance(node, ast.Call):
            return self._visit_Call(node)
        if isinstance(node, ast.Attribute):
            return self._visit_Attribute(node)
        if isinstance(node, ast.Subscript):
            return self._visit_Subscript(node)
        if isinstance(node, ast.Name):
            return self._visit_Name(node)
        if isinstance(node, ast.Constant):
            return L1Constant(value=node.value)
        if isinstance(node, ast.Tuple):
            return L1Tuple(elts=tuple(self._visit_expr(e) for e in node.elts))
        if isinstance(node, ast.List):
            return L1Tuple(elts=tuple(self._visit_expr(e) for e in node.elts))

        return None

    def _visit_BinOp(self, node):
        """Visit binary operation."""
        op = _BINOP_MAP.get(type(node.op), type(node.op).__name__)
        lhs = self._visit_expr(node.left)
        rhs = self._visit_expr(node.right)
        return L1BinOp(op=op, lhs=lhs, rhs=rhs)

    def _visit_UnaryOp(self, node):
        """Visit unary operation."""
        op = _UNARYOP_MAP.get(type(node.op), type(node.op).__name__)
        operand = self._visit_expr(node.operand)
        return L1UnaryOp(op=op, operand=operand)

    def _visit_Compare(self, node):
        """Visit comparison (supports chained comparisons)."""
        left = self._visit_expr(node.left)
        result = left

        for op_node, comparator in zip(node.ops, node.comparators):
            op = _CMPOP_MAP.get(type(op_node), type(op_node).__name__)
            right = self._visit_expr(comparator)
            result = L1Compare(op=op, left=result, right=right)

        return result

    def _visit_BoolOp(self, node):
        """Visit boolean operation (and/or)."""
        op = _BOOLOP_MAP.get(type(node.op), type(node.op).__name__)
        values = [self._visit_expr(v) for v in node.values]
        return L1BoolOp(op=op, values=values)

    def _visit_IfExp(self, node):
        """Visit ternary expression."""
        test = self._visit_expr(node.test)
        body = self._visit_expr(node.body)
        orelse = self._visit_expr(node.orelse)
        return L1IfExp(test=test, body=body, orelse=orelse)

    def _visit_Call(self, node):
        """Visit function call."""
        # Handle tensor method calls: tensor.data_ptr(), tensor.offsets(), tensor.stride()
        if isinstance(node.func, ast.Attribute):
            func_attr = node.func.attr

            if func_attr in ("data_ptr", "offsets", "stride"):
                return self._visit_tensor_method_call(node)

            # Generic attribute call
            obj = self._visit_expr(node.func.value)
            args = tuple(self._visit_expr(a) for a in node.args)
            kwargs = {kw.arg: self._visit_expr(kw.value) for kw in node.keywords if kw.arg}
            func_name = self._format_func_name(node.func)
            return L1Call(func=func_name, args=args, kwargs=kwargs)

        # Handle simple function call: func(args)
        func_name = self._format_func_name(node.func)
        args = tuple(self._visit_expr(a) for a in node.args)
        kwargs = {kw.arg: self._visit_expr(kw.value) for kw in node.keywords if kw.arg}
        return L1Call(func=func_name, args=args, kwargs=kwargs)

    def _visit_tensor_method_call(self, node):
        """Visit tensor.data_ptr(), tensor.offsets(dim), tensor.stride(dim)."""
        func_attr = node.func.attr
        value_node = node.func.value

        # Resolve tensor from the value node
        tensor = self._resolve_tensor(value_node)
        param_name = self._extract_param_name(value_node)
        if tensor is None:
            # Try visiting the value to get a tensor from attribute chain
            visited = self._visit_expr(value_node)
            if isinstance(visited, L1TensorAccess):
                tensor = visited.tensor

        if tensor is None:
            # Fallback: generate a generic call
            return self._visit_Call_generic(node)

        if func_attr == "data_ptr":
            return L1DataPtr(tensor=tensor, param_name=param_name)

        if func_attr == "offsets":
            dim = None
            if node.args:
                try:
                    dim = ast.literal_eval(node.args[0])
                except (ValueError, TypeError):
                    dim = self._visit_expr(node.args[0])
            return L1Offsets(tensor=tensor, dim=dim, param_name=param_name)

        if func_attr == "stride":
            dim = ast.literal_eval(node.args[0]) if node.args else 0
            return L1Stride(tensor=tensor, dim=dim, param_name=param_name)

        return self._visit_Call_generic(node)

    def _visit_Call_generic(self, node):
        """Generic call fallback."""
        func_name = self._format_func_name(node.func)
        args = tuple(self._visit_expr(a) for a in node.args)
        kwargs = {kw.arg: self._visit_expr(kw.value) for kw in node.keywords if kw.arg}
        return L1Call(func=func_name, args=args, kwargs=kwargs)

    def _visit_Attribute(self, node):
        """Visit attribute access: obj.attr."""
        # Check if this is a tensor parameter's attribute
        obj = node.value
        if isinstance(obj, ast.Name) and self._in_context(obj.id):
            tensor = self._context[obj.id]

            if node.attr == "dtype":
                return L1DtypeAttr(tensor=tensor, param_name=obj.id)

            # For .source, .shape, etc., return an attribute node
            obj_expr = self._visit_expr(obj)
            return L1Attribute(obj=obj_expr, attr=node.attr)

        # Check chained attributes: e.g., input.source.data_ptr()
        # is handled by _visit_Call which processes the outermost call
        obj_expr = self._visit_expr(obj)
        return L1Attribute(obj=obj_expr, attr=node.attr)

    def _visit_Subscript(self, node):
        """Visit subscript/indexing: value[slice]."""
        value = node.value

        # Check if this is a tensor access: tensor_name[i, j]
        if isinstance(value, ast.Name) and self._in_context(value.id):
            tensor = self._context[value.id]
            indices = self._extract_indices(node.slice)
            return L1TensorAccess(param_name=value.id, tensor=tensor, indices=indices)

        # Check if value is a tensor attribute chain
        resolved = self._resolve_tensor(value)
        if resolved is not None:
            indices = self._extract_indices(node.slice)
            param_name = value.id if isinstance(value, ast.Name) else None
            return L1TensorAccess(
                param_name=param_name, tensor=resolved, indices=indices
            )

        # Generic subscript
        value_expr = self._visit_expr(value)
        slice_expr = self._visit_slice(node.slice)
        return L1Subscript(value=value_expr, slice=slice_expr)

    def _visit_Name(self, node):
        """Visit variable reference."""
        if self._in_context(node.id) and isinstance(
            node.ctx, ast.Load
        ):
            # Tensor in load context -> tensor access (scalar load)
            tensor = self._context[node.id]
            return L1TensorAccess(param_name=node.id, tensor=tensor)
        return L1Name(name=node.id)

    # --- Helpers ---

    def _extract_indices(self, slice_node):
        """Extract index expressions from a subscript slice."""
        if isinstance(slice_node, ast.Tuple):
            return tuple(self._visit_expr(elt) for elt in slice_node.elts)
        return (self._visit_expr(slice_node),)

    def _visit_slice(self, slice_node):
        """Visit a slice node (for generic subscripts)."""
        if isinstance(slice_node, ast.Slice):
            parts = []
            if slice_node.lower is not None:
                parts.append(self._visit_expr(slice_node.lower))
            else:
                parts.append(None)
            if slice_node.upper is not None:
                parts.append(self._visit_expr(slice_node.upper))
            else:
                parts.append(None)
            if slice_node.step is not None:
                parts.append(self._visit_expr(slice_node.step))
            else:
                parts.append(None)
            return L1Tuple(elts=tuple(parts))
        if isinstance(slice_node, ast.Tuple):
            return L1Tuple(elts=tuple(self._visit_expr(elt) for elt in slice_node.elts))
        return self._visit_expr(slice_node)

    def _format_func_name(self, func_node):
        """Format a function name from an AST node."""
        if isinstance(func_node, ast.Name):
            return func_node.id
        if isinstance(func_node, ast.Attribute):
            obj_name = self._format_func_name(func_node.value)
            return f"{obj_name}.{func_node.attr}"
        return ast.unparse(func_node)
