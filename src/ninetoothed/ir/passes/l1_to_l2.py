import ast
import math

from ninetoothed.generation.generation import CodeGenerator
from ninetoothed.symbol import Symbol
from ninetoothed.language import call
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
    L1Stride,
    L1Subscript,
    L1TensorAccess,
    L1Tuple,
    L1UnaryOp,
)
from ninetoothed.ir.memory_ir import (
    L2Assign,
    L2BinOp,
    L2BoolOp,
    L2Call,
    L2Compare,
    L2Constant,
    L2ExprStmt,
    L2For,
    L2Function,
    L2If,
    L2IfExp,
    L2Invariant,
    L2Load,
    L2MaskExpr,
    L2Name,
    L2Param,
    L2Return,
    L2Store,
    L2Subscript,
    L2Tuple,
)

_NAME_FOR_PID = "ninetoothed_pid"


class L1ToL2Pass:
    """Transform Tensor + Tiling IR (L1) to Memory IR (L2).

    Uses the existing CodeGenerator static methods for pointer/mask computation
    (which work with Symbol objects), then converts the Symbol results to L2 IR nodes.
    """

    def __init__(self):
        self._symbol_invariants = {}  # name -> Symbol (used during offset computation)
        self._context = {}  # param_name -> Tensor

    def transform(self, l1_func):
        """Transform L1Function to L2Function.

        :param l1_func: L1Function
        :return: L2Function
        """
        self._context = {p.name: p.tensor for p in l1_func.params if p.tensor}
        self._symbol_invariants = {}

        # Set up a mock CodeGenerator-like invariants dict
        # The CodeGenerator static methods need this for pid indices etc.
        self._cg_invariants = {}

        l2_body = []
        for stmt in l1_func.body:
            result = self._visit_stmt(stmt)
            if result is not None:
                if isinstance(result, list):
                    l2_body.extend(result)
                else:
                    l2_body.append(result)

        # Convert symbol invariants to L2 invariants
        l2_invariants = []
        for name, sym in self._symbol_invariants.items():
            l2_invariants.append(L2Invariant(target=name, value=self._symbol_to_l2(sym)))

        l2_params = self._build_params(l1_func)

        return L2Function(
            name=l1_func.name,
            params=l2_params,
            body=l2_body,
            invariants=l2_invariants,
        )

    def _build_params(self, l1_func):
        """Build L2Param list from L1Function."""
        params = []
        seen = set()
        for p in l1_func.params:
            if p.tensor is None:
                continue
            tensor = p.tensor
            for name_sym in tensor.names():
                name_str = name_sym.node.id if hasattr(name_sym, "node") else str(name_sym)
                if name_str != "ninetoothed" and name_str not in seen:
                    seen.add(name_str)
                    params.append(L2Param(name=name_str))
        return params

    # --- Statement visitors ---

    def _visit_stmt(self, stmt):
        if isinstance(stmt, L1Assign):
            return self._visit_Assign(stmt)
        if isinstance(stmt, L1ExprStmt):
            return self._visit_ExprStmt(stmt)
        if isinstance(stmt, L1Return):
            value = self._convert_expr(stmt.value) if stmt.value else None
            return L2Return(value)
        if isinstance(stmt, L1For):
            return self._visit_For(stmt)
        if isinstance(stmt, L1If):
            return self._visit_If(stmt)
        return None

    def _visit_Assign(self, stmt):
        """Convert L1Assign to L2 statements."""
        target = stmt.target
        value = stmt.value

        if isinstance(target, L1TensorAccess) and target.tensor is not None:
            tensor = target.tensor
            indices = target.indices

            # Convert value expression first (may generate loads)
            l2_value = self._convert_expr(value)

            # Generate store using Symbol-based computation
            pointer_sym, mask_sym = self._compute_pointers_and_mask(
                tensor, indices
            )
            return L2Store(
                pointer=self._symbol_to_l2(pointer_sym),
                value=l2_value,
                mask=self._symbol_to_l2(mask_sym),
            )

        # Generic assignment
        l2_target = self._convert_expr(target)
        l2_value = self._convert_expr(value)
        if isinstance(l2_target, L2Name):
            return L2Assign(target=l2_target.name, value=l2_value)
        return L2Assign(target=str(l2_target), value=l2_value)

    def _visit_ExprStmt(self, stmt):
        expr = self._convert_expr(stmt.expr)
        return L2ExprStmt(expr)

    def _visit_For(self, stmt):
        iter_expr = self._convert_expr(stmt.iter)
        body = []
        for s in stmt.body:
            result = self._visit_stmt(s)
            if result is not None:
                if isinstance(result, list):
                    body.extend(result)
                else:
                    body.append(result)
        return L2For(target=stmt.target, iter=iter_expr, body=body)

    def _visit_If(self, stmt):
        test = self._convert_expr(stmt.test)
        body = []
        for s in stmt.body:
            result = self._visit_stmt(s)
            if result is not None:
                if isinstance(result, list):
                    body.extend(result)
                else:
                    body.append(result)
        orelse = []
        for s in stmt.orelse:
            result = self._visit_stmt(s)
            if result is not None:
                if isinstance(result, list):
                    orelse.extend(result)
                else:
                    orelse.append(result)
        return L2If(test=test, body=body, orelse=orelse)

    # --- Expression visitors ---

    def _convert_expr(self, expr):
        """Convert an L1 expression to an L2 expression."""
        if expr is None:
            return None

        if isinstance(expr, L1TensorAccess):
            return self._convert_tensor_access(expr)

        if isinstance(expr, L1BinOp):
            lhs = self._convert_expr(expr.lhs)
            rhs = self._convert_expr(expr.rhs)
            return L2BinOp(op=expr.op, lhs=lhs, rhs=rhs)

        if isinstance(expr, L1UnaryOp):
            operand = self._convert_expr(expr.operand)
            return L2BinOp(op=expr.op, lhs=L2Constant(value=None), rhs=operand)

        if isinstance(expr, L1Compare):
            left = self._convert_expr(expr.left)
            right = self._convert_expr(expr.right)
            return L2Compare(op=expr.op, left=left, right=right)

        if isinstance(expr, L1BoolOp):
            values = [self._convert_expr(v) for v in expr.values]
            return L2BoolOp(op=expr.op, values=values)

        if isinstance(expr, L1IfExp):
            test = self._convert_expr(expr.test)
            body = self._convert_expr(expr.body)
            orelse = self._convert_expr(expr.orelse)
            return L2IfExp(test=test, body=body, orelse=orelse)

        if isinstance(expr, L1Call):
            args = tuple(self._convert_expr(a) for a in expr.args)
            kwargs = {k: self._convert_expr(v) for k, v in expr.kwargs.items()}
            return L2Call(func=expr.func, args=args, kwargs=kwargs)

        if isinstance(expr, L1Name):
            return L2Name(name=expr.name)

        if isinstance(expr, L1Constant):
            return L2Constant(value=expr.value)

        if isinstance(expr, L1Attribute):
            obj = self._convert_expr(expr.obj)
            return L2BinOp(
                op=".",
                lhs=obj,
                rhs=L2Name(name=expr.attr),
            )

        if isinstance(expr, L1Subscript):
            value = self._convert_expr(expr.value)
            slice_expr = self._convert_expr(expr.slice)
            return L2Subscript(value=value, slice=slice_expr)

        if isinstance(expr, L1Tuple):
            return L2Tuple(elts=tuple(self._convert_expr(e) for e in expr.elts))

        if isinstance(expr, L1DataPtr):
            tensor = expr.tensor
            name = tensor.source.pointer_string()
            return L2Name(name=name)

        if isinstance(expr, L1Offsets):
            tensor = expr.tensor
            dim = expr.dim
            name = f"{tensor.source.name}_offsets"
            if dim is not None:
                name += f"_{dim}"
            return L2Name(name=name)

        if isinstance(expr, L1Stride):
            tensor = expr.tensor
            dim = expr.dim
            name = tensor.source.stride_string(dim)
            return L2Name(name=name)

        if isinstance(expr, L1DtypeAttr):
            tensor = expr.tensor
            name = f"{tensor.source.pointer_string()}.type.element_ty"
            return L2Name(name=name)

        # Fallback
        return L2Constant(value=str(expr))

    def _convert_tensor_access(self, expr):
        """Convert L1TensorAccess to L2 expression.

        For scalar tensors (ndim=0): return the tensor name.
        For non-scalar: generate a load with pointers and mask.
        """
        tensor = expr.tensor

        if tensor.ndim == 0:
            return L2Name(name=tensor.source.name)

        # Generate load using Symbol-based computation
        pointer_sym, mask_sym = self._compute_pointers_and_mask(
            tensor, expr.indices
        )
        other = self._generate_other(tensor)

        # Store the load as an invariant and return a name reference
        load_name = f"_load_{id(expr) % 100000}"
        self._symbol_invariants[load_name] = call(
            "load", pointer_sym, mask=mask_sym, other=other
        )

        return L2Name(name=load_name)

    # --- Pointer & Mask Generation (delegates to CodeGenerator) ---

    def _compute_pointers_and_mask(self, tensor, l1_indices):
        """Compute pointers and mask using CodeGenerator's Symbol-based logic.

        Converts L1 index expressions to Symbol objects, delegates to
        CodeGenerator static methods, and collects invariants.

        :param tensor: Tensor object
        :param l1_indices: tuple of L1 IR index expressions
        :return: (Symbol, Symbol) - pointer expression and mask expression
        """
        # Convert L1 indices to Symbol objects
        symbol_indices = tuple(self._l1_index_to_symbol(idx) for idx in l1_indices)

        # Use CodeGenerator's _generate_pointers_and_mask logic
        # We need to provide invariants dict for pid/arange generation
        if tensor is not tensor.source:
            symbol_indices = self._complete_indices(tensor, symbol_indices)

        # Wrap in Symbol objects (CodeGenerator does this)
        symbol_indices = tuple(Symbol(idx) for idx in symbol_indices)

        name_for_pointers = CodeGenerator._name_for_pointers(tensor)
        self._symbol_invariants[name_for_pointers] = Symbol(
            tensor.source.pointer_string()
        )

        overall_offsets, mask = CodeGenerator._generate_overall_offsets_and_mask(
            tensor, symbol_indices
        )

        pointers = Symbol(name_for_pointers) + overall_offsets

        return pointers, mask

    def _complete_indices(self, tensor, indices):
        """Complete indices with pid and innermost indices.

        Delegates to CodeGenerator methods which work with Symbol objects.
        """
        return (
            tuple(self._generate_pid_indices(tensor))
            + tuple(indices)
            + tuple(self._generate_innermost_indices(tensor))
        )

    def _generate_pid_indices(self, tensor):
        """Generate program_id indices using CodeGenerator's approach.

        Returns Symbol-based index expressions.
        """
        self._symbol_invariants[_NAME_FOR_PID] = call("program_id", 0)

        indices = list(
            Tensor._unravel_index(Symbol(_NAME_FOR_PID), tensor.shape)
        )

        for dim, index in enumerate(indices):
            name = CodeGenerator._name_for_index(tensor, dim)
            self._symbol_invariants[name] = index
            indices[dim] = Symbol(name)

        return indices

    def _generate_innermost_indices(self, tensor):
        """Generate arange indices for the innermost level.

        Delegates to CodeGenerator._generate_innermost_indices.
        """
        return CodeGenerator._generate_innermost_indices(tensor)

    def _l1_index_to_symbol(self, idx):
        """Convert an L1 index expression to a Symbol-compatible value.

        Returns a value that can be wrapped in Symbol().
        """
        if isinstance(idx, L1Name):
            return idx.name
        if isinstance(idx, L1Constant):
            return idx.value
        if isinstance(idx, L1BinOp):
            left = self._l1_index_to_symbol(idx.lhs)
            right = self._l1_index_to_symbol(idx.rhs)
            return Symbol(left) + Symbol(right) if idx.op == "+" else Symbol(left)
        if isinstance(idx, L1Subscript):
            return self._l1_subscript_to_str(idx)
        if isinstance(idx, L1Call):
            return self._l1_call_to_str(idx)
        if isinstance(idx, L1Attribute):
            obj = self._l1_index_to_symbol(idx.obj)
            return f"{obj}.{idx.attr}"
        if isinstance(idx, L1Tuple):
            return "(" + ", ".join(self._l1_index_to_symbol(e) for e in idx.elts) + ")"
        # Fallback: string representation
        return str(idx)

    def _l1_subscript_to_str(self, idx):
        """Convert L1Subscript to a string that Symbol can parse."""
        value = self._l1_index_to_symbol(idx.value)
        if isinstance(idx.slice, L1Tuple):
            slice_str = ", ".join(self._l1_index_to_symbol(e) for e in idx.slice.elts)
            return f"{value}[{slice_str}]"
        slice_str = self._l1_index_to_string(idx.slice)
        return f"{value}[{slice_str}]"

    def _l1_call_to_str(self, idx):
        """Convert L1Call to a string that Symbol can parse."""
        args = ", ".join(self._l1_index_to_string(a) for a in idx.args)
        kwargs = ", ".join(
            f"{k}={self._l1_index_to_string(v)}" for k, v in idx.kwargs.items()
        )
        parts = [f"{idx.func}({args}"]
        if kwargs:
            parts.append(f", {kwargs}")
        parts.append(")")
        return "".join(parts)

    def _l1_index_to_string(self, idx):
        """Convert L1 expression to a string representation."""
        if isinstance(idx, L1Name):
            return idx.name
        if isinstance(idx, L1Constant):
            return repr(idx.value)
        if isinstance(idx, L1BinOp):
            left = self._l1_index_to_string(idx.lhs)
            right = self._l1_index_to_string(idx.rhs)
            return f"({left} {idx.op} {right})"
        if isinstance(idx, L1Subscript):
            return self._l1_subscript_to_str(idx)
        if isinstance(idx, L1Call):
            return self._l1_call_to_str(idx)
        if isinstance(idx, L1Attribute):
            obj = self._l1_index_to_string(idx.obj)
            return f"{obj}.{idx.attr}"
        if isinstance(idx, L1Tuple):
            return "(" + ", ".join(self._l1_index_to_string(e) for e in idx.elts) + ")"
        return str(idx)

    def _generate_other(self, tensor):
        """Generate the 'other' (fill) value for out-of-bounds access."""
        other = tensor.source.other
        if isinstance(other, float) and not math.isfinite(other):
            return L2Constant(value=f"float('{other}')")
        return L2Constant(value=other) if other is not None else None

    # --- Symbol → L2 IR conversion ---

    def _symbol_to_l2(self, sym):
        """Convert a Symbol (or plain value) to an L2 IR expression."""
        if sym is None:
            return None

        if isinstance(sym, Symbol):
            return self._ast_to_l2(sym.node)

        if isinstance(sym, (int, float, bool, str)):
            return L2Constant(value=sym)

        return L2Constant(value=str(sym))

    def _ast_to_l2(self, node):
        """Convert an AST node to an L2 IR expression."""
        if node is None:
            return None

        if isinstance(node, ast.Name):
            return L2Name(name=node.id)

        if isinstance(node, ast.Constant):
            return L2Constant(value=node.value)

        if isinstance(node, ast.BinOp):
            lhs = self._ast_to_l2(node.left)
            rhs = self._ast_to_l2(node.right)
            op = self._ast_op_to_str(node.op)
            return L2BinOp(op=op, lhs=lhs, rhs=rhs)

        if isinstance(node, ast.UnaryOp):
            operand = self._ast_to_l2(node.operand)
            op = self._ast_unaryop_to_str(node.op)
            if op == "-":
                return L2BinOp(op=op, lhs=L2Constant(value=0), rhs=operand)
            return L2BinOp(op=op, lhs=L2Constant(value=None), rhs=operand)

        if isinstance(node, ast.Compare):
            left = self._ast_to_l2(node.left)
            ops = []
            comparators = [self._ast_to_l2(c) for c in node.comparators]
            result = left
            for op_node, right in zip(node.ops, comparators):
                op = self._ast_cmpop_to_str(op_node)
                result = L2Compare(op=op, left=result, right=right)
            return result

        if isinstance(node, ast.BoolOp):
            values = [self._ast_to_l2(v) for v in node.values]
            op = "and" if isinstance(node.op, ast.And) else "or"
            return L2BoolOp(op=op, values=values)

        if isinstance(node, ast.Call):
            func_name = ast.unparse(node.func)
            args = tuple(self._ast_to_l2(a) for a in node.args)
            kwargs = {}
            for kw in node.keywords:
                if kw.arg:
                    kwargs[kw.arg] = self._ast_to_l2(kw.value)
            return L2Call(func=func_name, args=args, kwargs=kwargs)

        if isinstance(node, ast.Subscript):
            value = self._ast_to_l2(node.value)
            slice_expr = self._ast_to_l2(node.slice)
            return L2Subscript(value=value, slice=slice_expr)

        if isinstance(node, ast.Tuple):
            return L2Tuple(elts=tuple(self._ast_to_l2(e) for e in node.elts))

        if isinstance(node, ast.IfExp):
            test = self._ast_to_l2(node.test)
            body = self._ast_to_l2(node.body)
            orelse = self._ast_to_l2(node.orelse)
            return L2IfExp(test=test, body=body, orelse=orelse)

        # Fallback: convert to string
        return L2Name(name=ast.unparse(node))

    @staticmethod
    def _ast_op_to_str(op):
        ops = {
            ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/",
            ast.FloorDiv: "//", ast.Mod: "%", ast.Pow: "**",
            ast.LShift: "<<", ast.RShift: ">>",
            ast.BitOr: "|", ast.BitXor: "^", ast.BitAnd: "&",
        }
        return ops.get(type(op), type(op).__name__)

    @staticmethod
    def _ast_unaryop_to_str(op):
        ops = {ast.UAdd: "+", ast.USub: "-", ast.Not: "not", ast.Invert: "~"}
        return ops.get(type(op), type(op).__name__)

    @staticmethod
    def _ast_cmpop_to_str(op):
        ops = {
            ast.Eq: "==", ast.NotEq: "!=",
            ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
            ast.Is: "is", ast.IsNot: "is not",
            ast.In: "in", ast.NotIn: "not in",
        }
        return ops.get(type(op), type(op).__name__)
