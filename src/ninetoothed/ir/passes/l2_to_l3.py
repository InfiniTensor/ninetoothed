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
    L2OffsetTerm,
    L2Param,
    L2PointerExpr,
    L2Return,
    L2Store,
    L2Subscript,
    L2Tuple,
)
from ninetoothed.ir.triton_ir import (
    L3Arange,
    L3Assign,
    L3BinOp,
    L3BoolOp,
    L3Call,
    L3Compare,
    L3Constant,
    L3ExprStmt,
    L3For,
    L3Function,
    L3Grid,
    L3If,
    L3IfExp,
    L3Invariant,
    L3Load,
    L3Name,
    L3Param,
    L3ProgramId,
    L3Return,
    L3Store,
    L3Subscript,
    L3Tuple,
)


class L2ToL3Pass:
    """Transform Memory IR (L2) to Triton IR (L3).

    Handles:
    - Converting L2 nodes to L3 nodes
    - Replacing 'ninetoothed' -> 'triton' in all name references
    - Converting program_id and arange calls to dedicated nodes
    """

    def __init__(self):
        pass

    def transform(self, l2_func):
        """Transform L2Function to L3Function.

        :param l2_func: L2Function
        :return: L3Function
        """
        l3_params = [self._convert_param(p) for p in l2_func.params]
        l3_invariants = [self._convert_invariant(inv) for inv in l2_func.invariants]
        l3_body = []
        for stmt in l2_func.body:
            result = self._visit_stmt(stmt)
            if result is not None:
                if isinstance(result, list):
                    l3_body.extend(result)
                else:
                    l3_body.append(result)

        grid = L3Grid(expr=l2_func.grid_expr) if l2_func.grid_expr else None

        return L3Function(
            name=l2_func.name,
            params=l3_params,
            body=l3_body,
            invariants=l3_invariants,
            grid=grid,
        )

    # --- Param & Invariant ---

    def _convert_param(self, l2_param):
        return L3Param(
            name=l2_param.name,
            is_constexpr=l2_param.is_constexpr,
            annotation=l2_param.annotation,
        )

    def _convert_invariant(self, l2_inv):
        value = self._convert_expr(l2_inv.value)
        return L3Invariant(target=l2_inv.target, value=value)

    # --- Statement visitors ---

    def _visit_stmt(self, stmt):
        if isinstance(stmt, L2Load):
            return self._visit_Load(stmt)
        if isinstance(stmt, L2Store):
            return self._visit_Store(stmt)
        if isinstance(stmt, L2Assign):
            target = stmt.target
            value = self._convert_expr(stmt.value)
            return L3Assign(target=target, value=value)
        if isinstance(stmt, L2ExprStmt):
            return L3ExprStmt(expr=self._convert_expr(stmt.expr))
        if isinstance(stmt, L2Return):
            value = self._convert_expr(stmt.value) if stmt.value else None
            return L3Return(value)
        if isinstance(stmt, L2For):
            return self._visit_For(stmt)
        if isinstance(stmt, L2If):
            return self._visit_If(stmt)
        return None

    def _visit_Load(self, stmt):
        pointer = self._convert_expr(stmt.pointer)
        mask = self._convert_expr(stmt.mask)
        other = self._convert_expr(stmt.other) if stmt.other else None
        return L3Load(pointer=pointer, mask=mask, other=other)

    def _visit_Store(self, stmt):
        pointer = self._convert_expr(stmt.pointer)
        value = self._convert_expr(stmt.value)
        mask = self._convert_expr(stmt.mask)
        return L3Store(pointer=pointer, value=value, mask=mask)

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
        return L3For(target=stmt.target, iter=iter_expr, body=body)

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
        return L3If(test=test, body=body, orelse=orelse)

    # --- Expression visitors ---

    def _convert_expr(self, expr):
        if expr is None:
            return None

        if isinstance(expr, L2Name):
            name = self._replace_ninetoothed(expr.name)
            return L3Name(name=name)

        if isinstance(expr, L2Constant):
            return L3Constant(value=expr.value)

        if isinstance(expr, L2BinOp):
            lhs = self._convert_expr(expr.lhs)
            rhs = self._convert_expr(expr.rhs)
            return L3BinOp(op=expr.op, lhs=lhs, rhs=rhs)

        if isinstance(expr, L2Compare):
            left = self._convert_expr(expr.left)
            right = self._convert_expr(expr.right)
            return L3Compare(op=expr.op, left=left, right=right)

        if isinstance(expr, L2BoolOp):
            values = [self._convert_expr(v) for v in expr.values]
            return L3BoolOp(op=expr.op, values=values)

        if isinstance(expr, L2IfExp):
            test = self._convert_expr(expr.test)
            body = self._convert_expr(expr.body)
            orelse = self._convert_expr(expr.orelse)
            return L3IfExp(test=test, body=body, orelse=orelse)

        if isinstance(expr, L2Call):
            func_name = self._replace_ninetoothed(expr.func)

            # Detect special calls
            if func_name == "program_id":
                if expr.args:
                    axis_arg = expr.args[0]
                    if isinstance(axis_arg, L2Constant):
                        return L3ProgramId(axis=axis_arg.value)
                    elif isinstance(axis_arg, L3Constant):
                        return L3ProgramId(axis=axis_arg.value)

            if func_name == "arange":
                start = self._convert_expr(expr.args[0]) if expr.args else L3Constant(0)
                end = self._convert_expr(expr.args[1]) if len(expr.args) > 1 else None
                return L3Arange(start=start, end=end)

            args = tuple(self._convert_expr(a) for a in expr.args)
            kwargs = {k: self._convert_expr(v) for k, v in expr.kwargs.items()}
            return L3Call(func=func_name, args=args, kwargs=kwargs)

        if isinstance(expr, L2Subscript):
            value = self._convert_expr(expr.value)
            slice_expr = self._convert_expr(expr.slice)
            return L3Subscript(value=value, slice=slice_expr)

        if isinstance(expr, L2Tuple):
            return L3Tuple(elts=tuple(self._convert_expr(e) for e in expr.elts))

        if isinstance(expr, L2PointerExpr):
            # Flatten pointer expression to a sum
            base = L3Name(name=self._replace_ninetoothed(expr.base))
            result = base
            for offset_term in expr.offsets:
                if isinstance(offset_term, L2OffsetTerm):
                    idx = self._convert_expr(offset_term.index)
                    if offset_term.stride is not None:
                        stride = self._convert_expr(offset_term.stride)
                        term = L3BinOp(op="*", lhs=stride, rhs=idx)
                    else:
                        term = idx
                    result = L3BinOp(op="+", lhs=result, rhs=term)
            return result

        if isinstance(expr, L2MaskExpr):
            # Flatten mask conditions with &
            if not expr.conditions:
                return L3Constant(value=True)
            result = self._convert_expr(expr.conditions[0])
            for cond in expr.conditions[1:]:
                result = L3BinOp(op="&", lhs=result, rhs=self._convert_expr(cond))
            return result

        # Fallback: try to get a name representation
        if hasattr(expr, "name"):
            return L3Name(name=self._replace_ninetoothed(expr.name))

        return L3Constant(value=str(expr))

    # --- Helpers ---

    @staticmethod
    def _replace_ninetoothed(name):
        """Replace 'ninetoothed' with 'triton' in a name string."""
        if isinstance(name, str):
            return name.replace("ninetoothed", "triton")
        return name
