from ninetoothed.ir.triton_ir import (
    L3Arange,
    L3Assign,
    L3Autotune,
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

# Map L3 binop operators to Python syntax
_BINOP_SYNTAX = {
    "+": "+", "-": "-", "*": "*", "/": "/", "//": "//", "%": "%",
    "**": "**", "<<": "<<", ">>": ">>", "|": "|", "^": "^", "&": "&",
}

_CMPOP_SYNTAX = {
    "==": "==", "!=": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">=",
    "is": "is", "is not": "is not", "in": "in", "not in": "not in",
}

_BOOLOP_SYNTAX = {
    "and": " and ", "or": " or ",
}


class L3ToCodePass:
    """Transform Triton IR (L3) to Python source code (L4).

    Generates a Triton-compatible Python source string from L3 IR nodes.
    """

    def __init__(self, prettify=False):
        self._prettify = prettify
        self._indent_str = "    "

    def transform(self, l3_func):
        """Generate Python source code from L3Function.

        :param l3_func: L3Function
        :return: str (Python source code)
        """
        lines = []

        # Imports
        lines.append("import triton")
        lines.append("import triton.language")
        lines.append("")

        # Decorator
        if l3_func.autotune is not None:
            lines.append(self._format_autotune_decorator(l3_func.autotune))
        else:
            lines.append("@triton.jit")
        lines.append("")

        # Function signature
        params_str = self._format_params(l3_func.params)
        lines.append(f"def {l3_func.name}({params_str}):")

        # Invariants
        has_body = False
        for inv in l3_func.invariants:
            value_str = self._format_expr(inv.value)
            lines.append(f"{self._indent_str}{inv.target} = {value_str}")
            has_body = True

        # Body
        for stmt in l3_func.body:
            stmt_lines = self._format_stmt(stmt, 1)
            lines.extend(stmt_lines)
            has_body = True

        if not has_body:
            lines.append(f"{self._indent_str}pass")

        source = "\n".join(lines)

        if self._prettify:
            source = self._prettify_source(source)

        return source

    # --- Formatting ---

    def _format_params(self, params):
        parts = []
        for p in params:
            s = p.name
            if p.annotation:
                s += f": {p.annotation}"
            parts.append(s)
        return ", ".join(parts)

    def _format_autotune_decorator(self, autotune):
        """Format the autotune decorator."""
        lines = []
        lines.append("@triton.autotune(")
        lines.append(f"{self._indent_str}configs=[")
        for i, cfg in enumerate(autotune.configs):
            comma = "," if i < len(autotune.configs) - 1 else ""
            lines.append(f"{self._indent_str}{self._indent_str}{cfg!r}{comma}")
        lines.append(f"{self._indent_str}],")
        lines.append(f"{self._indent_str}key=[{', '.join(repr(k) for k in autotune.key)}],")
        lines.append(")")
        return "\n".join(lines)

    def _format_stmt(self, stmt, indent_level):
        """Format a statement as Python code lines."""
        indent = self._indent_str * indent_level

        if isinstance(stmt, L3Assign):
            value_str = self._format_expr(stmt.value)
            return [f"{indent}{stmt.target} = {value_str}"]

        if isinstance(stmt, L3Load):
            ptr_str = self._format_expr(stmt.pointer)
            mask_str = self._format_expr(stmt.mask)
            parts = [f"triton.language.load({ptr_str}, mask={mask_str}"]
            if stmt.other is not None:
                other_str = self._format_expr(stmt.other)
                parts.append(f", other={other_str}")
            parts.append(")")
            return [f"{indent}_load_result = {''.join(parts)}"]

        if isinstance(stmt, L3Store):
            ptr_str = self._format_expr(stmt.pointer)
            val_str = self._format_expr(stmt.value)
            mask_str = self._format_expr(stmt.mask)
            return [f"{indent}triton.language.store({ptr_str}, {val_str}, mask={mask_str})"]

        if isinstance(stmt, L3ExprStmt):
            expr_str = self._format_expr(stmt.expr)
            return [f"{indent}{expr_str}"]

        if isinstance(stmt, L3Return):
            if stmt.value is not None:
                return [f"{indent}return {self._format_expr(stmt.value)}"]
            return [f"{indent}return"]

        if isinstance(stmt, L3For):
            return self._format_for(stmt, indent_level)

        if isinstance(stmt, L3If):
            return self._format_if(stmt, indent_level)

        return [f"{indent}# unhandled: {type(stmt).__name__}"]

    def _format_for(self, stmt, indent_level):
        indent = self._indent_str * indent_level
        iter_str = self._format_expr(stmt.iter)
        lines = [f"{indent}for {stmt.target} in {iter_str}:"]
        for s in stmt.body:
            lines.extend(self._format_stmt(s, indent_level + 1))
        if not stmt.body:
            lines.append(f"{indent}{self._indent_str}pass")
        return lines

    def _format_if(self, stmt, indent_level):
        indent = self._indent_str * indent_level
        test_str = self._format_expr(stmt.test)
        lines = [f"{indent}if {test_str}:"]
        for s in stmt.body:
            lines.extend(self._format_stmt(s, indent_level + 1))
        if not stmt.body:
            lines.append(f"{indent}{self._indent_str}pass")
        if stmt.orelse:
            lines.append(f"{indent}else:")
            for s in stmt.orelse:
                lines.extend(self._format_stmt(s, indent_level + 1))
            if not stmt.orelse:
                lines.append(f"{indent}{self._indent_str}pass")
        return lines

    def _format_expr(self, expr):
        """Format an expression as a Python expression string."""
        if expr is None:
            return "None"

        if isinstance(expr, L3Name):
            return expr.name

        if isinstance(expr, L3Constant):
            return repr(expr.value)

        if isinstance(expr, L3BinOp):
            op = _BINOP_SYNTAX.get(expr.op, expr.op)
            lhs = self._format_expr(expr.lhs)
            rhs = self._format_expr(expr.rhs)
            return f"({lhs} {op} {rhs})"

        if isinstance(expr, L3Compare):
            op = _CMPOP_SYNTAX.get(expr.op, expr.op)
            left = self._format_expr(expr.left)
            right = self._format_expr(expr.right)
            return f"({left} {op} {right})"

        if isinstance(expr, L3BoolOp):
            sep = _BOOLOP_SYNTAX.get(expr.op, f" {expr.op} ")
            parts = [self._format_expr(v) for v in expr.values]
            return f"({sep.join(parts)})"

        if isinstance(expr, L3IfExp):
            test = self._format_expr(expr.test)
            body = self._format_expr(expr.body)
            orelse = self._format_expr(expr.orelse)
            return f"({body} if {test} else {orelse})"

        if isinstance(expr, L3Call):
            func = expr.func
            args = ", ".join(self._format_expr(a) for a in expr.args)
            kwargs = ", ".join(
                f"{k}={self._format_expr(v)}" for k, v in expr.kwargs.items()
            )
            parts = [f"{func}({args}"]
            if kwargs:
                parts.append(f", {kwargs}")
            parts.append(")")
            return "".join(parts)

        if isinstance(expr, L3ProgramId):
            return f"triton.language.program_id({expr.axis})"

        if isinstance(expr, L3Arange):
            start = self._format_expr(expr.start)
            end = self._format_expr(expr.end)
            return f"triton.language.arange({start}, {end})"

        if isinstance(expr, L3Subscript):
            value = self._format_expr(expr.value)
            if isinstance(expr.slice, L3Tuple):
                slice_str = ", ".join(self._format_expr(e) for e in expr.slice.elts)
                return f"{value}[{slice_str}]"
            return f"{value}[{self._format_expr(expr.slice)}]"

        if isinstance(expr, L3Tuple):
            return "(" + ", ".join(self._format_expr(e) for e in expr.elts) + ")"

        return str(expr)

    def _prettify_source(self, source):
        """Try to prettify source with ruff format."""
        try:
            import subprocess
            result = subprocess.run(
                ["ruff", "format", "-"],
                input=source,
                encoding="utf-8",
                capture_output=True,
            )
            if result.returncode == 0:
                return result.stdout
        except (FileNotFoundError, OSError):
            pass
        return source
