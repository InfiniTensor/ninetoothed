from ninetoothed.ir.base import IRNode


# --- Expressions ---

class L2Expr(IRNode):
    """Base class for L2 expressions."""
    pass


class L2PointerExpr(L2Expr):
    """Memory pointer expression: base_ptr + offset_0 + offset_1 + ..."""

    def __init__(self, base, offsets=None):
        self.base = base
        self.offsets = offsets if offsets is not None else []

    def _op_name(self):
        return "ptr"

    def _ssa_format(self, fmt):
        if id(self) in fmt._names:
            return
        base_str = fmt._emit_expr(self.base) if isinstance(self.base, IRNode) else repr(self.base)
        parts = [f"base={base_str}"]
        if self.offsets:
            offset_strs = [fmt._emit_expr(o) for o in self.offsets]
            parts.append(f"offsets=[{', '.join(offset_strs)}]")
        result = fmt._fresh()
        fmt._bind(self, result)
        fmt._lines.append(f"  {result} = ptr {', '.join(parts)}")


class L2OffsetTerm(L2Expr):
    """Offset term: stride * index or just index."""

    def __init__(self, index, stride=None):
        self.stride = stride
        self.index = index

    def _op_name(self):
        return "offset"

    def _ssa_format(self, fmt):
        if id(self) in fmt._names:
            return
        idx = fmt._emit_expr(self.index)
        if self.stride is not None:
            stride = fmt._emit_expr(self.stride) if isinstance(self.stride, IRNode) else repr(self.stride)
            result = fmt._fresh()
            fmt._bind(self, result)
            fmt._lines.append(f"  {result} = offset {stride} * {idx}")
        else:
            result = fmt._fresh()
            fmt._bind(self, result)
            fmt._lines.append(f"  {result} = offset {idx}")


class L2MaskExpr(L2Expr):
    """Mask expression: cond_0 & cond_1 & ..."""

    def __init__(self, conditions=None):
        self.conditions = conditions if conditions is not None else []

    def _op_name(self):
        return "mask"

    def _ssa_format(self, fmt):
        if id(self) in fmt._names:
            return
        cond_strs = [fmt._emit_expr(c) for c in self.conditions]
        result = fmt._fresh()
        fmt._bind(self, result)
        fmt._lines.append(f"  {result} = mask [{', '.join(cond_strs)}]")


class L2BinOp(L2Expr):
    """Binary operation."""

    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def _to_inline(self):
        lhs = self.lhs._to_inline() if isinstance(self.lhs, IRNode) else repr(self.lhs)
        rhs = self.rhs._to_inline() if isinstance(self.rhs, IRNode) else repr(self.rhs)
        # Flatten chained & operations for readability
        if self.op == "&":
            parts = self._collect_and_chain()
            return " & ".join(parts)
        return f"({lhs} {self.op} {rhs})"

    def _collect_and_chain(self):
        """Recursively collect operands of chained & operations."""
        parts = []
        node = self
        while True:
            lhs = node.lhs
            rhs = node.rhs
            if isinstance(rhs, L2BinOp) and rhs.op == "&":
                parts.append(lhs._to_inline() if isinstance(lhs, IRNode) else repr(lhs))
                node = rhs
            else:
                parts.append(lhs._to_inline() if isinstance(lhs, IRNode) else repr(lhs))
                parts.append(rhs._to_inline() if isinstance(rhs, IRNode) else repr(rhs))
                break
        return parts


class L2Compare(L2Expr):
    """Comparison."""

    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

    def _to_inline(self):
        left = self.left._to_inline() if isinstance(self.left, IRNode) else repr(self.left)
        right = self.right._to_inline() if isinstance(self.right, IRNode) else repr(self.right)
        return f"({left} {self.op} {right})"


class L2BoolOp(L2Expr):
    """Boolean operation."""

    def __init__(self, op, values):
        self.op = op
        self.values = values

    def _op_name(self):
        return self.op  # "and" or "or"

    def _ssa_format(self, fmt):
        if id(self) in fmt._names:
            return
        val_strs = [fmt._emit_expr(v) for v in self.values]
        result = fmt._fresh()
        fmt._bind(self, result)
        fmt._lines.append(f"  {result} = {self.op} {', '.join(val_strs)}")


class L2Call(L2Expr):
    """Function call."""

    def __init__(self, func, args=None, kwargs=None):
        self.func = func
        self.args = args if args is not None else ()
        self.kwargs = kwargs if kwargs is not None else {}

    def _op_name(self):
        return "call"

    def _to_inline(self):
        short_func = IRNode._shorten_name(self.func)
        parts = []
        for a in self.args:
            parts.append(a._to_inline() if isinstance(a, IRNode) else repr(a))
        for k, v in self.kwargs.items():
            parts.append(f"{k}={v._to_inline() if isinstance(v, IRNode) else repr(v)}")
        return f"{short_func}({', '.join(parts)})"


class L2Name(L2Expr):
    """Variable reference."""

    def __init__(self, name):
        self.name = name

    def _to_inline(self):
        return IRNode._shorten_name(self.name)


class L2Constant(L2Expr):
    """Literal value."""

    def __init__(self, value):
        self.value = value

    def _to_inline(self):
        return repr(self.value)


class L2Subscript(L2Expr):
    """Indexing."""

    def __init__(self, value, slice):
        self.value = value
        self.slice = slice

    def _to_inline(self):
        val = self.value._to_inline() if isinstance(self.value, IRNode) else repr(self.value)
        sl = self.slice._to_inline() if isinstance(self.slice, IRNode) else repr(self.slice)
        return f"{val}[{sl}]"


class L2Attribute(L2Expr):
    """Attribute access."""

    def __init__(self, obj, attr):
        self.obj = obj
        self.attr = attr

    def _op_name(self):
        return "attr"

    def _ssa_format(self, fmt):
        if id(self) in fmt._names:
            return
        obj = fmt._emit_expr(self.obj)
        result = fmt._fresh()
        fmt._bind(self, result)
        fmt._lines.append(f"  {result} = {obj}.{self.attr}")


class L2IfExp(L2Expr):
    """Ternary expression."""

    def __init__(self, test, body, orelse):
        self.test = test
        self.body = body
        self.orelse = orelse

    def _op_name(self):
        return "select"

    def _ssa_format(self, fmt):
        if id(self) in fmt._names:
            return
        test = fmt._emit_expr(self.test)
        body = fmt._emit_expr(self.body)
        orelse = fmt._emit_expr(self.orelse)
        result = fmt._fresh()
        fmt._bind(self, result)
        fmt._lines.append(f"  {result} = select {test}, {body}, {orelse}")


class L2Tuple(L2Expr):
    """Tuple expression."""

    def __init__(self, elts):
        self.elts = elts

    def _to_inline(self):
        parts = []
        for e in self.elts:
            parts.append(e._to_inline() if isinstance(e, IRNode) else repr(e))
        return "(" + ", ".join(parts) + ")"


# --- Statements ---

class L2Statement(IRNode):
    """Base class for L2 statements."""
    pass


class L2Load(L2Statement):
    """Memory load operation."""

    def __init__(self, pointer, mask=None, other=None):
        self.pointer = pointer
        self.mask = mask
        self.other = other

    def _op_name(self):
        return "load"


class L2Store(L2Statement):
    """Memory store operation."""

    def __init__(self, pointer, value, mask=None):
        self.pointer = pointer
        self.value = value
        self.mask = mask

    def _to_inline(self):
        ptr = self.pointer._to_inline() if isinstance(self.pointer, IRNode) else repr(self.pointer)
        val = self.value._to_inline() if isinstance(self.value, IRNode) else repr(self.value)
        parts = [ptr, val]
        if self.mask is not None:
            mask_str = self.mask._to_inline() if isinstance(self.mask, IRNode) else repr(self.mask)
            parts.append(f"mask={mask_str}")
        return f"store({', '.join(parts)})"


class L2Assign(L2Statement):
    """Assignment."""

    def __init__(self, target, value):
        self.target = target
        self.value = value


class L2ExprStmt(L2Statement):
    """Expression statement."""

    def __init__(self, expr):
        self.expr = expr


class L2Return(L2Statement):
    """Return statement."""

    def __init__(self, value=None):
        self.value = value


class L2For(L2Statement):
    """For loop."""

    def __init__(self, target, iter, body):
        self.target = target
        self.iter = iter
        self.body = body


class L2If(L2Statement):
    """If statement."""

    def __init__(self, test, body, orelse=None):
        self.test = test
        self.body = body
        self.orelse = orelse if orelse is not None else []


# --- Function & Params ---

class L2Param(IRNode):
    """Function parameter."""

    def __init__(self, name, is_constexpr=False, annotation=None):
        self.name = name
        self.is_constexpr = is_constexpr
        self.annotation = annotation


class L2Invariant(IRNode):
    """Pre-computed invariant assignment."""

    def __init__(self, target, value):
        self.target = target
        self.value = value


class L2Function(IRNode):
    """Top-level function."""

    def __init__(self, name, params=None, body=None, invariants=None, grid_expr=None):
        self.name = name
        self.params = params if params is not None else []
        self.body = body if body is not None else []
        self.invariants = invariants if invariants is not None else []
        self.grid_expr = grid_expr

    def _op_name(self):
        return "func"

    def _dump_body(self, indent):
        prefix = "  " * indent
        parts = [f"{prefix}name={self.name!r}"]

        if self.params:
            param_names = [IRNode._shorten_name(p.name) for p in self.params]
            parts.append(f"{prefix}params=[{', '.join(param_names)}]")

        if self.invariants:
            parts.append(f"{prefix}invariants=")
            for inv in self.invariants:
                val_str = inv.value._to_inline() if isinstance(inv.value, IRNode) else repr(inv.value)
                target_short = IRNode._shorten_name(inv.target)
                parts.append(f"{prefix}  {target_short} = {val_str}")

        if self.body:
            parts.append(f"{prefix}body=")
            for i, stmt in enumerate(self.body):
                val_str = stmt._to_inline() if isinstance(stmt, IRNode) else repr(stmt)
                parts.append(f"{prefix}  [{i}]={val_str}")

        if self.grid_expr is not None:
            parts.append(f"{prefix}grid_expr={self.grid_expr!r}")

        return "\n".join(parts)

    def _ssa_format(self, fmt):
        """Custom SSA format for L2Function — handles list-style invariants."""
        name = fmt._short(self.name)

        # Params
        param_strs = []
        for i, p in enumerate(self.params):
            p_name = f"%arg{i}"
            fmt._bind(p, p_name)
            # Record param name -> %argN mapping for body name resolution
            if hasattr(p, "name"):
                fmt._param_name_map[fmt._short(p.name)] = p_name
            if p.is_constexpr:
                param_strs.append(f"{p_name}: constexpr")
            elif p.annotation:
                param_strs.append(f"{p_name}: {p.annotation}")
            else:
                param_strs.append(p_name)

        fmt._lines.append(f"func @{name}({', '.join(param_strs)}) {{")

        # Invariants (L2 uses a list of L2Invariant objects)
        if self.invariants:
            fmt._lines.append("  // invariants")
            for inv in self.invariants:
                target_short = fmt._short(inv.target)
                val = fmt._emit_expr(inv.value)
                fmt._lines.append(f"  {target_short} = {val}")

        # Body
        if self.body:
            fmt._lines.append("  // body")
            for stmt in self.body:
                fmt._format_node(stmt)

        # Grid expr
        if self.grid_expr is not None:
            fmt._lines.append(f"  grid = {self.grid_expr!r}")

        fmt._lines.append("}")
