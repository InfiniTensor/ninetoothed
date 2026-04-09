from ninetoothed.ir.base import IRNode


# --- Expressions ---

class L3Expr(IRNode):
    """Base class for L3 expressions."""
    pass


class L3ProgramId(L3Expr):
    """program_id(axis) call."""

    def __init__(self, axis):
        self.axis = axis

    def _to_inline(self):
        return f"program_id({self.axis})"


class L3Arange(L3Expr):
    """arange(start, end) call."""

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def _to_inline(self):
        start = self.start._to_inline() if isinstance(self.start, IRNode) else repr(self.start)
        end = self.end._to_inline() if isinstance(self.end, IRNode) else repr(self.end)
        return f"arange({start}, {end})"


class L3BinOp(L3Expr):
    """Binary operation."""

    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def _to_inline(self):
        lhs = self.lhs._to_inline() if isinstance(self.lhs, IRNode) else repr(self.lhs)
        rhs = self.rhs._to_inline() if isinstance(self.rhs, IRNode) else repr(self.rhs)
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
            if isinstance(rhs, L3BinOp) and rhs.op == "&":
                parts.append(lhs._to_inline() if isinstance(lhs, IRNode) else repr(lhs))
                node = rhs
            else:
                parts.append(lhs._to_inline() if isinstance(lhs, IRNode) else repr(lhs))
                parts.append(rhs._to_inline() if isinstance(rhs, IRNode) else repr(rhs))
                break
        return parts


class L3Compare(L3Expr):
    """Comparison."""

    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

    def _to_inline(self):
        left = self.left._to_inline() if isinstance(self.left, IRNode) else repr(self.left)
        right = self.right._to_inline() if isinstance(self.right, IRNode) else repr(self.right)
        return f"({left} {self.op} {right})"


class L3BoolOp(L3Expr):
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


class L3Call(L3Expr):
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


class L3Name(L3Expr):
    """Variable reference."""

    def __init__(self, name):
        self.name = name

    def _to_inline(self):
        return IRNode._shorten_name(self.name)


class L3Constant(L3Expr):
    """Literal value."""

    def __init__(self, value):
        self.value = value

    def _to_inline(self):
        return repr(self.value)


class L3Subscript(L3Expr):
    """Indexing."""

    def __init__(self, value, slice):
        self.value = value
        self.slice = slice

    def _to_inline(self):
        val = self.value._to_inline() if isinstance(self.value, IRNode) else repr(self.value)
        sl = self.slice._to_inline() if isinstance(self.slice, IRNode) else repr(self.slice)
        return f"{val}[{sl}]"


class L3Attribute(L3Expr):
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


class L3IfExp(L3Expr):
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


class L3Tuple(L3Expr):
    """Tuple expression."""

    def __init__(self, elts):
        self.elts = elts

    def _to_inline(self):
        parts = []
        for e in self.elts:
            parts.append(e._to_inline() if isinstance(e, IRNode) else repr(e))
        return "(" + ", ".join(parts) + ")"


# --- Statements ---

class L3Statement(IRNode):
    """Base class for L3 statements."""
    pass


class L3Load(L3Statement):
    """Triton load operation."""

    def __init__(self, pointer, mask=None, other=None):
        self.pointer = pointer
        self.mask = mask
        self.other = other

    def _op_name(self):
        return "load"


class L3Store(L3Statement):
    """Triton store operation."""

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


class L3Assign(L3Statement):
    """Assignment."""

    def __init__(self, target, value):
        self.target = target
        self.value = value

    def _to_inline(self):
        target_str = IRNode._shorten_name(self.target)
        val_str = self.value._to_inline() if isinstance(self.value, IRNode) else repr(self.value)
        return f"{target_str} = {val_str}"


class L3ExprStmt(L3Statement):
    """Expression statement."""

    def __init__(self, expr):
        self.expr = expr


class L3Return(L3Statement):
    """Return statement."""

    def __init__(self, value=None):
        self.value = value


class L3For(L3Statement):
    """For loop."""

    def __init__(self, target, iter, body):
        self.target = target
        self.iter = iter
        self.body = body


class L3If(L3Statement):
    """If statement."""

    def __init__(self, test, body, orelse=None):
        self.test = test
        self.body = body
        self.orelse = orelse if orelse is not None else []


# --- Function & Params ---

class L3Param(IRNode):
    """Function parameter."""

    def __init__(self, name, is_constexpr=False, annotation=None):
        self.name = name
        self.is_constexpr = is_constexpr
        self.annotation = annotation


class L3Invariant(IRNode):
    """Pre-computed invariant assignment."""

    def __init__(self, target, value):
        self.target = target
        self.value = value


class L3Grid(IRNode):
    """Grid configuration."""

    def __init__(self, expr):
        self.expr = expr

    def _ssa_format(self, fmt):
        fmt._lines.append(f"  grid = {self.expr!r}")


class L3Autotune(IRNode):
    """Autotune configuration."""

    def __init__(self, configs=None, key=None):
        self.configs = configs if configs is not None else []
        self.key = key if key is not None else []

    def _dump_body(self, indent):
        prefix = "  " * indent
        parts = []
        if self.key:
            parts.append(f"{prefix}key={self.key!r}")
        if self.configs:
            parts.append(f"{prefix}configs=")
            for i, cfg in enumerate(self.configs):
                parts.append(f"{prefix}  [{i}]={cfg!r}")
        return "\n".join(parts)

    def _ssa_format(self, fmt):
        if self.key:
            fmt._lines.append(f"  // autotune key={self.key!r}")
        if self.configs:
            for i, cfg in enumerate(self.configs):
                fmt._lines.append(f"  // config[{i}]={cfg!r}")


class L3Function(IRNode):
    """Top-level Triton kernel function."""

    def __init__(
        self, name, params=None, body=None, invariants=None, grid=None, autotune=None
    ):
        self.name = name
        self.params = params if params is not None else []
        self.body = body if body is not None else []
        self.invariants = invariants if invariants is not None else []
        self.grid = grid
        self.autotune = autotune

    def _op_name(self):
        return "func"

    def _dump_body(self, indent):
        prefix = "  " * indent
        parts = [f"{prefix}name={self.name!r}"]

        if self.params:
            param_names = [IRNode._shorten_name(p.name) for p in self.params]
            parts.append(f"{prefix}params=[{', '.join(param_names)}]")

        if self.autotune is not None:
            parts.append(f"{prefix}autotune={self.autotune.dump(indent + 1)}")

        if self.grid is not None:
            parts.append(f"{prefix}grid={self.grid.dump(indent + 1)}")

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

        return "\n".join(parts)

    def _ssa_format(self, fmt):
        """Custom SSA format for L3Function — handles grid, autotune, invariants."""
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

        # Autotune
        if self.autotune is not None:
            fmt._format_node(self.autotune)

        # Grid
        if self.grid is not None:
            fmt._format_node(self.grid)

        # Invariants
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

        fmt._lines.append("}")
