from ninetoothed.ir.base import IRNode


# --- Metadata ---

class TileOp(IRNode):
    """One tile/expand/squeeze/etc. operation from Tensor._history."""

    def __init__(self, kind, args=None, kwargs=None):
        self.kind = kind
        self.args = args if args is not None else ()
        self.kwargs = kwargs if kwargs is not None else {}


# --- Expressions ---

class L1Expr(IRNode):
    """Base class for L1 expressions."""
    pass


class L1TensorAccess(L1Expr):
    """Tensor read/write access: tensor_name or tensor_name[i, j]."""

    def __init__(self, param_name, tensor=None, indices=None):
        self.param_name = param_name
        self.tensor = tensor
        self.indices = indices if indices is not None else ()

    def _op_name(self):
        return "tensor.access"

    def _ssa_format(self, fmt):
        if id(self) in fmt._names:
            return
        indices_str = ""
        if self.indices:
            idx_parts = [fmt._emit_expr(i) for i in self.indices]
            indices_str = "[" + ", ".join(idx_parts) + "]"
        # Resolve param_name to %argN if it matches a function parameter
        short_name = fmt._short(self.param_name)
        if short_name in fmt._param_name_map:
            result = fmt._param_name_map[short_name]
        else:
            result = f"%{short_name}"
        fmt._bind(self, result + indices_str)

    def _dump_body(self, indent):
        prefix = "  " * indent
        parts = [f'{prefix}param_name={self.param_name!r}']

        if self.tensor is not None:
            shape_str = ", ".join(str(s) for s in self.tensor.shape)
            parts.append(f"{prefix}tensor.shape=({shape_str})")

        if self.indices:
            idx_str = ", ".join(
                v.dump(indent) if isinstance(v, IRNode) else repr(v)
                for v in self.indices
            )
            parts.append(f"{prefix}indices=({idx_str})")
        else:
            parts.append(f"{prefix}indices=()")

        return "\n".join(parts)


class L1BinOp(L1Expr):
    """Binary operation: lhs + rhs, lhs * rhs, etc."""

    def __init__(self, op, lhs, rhs):
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def _to_inline(self):
        lhs = self.lhs._to_inline() if isinstance(self.lhs, IRNode) else repr(self.lhs)
        rhs = self.rhs._to_inline() if isinstance(self.rhs, IRNode) else repr(self.rhs)
        return f"({lhs} {self.op} {rhs})"


class L1UnaryOp(L1Expr):
    """Unary operation: -x, ~x."""

    def __init__(self, op, operand):
        self.op = op
        self.operand = operand

    def _op_name(self):
        if self.op == "~":
            return "not"
        return f"neg"

    def _to_inline(self):
        operand = self.operand._to_inline() if isinstance(self.operand, IRNode) else repr(self.operand)
        return f"{self.op}{operand}"

    def _ssa_format(self, fmt):
        if id(self) in fmt._names:
            return
        operand = fmt._emit_expr(self.operand)
        result = fmt._fresh()
        fmt._bind(self, result)
        op_name = self._op_name()
        fmt._lines.append(f"  {result} = {op_name} {operand}")


class L1Compare(L1Expr):
    """Comparison: x < y, x >= y, etc."""

    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

    def _to_inline(self):
        left = self.left._to_inline() if isinstance(self.left, IRNode) else repr(self.left)
        right = self.right._to_inline() if isinstance(self.right, IRNode) else repr(self.right)
        return f"({left} {self.op} {right})"


class L1BoolOp(L1Expr):
    """Boolean operation: x and y, x or y."""

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


class L1Call(L1Expr):
    """Function call: func(arg1, arg2, key=val)."""

    def __init__(self, func, args=None, kwargs=None):
        self.func = func
        self.args = args if args is not None else ()
        self.kwargs = kwargs if kwargs is not None else {}

    def _op_name(self):
        return "call"

    def _to_inline(self):
        parts = []
        for a in self.args:
            parts.append(a._to_inline() if isinstance(a, IRNode) else repr(a))
        for k, v in self.kwargs.items():
            parts.append(f"{k}={v._to_inline() if isinstance(v, IRNode) else repr(v)}")
        return f"{self.func}({', '.join(parts)})"


class L1Name(L1Expr):
    """Variable reference."""

    def __init__(self, name):
        self.name = name

    def _to_inline(self):
        return self.name


class L1Constant(L1Expr):
    """Literal value."""

    def __init__(self, value):
        self.value = value

    def _to_inline(self):
        return repr(self.value)


class L1Attribute(L1Expr):
    """Attribute access: obj.attr."""

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


class L1Subscript(L1Expr):
    """Indexing: value[slice]."""

    def __init__(self, value, slice):
        self.value = value
        self.slice = slice


class L1Tuple(L1Expr):
    """Tuple expression."""

    def __init__(self, elts):
        self.elts = elts


class L1IfExp(L1Expr):
    """Ternary expression: body if test else orelse."""

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


# --- Special Tensor Method Calls ---

class L1DataPtr(L1Expr):
    """tensor.data_ptr() call."""

    def __init__(self, tensor, param_name=None):
        self.tensor = tensor
        self.param_name = param_name

    def _op_name(self):
        return "data_ptr"

    def _ssa_format(self, fmt):
        if id(self) in fmt._names:
            return
        tensor_ref = self._resolve_tensor_ref(fmt)
        result = fmt._fresh()
        fmt._bind(self, result)
        fmt._lines.append(f"  {result} = data_ptr {tensor_ref}")

    def _dump_body(self, indent):
        prefix = "  " * indent
        return f"{prefix}tensor={self.tensor.source.name!r}"


class L1Offsets(L1Expr):
    """tensor.offsets(dim) call."""

    def __init__(self, tensor, dim=None, param_name=None):
        self.tensor = tensor
        self.dim = dim
        self.param_name = param_name

    def _op_name(self):
        return "offsets"

    def _ssa_format(self, fmt):
        if id(self) in fmt._names:
            return
        tensor_ref = self._resolve_tensor_ref(fmt)
        parts = [tensor_ref]
        if self.dim is not None:
            parts.append(repr(self.dim))
        result = fmt._fresh()
        fmt._bind(self, result)
        fmt._lines.append(f"  {result} = offsets({', '.join(parts)})")

    def _dump_body(self, indent):
        prefix = "  " * indent
        parts = [f"{prefix}tensor={self.tensor.source.name!r}"]
        if self.dim is not None:
            parts.append(f"{prefix}dim={self.dim!r}")
        return "\n".join(parts)


class L1Stride(L1Expr):
    """tensor.stride(dim) call."""

    def __init__(self, tensor, dim, param_name=None):
        self.tensor = tensor
        self.dim = dim
        self.param_name = param_name

    def _op_name(self):
        return "stride"

    def _ssa_format(self, fmt):
        if id(self) in fmt._names:
            return
        tensor_ref = self._resolve_tensor_ref(fmt)
        result = fmt._fresh()
        fmt._bind(self, result)
        fmt._lines.append(f"  {result} = stride({tensor_ref}, {self.dim!r})")

    def _dump_body(self, indent):
        prefix = "  " * indent
        return f"{prefix}tensor={self.tensor.source.name!r}\n{prefix}dim={self.dim!r}"


class L1DtypeAttr(L1Expr):
    """tensor.dtype (pointer element type)."""

    def __init__(self, tensor, param_name=None):
        self.tensor = tensor
        self.param_name = param_name

    def _op_name(self):
        return "dtype"

    def _ssa_format(self, fmt):
        if id(self) in fmt._names:
            return
        tensor_ref = self._resolve_tensor_ref(fmt)
        result = fmt._fresh()
        fmt._bind(self, result)
        fmt._lines.append(f"  {result} = dtype({tensor_ref})")

    def _dump_body(self, indent):
        prefix = "  " * indent
        return f"{prefix}tensor={self.tensor.source.name!r}"


# --- Statements ---

class L1Statement(IRNode):
    """Base class for L1 statements."""
    pass


class L1Assign(L1Statement):
    """Assignment: target = value."""

    def __init__(self, target, value):
        self.target = target
        self.value = value


class L1ExprStmt(L1Statement):
    """Expression statement."""

    def __init__(self, expr):
        self.expr = expr


class L1Return(L1Statement):
    """Return statement."""

    def __init__(self, value=None):
        self.value = value


class L1For(L1Statement):
    """For loop: for target in iter: body."""

    def __init__(self, target, iter, body):
        self.target = target
        self.iter = iter
        self.body = body


class L1If(L1Statement):
    """If statement: if test: body else: orelse."""

    def __init__(self, test, body, orelse=None):
        self.test = test
        self.body = body
        self.orelse = orelse if orelse is not None else []


# --- Function & Params ---

class L1TensorParam(IRNode):
    """Function parameter annotated with a Tensor type (includes tiling info)."""

    def __init__(
        self,
        name,
        tensor=None,
        tile_history=None,
        ndim=None,
        dtype=None,
        other=None,
        jagged_dim=None,
    ):
        self.name = name
        self.tensor = tensor
        self.tile_history = tile_history if tile_history is not None else []
        self.ndim = ndim if ndim is not None else (tensor.ndim if tensor else 0)
        self.dtype = dtype
        self.other = other
        self.jagged_dim = jagged_dim

    def _dump_body(self, indent):
        prefix = "  " * indent
        parts = [f"{prefix}name={self.name!r}"]
        parts.append(f"{prefix}ndim={self.ndim}")

        if self.dtype is not None:
            parts.append(f"{prefix}dtype={self.dtype!r}")

        if self.other is not None:
            parts.append(f"{prefix}other={self.other!r}")

        if self.jagged_dim is not None:
            parts.append(f"{prefix}jagged_dim={self.jagged_dim!r}")

        if self.tile_history:
            parts.append(f"{prefix}tile_history=")
            for i, op in enumerate(self.tile_history):
                parts.append(f"{prefix}  [{i}]={op.dump(indent + 2)}")

        if self.tensor is not None:
            shape_str = ", ".join(str(s) for s in self.tensor.shape)
            parts.append(f"{prefix}shape=({shape_str})")
            innermost = self.tensor.innermost()
            inner_shape_str = ", ".join(str(s) for s in innermost.shape)
            parts.append(f"{prefix}innermost_shape=({inner_shape_str})")

        return "\n".join(parts)


class L1Function(IRNode):
    """Top-level kernel function."""

    def __init__(self, name, params=None, body=None, invariants=None):
        self.name = name
        self.params = params if params is not None else []
        self.body = body if body is not None else []
        self.invariants = invariants if invariants is not None else {}

    def _op_name(self):
        return "func"

    def _dump_body(self, indent):
        prefix = "  " * indent
        parts = [f"{prefix}name={self.name!r}"]

        if self.params:
            parts.append(f"{prefix}params=")
            for i, p in enumerate(self.params):
                parts.append(f"{prefix}  [{i}]={p.dump(indent + 2)}")

        if self.invariants:
            parts.append(f"{prefix}invariants=")
            for k, v in self.invariants.items():
                val_str = v.dump(indent + 2) if isinstance(v, IRNode) else repr(v)
                parts.append(f"{prefix}  {k} = {val_str}")

        if self.body:
            parts.append(f"{prefix}body=")
            for i, stmt in enumerate(self.body):
                parts.append(f"{prefix}  [{i}]={stmt.dump(indent + 2)}")

        return "\n".join(parts)

    def _ssa_format(self, fmt):
        """Custom SSA format for L1Function — handles dict-style invariants."""
        name = fmt._short(self.name)

        # Prefixes to shorten in shape/tile expressions
        _verbose_prefixes = (
            "ninetoothed_ninetoothed_",
            "ninetoothed_constexpr_prefix_",
            "ninetoothed_next_power_of_2_prefix_ninetoothed_constexpr_prefix_",
            "triton_next_power_of_2_prefix_triton_constexpr_prefix_",
            "triton_triton_",
            "triton_constexpr_prefix_",
            "triton_next_power_of_2_prefix_triton_constexpr_prefix_",
            "ninetoothed_tensor_",
            "triton_tensor_",
        )

        def _shorten_expr(s):
            """Shorten verbose prefixes in an expression string."""
            for pfx in _verbose_prefixes:
                s = s.replace(pfx, "")
            return s

        def _format_shape(shape):
            """Format a shape tuple into a shortened string."""
            raw = ", ".join(repr(s) for s in shape)
            return _shorten_expr(raw)

        def _format_tile_op(op):
            """Format a TileOp into a compact string like tile((BLOCK_SIZE,))."""
            args_str = ", ".join(_shorten_expr(repr(a)) for a in op.args)
            if op.kwargs:
                kw_parts = [f"{k}={_shorten_expr(repr(v))}" for k, v in op.kwargs.items()]
                return f"{op.kind}({args_str}, {', '.join(kw_parts)})"
            return f"{op.kind}({args_str})"

        # Params
        param_strs = []
        for i, p in enumerate(self.params):
            p_name = f"%arg{i}"
            fmt._bind(p, p_name)
            # Record param name -> %argN mapping for body name resolution
            if hasattr(p, "name"):
                fmt._param_name_map[fmt._short(p.name)] = p_name
            if hasattr(p, "tensor") and p.tensor is not None:
                try:
                    shape_str = _format_shape(p.tensor.shape)
                except Exception:
                    shape_str = str(p.ndim) + "d"

                # Tile info: inner shape and tile history
                tile_parts = []
                try:
                    innermost = p.tensor.innermost()
                    inner_shape = _format_shape(innermost.shape)
                    tile_parts.append(f"inner={inner_shape}")
                except Exception:
                    pass

                if p.tile_history:
                    ops_str = " -> ".join(_format_tile_op(op) for op in p.tile_history)
                    tile_parts.append(ops_str)

                tile_str = f", tile=<{', '.join(tile_parts)}>" if tile_parts else ""

                # Only show dtype if it's a simple type (not a nested Tensor)
                dtype_val = getattr(p, "dtype", None)
                if dtype_val is not None and not hasattr(dtype_val, "shape"):
                    dtype_str = f", {dtype_val}"
                else:
                    dtype_str = ""
                param_strs.append(f"{p_name}: tensor<{shape_str}{dtype_str}{tile_str}>")
            elif hasattr(p, "ndim"):
                param_strs.append(f"{p_name}: tensor<{p.ndim}d>")
            else:
                param_strs.append(p_name)

        fmt._lines.append(f"func @{name}({', '.join(param_strs)}) {{")

        # Invariants (L1 uses a dict)
        if self.invariants:
            fmt._lines.append("  // invariants")
            for k, v in self.invariants.items():
                val = fmt._emit_expr(v)
                fmt._lines.append(f"  %{fmt._short(k)} = {val}")

        # Body
        if self.body:
            fmt._lines.append("  // body")
            for stmt in self.body:
                fmt._format_node(stmt)

        fmt._lines.append("}")


# --- Mixin: shared _resolve_tensor_ref for tensor method nodes ---

def _resolve_tensor_ref(self, fmt):
    """Resolve a tensor parameter reference to its SSA name.

    Uses param_name (if available) to look up %argN in the formatter's
    parameter map.  Falls back to tensor.source.name when no param_name
    is provided.
    """
    if self.param_name is not None:
        short = fmt._short(self.param_name)
        if short in fmt._param_name_map:
            return fmt._param_name_map[short]
        return f"%{short}"
    tensor_name = self.tensor.source.name if hasattr(self.tensor, "source") else "?"
    return f"%{fmt._short(tensor_name)}"


L1DataPtr._resolve_tensor_ref = _resolve_tensor_ref
L1Offsets._resolve_tensor_ref = _resolve_tensor_ref
L1Stride._resolve_tensor_ref = _resolve_tensor_ref
L1DtypeAttr._resolve_tensor_ref = _resolve_tensor_ref
