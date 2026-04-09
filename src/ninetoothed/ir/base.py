import re


class IRNode:
    """Base class for all IR nodes.

    Provides structured text dump (S-expression style) and equality comparison.
    """

    def dump(self, indent=0) -> str:
        """Dump this node as structured text (S-expression style).

        Subclasses should override _dump_body() to customize output.
        """
        prefix = "  " * indent
        lines = [f"{prefix}({self.__class__.__name__}"]
        body = self._dump_body(indent + 1)
        if body:
            lines.append(body)
        lines.append(f"{prefix})")
        return "\n".join(lines)

    def dump_ssa(self) -> str:
        """Dump this node in MLIR-style SSA format.

        Produces flat, linear output where sub-expressions are assigned
        to %N names and referenced by name.
        """
        return SSAFormatter().format(self)

    def _to_inline(self):
        """Render this node as a compact single-line string.

        Subclasses should override to produce human-readable inline form.
        """
        parts = []
        for name, value in self._iter_fields():
            if value is None:
                continue
            if isinstance(value, IRNode):
                parts.append(value._to_inline())
            elif isinstance(value, (list, tuple)):
                if all(isinstance(v, IRNode) for v in value):
                    parts.append("[" + ", ".join(v._to_inline() for v in value) + "]")
                else:
                    parts.append(repr(value))
            else:
                parts.append(repr(value))
        return "(" + " ".join(parts) + ")"

    def _op_name(self):
        """Return the operation name for SSA formatting.

        Subclasses should override to customize the op name used in
        SSA output (e.g., L1BinOp with op='+' returns 'add').
        """
        return self.__class__.__name__

    def _ssa_attrs(self):
        """Return a dict of extra attributes for SSA formatting.

        Subclasses can override to add type annotations, etc.
        """
        return {}

    @staticmethod
    def _shorten_name(name):
        """Shorten verbose internal names for dump readability."""
        if not isinstance(name, str):
            return str(name)
        # ninetoothed_ninetoothed_tensor_0_size_0 -> tensor_0_size_0
        # ninetoothed_constexpr_prefix_BLOCK_SIZE -> BLOCK_SIZE
        # triton_next_power_of_2_prefix_triton_constexpr_prefix_BLOCK_SIZE -> BLOCK_SIZE
        # triton_triton_tensor_0_pointer -> tensor_0_pointer
        # ninetoothed_tensor_0_index_0 -> tensor_0_index_0
        # triton_constexpr_prefix_BLOCK_SIZE -> BLOCK_SIZE
        # triton_next_power_of_2_prefix_triton_constexpr_prefix_BLOCK_SIZE -> BLOCK_SIZE
        for prefix in (
            "ninetoothed_ninetoothed_",
            "ninetoothed_constexpr_prefix_",
            "ninetoothed_next_power_of_2_prefix_ninetoothed_constexpr_prefix_",
            "triton_next_power_of_2_prefix_triton_constexpr_prefix_",
            "triton_triton_",
            "triton_constexpr_prefix_",
            "triton_next_power_of_2_prefix_triton_constexpr_prefix_",
            "ninetoothed_tensor_",
            "triton_tensor_",
            "ninetoothed_pid",
            "triton_pid",
        ):
            if name.startswith(prefix):
                if name == "ninetoothed_pid":
                    return "pid"
                if name == "triton_pid":
                    return "pid"
                return name[len(prefix):]
        return name

    def _dump_body(self, indent) -> str:
        """Override in subclasses to customize dump output."""
        prefix = "  " * indent
        parts = []

        for name, value in self._iter_fields():
            if value is None:
                continue

            if isinstance(value, IRNode):
                parts.append(f"{prefix}{name}={value.dump(indent)}")
            elif isinstance(value, (list, tuple)) and all(
                isinstance(v, IRNode) for v in value
            ):
                if not value:
                    parts.append(f"{prefix}{name}=()")
                else:
                    for i, v in enumerate(value):
                        parts.append(f"{prefix}{name}[{i}]={v.dump(indent)}")
            elif isinstance(value, dict):
                # Handle kwargs dicts - use _to_inline for IRNode values
                items = []
                for k, v in value.items():
                    if isinstance(v, IRNode):
                        items.append(f"{k!r}: {v._to_inline()}")
                    else:
                        items.append(f"{k!r}: {v!r}")
                parts.append(f"{prefix}{name}={{{', '.join(items)}}}")
            elif isinstance(value, (list, tuple)):
                parts.append(f"{prefix}{name}={value!r}")
            else:
                parts.append(f"{prefix}{name}={value!r}")

        return "\n".join(parts)

    def _iter_fields(self):
        """Yield (field_name, field_value) pairs for all non-private attributes."""
        for name in vars(self):
            if not name.startswith("_"):
                yield name, getattr(self, name)

    def __eq__(self, other):
        if type(self) is not type(other):
            return False

        for name, value in self._iter_fields():
            other_value = getattr(other, name, None)
            if value != other_value:
                return False

        return True

    def __repr__(self):
        return self.dump()

    def __hash__(self):
        return id(self)


# ---------------------------------------------------------------------------
# BinOp symbolic-name mappings (shared across all IR levels)
# ---------------------------------------------------------------------------
_BINOP_SYMBOLS = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
    "//": "floordiv",
    "%": "mod",
    "**": "pow",
    "<<": "shl",
    ">>": "shr",
    "&": "and",
    "|": "or",
    "^": "xor",
}

_COMPARE_SYMBOLS = {
    "==": "eq",
    "!=": "ne",
    "<": "lt",
    "<=": "le",
    ">": "gt",
    ">=": "ge",
}


class SSAFormatter:
    """Formats IRNode trees in MLIR-style SSA form.

    Complex sub-expressions are hoisted and assigned to ``%N`` temporaries
    so that every definition is a flat, single-line statement.
    """

    def __init__(self):
        self._counter = 0
        self._names: dict[int, str] = {}  # id(node) -> ssa name
        self._lines: list[str] = []
        self._param_name_map: dict[str, str] = {}  # param name -> %argN
        self._def_map: dict[str, str] = {}  # variable name -> latest %N

    # -- public entry --------------------------------------------------------

    def format(self, node) -> str:
        """Format *node* and return the SSA text."""
        self._lines = []
        self._format_node(node)
        return "\n".join(self._lines)

    # -- name allocation -----------------------------------------------------

    def _fresh(self) -> str:
        name = f"%{self._counter}"
        self._counter += 1
        return name

    def _name_of(self, node) -> str | None:
        return self._names.get(id(node))

    def _bind(self, node, name: str | None):
        if name is not None:
            self._names[id(node)] = name

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _short(name):
        return IRNode._shorten_name(name)

    @staticmethod
    def _is_simple(node) -> bool:
        """Return True if *node* can be inlined without hoisting."""
        return isinstance(node, (SSAName, SSAConstant))

    def _inline(self, node) -> str:
        """Return the inline SSA text for *node*.

        If the node was already hoisted, return its SSA name.
        Otherwise return its inline representation.
        """
        name = self._name_of(node)
        if name is not None:
            return name
        return node._to_inline()

    def _resolve_names_in_str(self, text) -> str:
        """Replace parameter / variable names in *text* with SSA forms.

        Uses word-boundary matching to avoid replacing substrings
        (e.g. ``scale`` inside ``scaled``).
        """
        # Build combined map: params take priority over defs
        combined = {**self._def_map, **self._param_name_map}
        if not combined:
            return text
        # Sort keys by length (longest first) to avoid partial matches.
        for key in sorted(combined, key=len, reverse=True):
            text = re.sub(r'\b' + re.escape(key) + r'\b',
                          combined[key], text)
        return text

    # -- core recursive formatting -------------------------------------------

    def _format_node(self, node):
        """Dispatch to the appropriate handler."""
        method = getattr(node, "_ssa_format", None)
        if method is not None:
            method(self)
            return

        # Generic fallback — works for any IRNode
        self._format_generic(node)

    def _format_generic(self, node):
        """Generic SSA formatting for nodes without a custom _ssa_format."""
        # If already bound, skip (already emitted)
        if id(node) in self._names:
            return

        cls_name = type(node).__name__

        # Detect BinOp-like pattern (has op, lhs, rhs)
        if hasattr(node, "op") and hasattr(node, "lhs") and hasattr(node, "rhs"):
            # Check if this is actually a comparison operator (common pattern
            # where BinOp is used for comparisons in some IR levels)
            op_val = getattr(node, "op", None)
            if op_val in _COMPARE_SYMBOLS:
                self._format_compare_like(node)
            else:
                self._format_binop_like(node)
            return

        # Detect Compare-like pattern (has op, left, right)
        if hasattr(node, "op") and hasattr(node, "left") and hasattr(node, "right"):
            self._format_compare_like(node)
            return

        # Detect Function-like pattern (has name, params, body)
        if hasattr(node, "name") and hasattr(node, "params") and hasattr(node, "body"):
            self._format_function_like(node)
            return

        # Detect Assign-like pattern (has target, value)
        if hasattr(node, "target") and hasattr(node, "value"):
            self._format_assign_like(node)
            return

        # Detect For-like pattern (has target, iter, body)
        if hasattr(node, "target") and hasattr(node, "iter") and hasattr(node, "body"):
            self._format_for_like(node)
            return

        # Detect If-like pattern (has test, body, orelse)
        if hasattr(node, "test") and hasattr(node, "body"):
            self._format_if_like(node)
            return

        # Detect Return-like pattern
        if cls_name.endswith("Return"):
            self._format_return_like(node)
            return

        # Detect ExprStmt-like pattern
        if cls_name.endswith("ExprStmt"):
            self._format_exprstmt_like(node)
            return

        # Detect Store-like pattern (has pointer, value)
        if hasattr(node, "pointer") and hasattr(node, "value"):
            self._format_store_like(node)
            return

        # Detect Load-like pattern (has pointer)
        if hasattr(node, "pointer"):
            self._format_load_like(node)
            return

        # Detect Call-like pattern (has func, args)
        if hasattr(node, "func") and hasattr(node, "args"):
            self._format_call_like(node)
            return

        # Detect Name-like pattern (has name, no other IRNode fields)
        if hasattr(node, "name") and cls_name.endswith("Name"):
            self._format_name_like(node)
            return

        # Detect Constant-like pattern (has value, no other IRNode fields)
        if hasattr(node, "value") and cls_name.endswith("Constant"):
            self._format_constant_like(node)
            return

        # Detect Tuple-like pattern
        if hasattr(node, "elts"):
            self._format_tuple_like(node)
            return

        # Detect Subscript-like pattern
        if hasattr(node, "value") and hasattr(node, "slice"):
            self._format_subscript_like(node)
            return

        # Fallback: use _op_name and _to_inline
        self._format_fallback(node)

    # -- concrete formatters -------------------------------------------------

    def _format_function_like(self, node):
        name = self._short(node.name)
        func_label = node._op_name()

        # Params
        param_strs = []
        for i, p in enumerate(node.params):
            p_name = f"%arg{i}"
            self._bind(p, p_name)
            # Record param name -> %argN mapping for body name resolution
            p_attr_name = getattr(p, "name", None)
            if p_attr_name is not None:
                self._param_name_map[self._short(p_attr_name)] = p_name
            if hasattr(p, "is_constexpr") and p.is_constexpr:
                param_strs.append(f"{p_name}: constexpr")
            elif hasattr(p, "annotation") and p.annotation:
                param_strs.append(f"{p_name}: {p.annotation}")
            elif hasattr(p, "ndim"):
                param_strs.append(f"{p_name}: tensor<{p.ndim}d>")
            else:
                param_strs.append(p_name)

        self._lines.append(f"{func_label} @{name}({', '.join(param_strs)}) {{")

        # Invariants
        invariants = getattr(node, "invariants", None)
        if invariants:
            self._lines.append("  // invariants")
            for inv in invariants:
                if hasattr(inv, "target") and hasattr(inv, "value"):
                    target_short = self._short(inv.target)
                    val = self._emit_expr(inv.value)
                    self._lines.append(f"  {target_short} = {val}")
                else:
                    self._format_node(inv)

        # Body
        body = getattr(node, "body", None)
        if body:
            self._lines.append("  // body")
            for stmt in body:
                self._format_node(stmt)

        # Grid / Autotune (L3)
        grid = getattr(node, "grid", None)
        if grid is not None:
            self._format_node(grid)
        autotune = getattr(node, "autotune", None)
        if autotune is not None:
            self._format_node(autotune)
        grid_expr = getattr(node, "grid_expr", None)
        if grid_expr is not None:
            self._lines.append(f"  grid = {grid_expr!r}")

        self._lines.append("}")

    def _format_binop_like(self, node):
        if id(node) in self._names:
            return
        op_sym = getattr(node, "op", None)
        if op_sym in _BINOP_SYMBOLS:
            op_name = _BINOP_SYMBOLS[op_sym]
        else:
            op_name = node._op_name()

        lhs = self._emit_expr(node.lhs)
        rhs = self._emit_expr(node.rhs)

        result = self._fresh()
        self._bind(node, result)
        self._lines.append(f"  {result} = {op_name} {lhs}, {rhs}")

    def _format_compare_like(self, node):
        if id(node) in self._names:
            return
        op_sym = getattr(node, "op", None)
        if op_sym in _COMPARE_SYMBOLS:
            op_name = _COMPARE_SYMBOLS[op_sym]
        else:
            op_name = node._op_name()

        # Support both left/right and lhs/rhs field names
        left_node = getattr(node, "left", None) or getattr(node, "lhs", None)
        right_node = getattr(node, "right", None) or getattr(node, "rhs", None)
        left = self._emit_expr(left_node)
        right = self._emit_expr(right_node)

        result = self._fresh()
        self._bind(node, result)
        self._lines.append(f"  {result} = {op_name} {left}, {right}")

    def _format_assign_like(self, node):
        target = node.target
        value = node.value

        val_str = self._emit_expr(value)

        # Target can be a string or an IRNode.
        # Tensor accesses (stores) keep their resolved SSA name.
        # All other targets get a fresh numeric name for proper SSA form.
        if isinstance(target, IRNode):
            cls_name = type(target).__name__
            if cls_name == "L1TensorAccess" or (hasattr(target, "param_name") and hasattr(target, "tensor")):
                # Tensor store: keep the resolved tensor reference
                target_str = self._emit_expr(target)
            else:
                # Variable assignment: use fresh numeric name
                target_str = self._fresh()
                self._bind(target, target_str)
                # Register in _def_map so that later L1Name nodes with the
                # same name resolve to this SSA name.
                target_name = getattr(target, "name", None)
                if target_name is not None:
                    self._def_map[target_name] = target_str
        else:
            target_str = self._fresh()
            self._def_map[target] = target_str

        self._lines.append(f"  {target_str} = {val_str}")

    def _format_store_like(self, node):
        ptr = self._emit_expr(node.pointer)
        val = self._emit_expr(node.value)
        parts = [ptr, val]

        mask = getattr(node, "mask", None)
        if mask is not None:
            mask_str = self._emit_expr(mask)
            parts.append(f"mask={mask_str}")

        other = getattr(node, "other", None)
        if other is not None:
            other_str = self._emit_expr(other)
            parts.append(f"other={other_str}")

        self._lines.append(f"  store {', '.join(parts)}")

    def _format_load_like(self, node):
        if id(node) in self._names:
            return
        ptr = self._emit_expr(node.pointer)
        parts = [ptr]

        mask = getattr(node, "mask", None)
        if mask is not None:
            mask_str = self._emit_expr(mask)
            parts.append(f"mask={mask_str}")

        other = getattr(node, "other", None)
        if other is not None:
            other_str = self._emit_expr(other)
            parts.append(f"other={other_str}")

        result = self._fresh()
        self._bind(node, result)
        op_name = node._op_name()
        self._lines.append(f"  {result} = {op_name} {', '.join(parts)}")

    def _format_for_like(self, node):
        target = self._short(node.target) if isinstance(node.target, str) else self._inline(node.target)
        iter_str = self._emit_expr(node.iter)
        self._lines.append(f"  for {target} in {iter_str} {{")

        # Temporarily increase indent for body
        old_len = len(self._lines)
        for stmt in node.body:
            self._format_node(stmt)
        # Add extra indent to body lines
        for i in range(old_len, len(self._lines)):
            if not self._lines[i].startswith("  //"):
                self._lines[i] = "    " + self._lines[i]

        self._lines.append("  }")

    def _format_if_like(self, node):
        test_str = self._emit_expr(node.test)
        self._lines.append(f"  if {test_str} {{")

        old_len = len(self._lines)
        for stmt in node.body:
            self._format_node(stmt)
        for i in range(old_len, len(self._lines)):
            if not self._lines[i].startswith("  //"):
                self._lines[i] = "    " + self._lines[i]

        orelse = getattr(node, "orelse", None)
        if orelse:
            self._lines.append("  } else {")
            old_len = len(self._lines)
            for stmt in orelse:
                self._format_node(stmt)
            for i in range(old_len, len(self._lines)):
                if not self._lines[i].startswith("  //"):
                    self._lines[i] = "    " + self._lines[i]

        self._lines.append("  }")

    def _format_return_like(self, node):
        value = getattr(node, "value", None)
        if value is not None:
            val_str = self._emit_expr(value)
            self._lines.append(f"  return {val_str}")
        else:
            self._lines.append("  return")

    def _format_exprstmt_like(self, node):
        expr = node.expr
        self._emit_expr(expr)

    def _format_call_like(self, node):
        if id(node) in self._names:
            return
        func = getattr(node, "func", None)
        if isinstance(func, str):
            func_str = self._short(func)
            # Replace parameter names in the func string with their %argN forms.
            # This handles method calls like "(q * scale).to(...)" where the
            # entire expression was unparse'd into a string during L1 construction.
            func_str = self._resolve_names_in_str(func_str)
        else:
            func_str = self._inline(func)

        args = getattr(node, "args", ())
        arg_strs = [self._emit_expr(a) for a in args]

        kwargs = getattr(node, "kwargs", {})
        for k, v in kwargs.items():
            arg_strs.append(f"{k}={self._emit_expr(v)}")

        result = self._fresh()
        self._bind(node, result)
        op_name = node._op_name()
        self._lines.append(f"  {result} = {op_name} {func_str}({', '.join(arg_strs)})")

    def _format_name_like(self, node):
        if id(node) in self._names:
            return
        name = self._short(node.name)
        # If this name matches a function parameter, use the %argN form
        if name in self._param_name_map:
            self._bind(node, self._param_name_map[name])
        elif name in self._def_map:
            # If this name was defined by a previous assignment, reuse its
            # SSA name so that all references are consistent.
            self._bind(node, self._def_map[name])
        else:
            self._bind(node, f"%{name}")

    def _format_constant_like(self, node):
        if id(node) in self._names:
            return
        # Constants are always inlined, but we still bind them so
        # _emit_expr can find them.
        self._bind(node, repr(node.value))

    def _format_tuple_like(self, node):
        if id(node) in self._names:
            return
        elts = [self._emit_expr(e) for e in node.elts]
        result = self._fresh()
        self._bind(node, result)
        self._lines.append(f"  {result} = ({', '.join(elts)})")

    def _format_subscript_like(self, node):
        if id(node) in self._names:
            return
        val = self._emit_expr(node.value)
        sl = self._emit_expr(node.slice)
        result = self._fresh()
        self._bind(node, result)
        self._lines.append(f"  {result} = {val}[{sl}]")

    def _format_fallback(self, node):
        """Generic fallback for nodes that don't match any pattern."""
        if id(node) in self._names:
            return
        op_name = node._op_name()
        parts = []
        for fname, fval in node._iter_fields():
            if fval is None:
                continue
            if isinstance(fval, IRNode):
                parts.append(self._emit_expr(fval))
            elif isinstance(fval, (list, tuple)):
                if all(isinstance(v, IRNode) for v in fval):
                    parts.append("[" + ", ".join(self._emit_expr(v) for v in fval) + "]")
                else:
                    parts.append(repr(fval))
            else:
                parts.append(repr(fval))

        if parts:
            result = self._fresh()
            self._bind(node, result)
            self._lines.append(f"  {result} = {op_name} {', '.join(parts)}")

    # -- expression emission (hoist complex, inline simple) ------------------

    def _emit_expr(self, node) -> str:
        """Emit an expression, hoisting complex sub-expressions as needed.

        Returns the SSA name or inline text for the expression.
        """
        if not isinstance(node, IRNode):
            return repr(node)

        # Already hoisted
        name = self._name_of(node)
        if name is not None:
            return name

        # Simple nodes: bind and inline directly
        if self._is_simple(node):
            cls_name = type(node).__name__
            if cls_name.endswith("Name") and not cls_name.endswith("ParamName"):
                self._format_name_like(node)
            elif cls_name.endswith("Constant"):
                self._format_constant_like(node)
            elif cls_name == "L3ProgramId":
                self._bind(node, node._to_inline())
            elif cls_name == "L3Arange":
                self._bind(node, node._to_inline())
            return self._name_of(node)

        # Complex node: hoist it (emit as a statement, return its name)
        self._format_node(node)
        return self._name_of(node) or node._to_inline()


# Sentinel types used by _is_simple to decide inlining.
# Real Name/Constant subclasses are NOT instances of these — we check
# the actual class names in _is_simple instead.

class SSAName:
    """Mixin marker — not used directly, see SSAFormatter._is_simple."""
    pass


class SSAConstant:
    """Mixin marker — not used directly, see SSAFormatter._is_simple."""
    pass


# Monkey-patch _is_simple to use class-name heuristics
def _is_simple_impl(self, node) -> bool:
    if not isinstance(node, IRNode):
        return True
    cls_name = type(node).__name__
    if cls_name.endswith("Name") and not cls_name.endswith("ParamName"):
        return True
    if cls_name.endswith("Constant"):
        return True
    if cls_name == "L3ProgramId":
        return True
    if cls_name == "L3Arange":
        return True
    return False


SSAFormatter._is_simple = _is_simple_impl
