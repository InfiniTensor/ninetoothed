import ast


class Ascendifier(ast.NodeTransformer):
    def __init__(self, *, omit_static_bounds=False, scalar_values=None):
        super().__init__()
        self.omit_static_bounds = omit_static_bounds
        self.scalar_values = dict(scalar_values or {})
        self.max_axes = None

        try:
            from triton.backends.ascend.runtime.utils import valid_axis_names

            self.max_axes = len(valid_axis_names)
        except ImportError:
            pass

    def transform(self, source: str) -> str:
        tree = ast.parse(source)

        if self.omit_static_bounds:
            if self.scalar_values:
                tree = _ScalarSubstituter(self.scalar_values).visit(tree)

            tree = _StaticBoundsEliminator(tree).visit(tree)

        tree = self.visit(tree)

        return ast.unparse(ast.fix_missing_locations(tree)) + "\n"

    @staticmethod
    def _is_triton_language_name(node):
        return isinstance(node, ast.Name) and node.id in ("triton.language", "tl")

    @staticmethod
    def _is_triton_language(node):
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "language"
            and isinstance(node.value, ast.Name)
            and node.value.id == "triton"
        )

    @classmethod
    def _is_triton_language_member(cls, node, member):
        return (
            isinstance(node, ast.Attribute)
            and node.attr == member
            and (
                cls._is_triton_language_name(node.value)
                or cls._is_triton_language(node.value)
            )
        )

    @staticmethod
    def _clone(node):
        return ast.fix_missing_locations(ast.parse(ast.unparse(node), mode="eval").body)

    @classmethod
    def _make_member_call(cls, namespace, member, *args):
        return ast.Call(
            func=ast.Attribute(value=cls._clone(namespace), attr=member),
            args=[cls._clone(arg) for arg in args],
            keywords=[],
        )

    @staticmethod
    def _triton_language_namespace():
        return ast.Attribute(value=ast.Name(id="triton"), attr="language")

    @staticmethod
    def _is_triton_autotune_call(node):
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "triton"
            and node.func.attr == "autotune"
        )

    @staticmethod
    def _is_sequence_literal(node):
        return isinstance(node, (ast.List, ast.Tuple))

    @staticmethod
    def _autotune_key_priority(item):
        index, key_node = item
        value = str(key_node.value)

        if "next_power_of_2" in value:
            priority = 2
        elif "constexpr" in value:
            priority = 1
        else:
            priority = 0

        return priority, index

    @classmethod
    def _filter_autotune_keys(cls, key_nodes, max_axes):
        size_keys = [
            key_node
            for key_node in key_nodes
            if isinstance(key_node, ast.Constant) and "size" in str(key_node.value)
        ]

        return [
            key_node
            for _, key_node in sorted(
                enumerate(size_keys), key=cls._autotune_key_priority
            )
        ][:max_axes]

    @classmethod
    def _rewrite_autotune_keyword(cls, keyword, max_axes):
        if keyword.arg == "key" and cls._is_sequence_literal(keyword.value):
            keyword.value.elts = cls._filter_autotune_keys(keyword.value.elts, max_axes)

    @classmethod
    def _rewrite_autotune_call(cls, node, max_axes):
        for keyword in node.keywords:
            cls._rewrite_autotune_keyword(keyword, max_axes)

        return

    @classmethod
    def _rewrite_load_call(cls, node):
        if not cls._is_triton_language_member(node.func, "load"):
            return

        for keyword in node.keywords:
            if (
                keyword.arg == "other"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is None
            ):
                keyword.value.value = 0.0

    @classmethod
    def _rewrite_clamp_call(cls, node):
        if not cls._is_triton_language_member(node.func, "clamp"):
            return node

        if len(node.args) < 3 or node.keywords:
            return node

        maximum = cls._make_member_call(
            node.func.value, "maximum", node.args[0], node.args[1]
        )

        return cls._make_member_call(node.func.value, "minimum", maximum, node.args[2])

    @staticmethod
    def _rewrite_bool_op(node):
        if isinstance(node.op, ast.And):
            operator = ast.BitAnd()
        elif isinstance(node.op, ast.Or):
            operator = ast.BitOr()
        else:
            return node

        result = node.values[0]

        for value in node.values[1:]:
            result = ast.copy_location(
                ast.BinOp(left=result, op=operator, right=value),
                node,
            )

        return ast.copy_location(result, node)

    def visit_Attribute(self, node):
        self.generic_visit(node)

        if type(self)._is_triton_language_member(node, "float64"):
            raise ValueError("Ascend does not support float64; use float32 explicitly.")

        return node

    def visit_ImportFrom(self, node):
        self.generic_visit(node)

        if node.module == "triton.language.extra":
            for alias in node.names:
                if alias.name == "libdevice":
                    node.module = "triton.language.extra.ascend"

        return node

    def visit_Call(self, node):
        self.generic_visit(node)

        if type(self)._is_triton_autotune_call(node):
            type(self)._rewrite_autotune_call(node, self.max_axes)

        type(self)._rewrite_load_call(node)

        return type(self)._rewrite_clamp_call(node)

    def visit_BoolOp(self, node):
        self.generic_visit(node)

        return type(self)._rewrite_bool_op(node)


class _ScalarSubstituter(ast.NodeTransformer):
    def __init__(self, values):
        super().__init__()
        self.values = values

    def visit_Name(self, node):
        self.generic_visit(node)

        if isinstance(node.ctx, ast.Load) and node.id in self.values:
            return ast.copy_location(ast.Constant(self.values[node.id]), node)

        return node


class _StaticBoundsEliminator(ast.NodeTransformer):
    def __init__(self, tree):
        super().__init__()
        self.grid_bounds = _static_grid_bounds(tree)
        self.loop_bounds = _static_loop_bounds(tree)

    def visit_Call(self, node):
        self.generic_visit(node)

        if not (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"load", "store"}
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in {"tl", "triton.language"}
        ):
            return node

        mask = next(
            (keyword for keyword in node.keywords if keyword.arg == "mask"),
            None,
        )

        if mask is None or not _proven_true(
            mask.value,
            self.grid_bounds,
            self.loop_bounds,
        ):
            return node

        removed = {"mask", "other"} if node.func.attr == "load" else {"mask"}
        node.keywords = [
            keyword for keyword in node.keywords if keyword.arg not in removed
        ]

        return node


def _static_grid_bounds(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue

        if not any(
            isinstance(target, ast.Name) and target.id == "grid"
            for target in node.targets
        ):
            continue

        if not isinstance(node.value, ast.Tuple):
            return None

        bounds = tuple(
            _static_interval(element, None, {}) for element in node.value.elts
        )

        if any(
            bound is None or bound[0] != bound[1] or bound[0] < 0 for bound in bounds
        ):
            return None

        return tuple(int(bound[0]) for bound in bounds)

    return None


def _static_loop_bounds(tree):
    bounds = {}

    for _ in range(3):
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
            ):
                continue

            value = _static_interval(node.value, None, bounds)

            if value is not None:
                bounds[node.targets[0].id] = value

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
        ):
            continue

        arguments = tuple(node.iter.args)

        if len(arguments) == 1:
            lower, upper, step = 0, arguments[0], None
        elif len(arguments) in {2, 3}:
            lower, upper = arguments[:2]
            step = arguments[2] if len(arguments) == 3 else None
        else:
            continue

        if step is not None:
            step_interval = _static_interval(step, None, bounds)

            if step_interval is None or step_interval != (1, 1):
                continue

        lower_interval = _static_interval(lower, None, bounds)
        upper_interval = _static_interval(upper, None, bounds)

        if (
            lower_interval is None
            or upper_interval is None
            or lower_interval[0] != lower_interval[1]
            or upper_interval[0] != upper_interval[1]
            or upper_interval[0] <= lower_interval[0]
        ):
            continue

        if isinstance(node.target, ast.Name):
            bounds[node.target.id] = (
                lower_interval[0],
                upper_interval[0] - 1,
            )

    return bounds


def _proven_true(node, grid_bounds, loop_bounds):
    if isinstance(node, ast.Constant):
        return bool(node.value)

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
        return all(
            _proven_true(value, grid_bounds, loop_bounds)
            for value in (node.left, node.right)
        )

    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        return all(
            _proven_true(value, grid_bounds, loop_bounds) for value in node.values
        )

    if not isinstance(node, ast.Compare) or len(node.ops) != 1:
        return False

    left = _static_interval(node.left, grid_bounds, loop_bounds)
    right = _static_interval(node.comparators[0], grid_bounds, loop_bounds)

    if left is None or right is None:
        return False

    operator = node.ops[0]

    if isinstance(operator, ast.Lt):
        return left[1] < right[0]

    if isinstance(operator, ast.LtE):
        return left[1] <= right[0]

    if isinstance(operator, ast.Gt):
        return left[0] > right[1]

    if isinstance(operator, ast.GtE):
        return left[0] >= right[1]
    return False


def _static_interval(node, grid_bounds, loop_bounds):
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return (node.value, node.value)

    if isinstance(node, ast.Name):
        if node.id in loop_bounds:
            return loop_bounds[node.id]

        if node.id == "True":
            return (1, 1)

        return None

    if isinstance(node, ast.Subscript):
        return _static_interval(node.value, grid_bounds, loop_bounds)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _static_interval(node.operand, grid_bounds, loop_bounds)

        if value is None:
            return None

        return (-value[1], -value[0])

    if isinstance(node, ast.BinOp):
        left = _static_interval(node.left, grid_bounds, loop_bounds)
        right = _static_interval(node.right, grid_bounds, loop_bounds)

        if left is None or right is None:
            return None

        if isinstance(node.op, ast.Add):
            return (left[0] + right[0], left[1] + right[1])

        if isinstance(node.op, ast.Sub):
            return (left[0] - right[1], left[1] - right[0])

        if isinstance(node.op, ast.Mult):
            values = (
                left[0] * right[0],
                left[0] * right[1],
                left[1] * right[0],
                left[1] * right[1],
            )

            return (min(values), max(values))

        if isinstance(node.op, ast.FloorDiv) and left[0] >= 0 and right[0] > 0:
            return (left[0] // right[1], left[1] // right[0])

        if isinstance(node.op, ast.Mod) and left[0] >= 0 and right[0] > 0:
            return (0, right[1] - 1)

        return None

    if isinstance(node, ast.Call):
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "arange"
            and len(node.args) == 2
        ):
            lower = _static_interval(node.args[0], grid_bounds, loop_bounds)
            upper = _static_interval(node.args[1], grid_bounds, loop_bounds)

            if (
                lower is None
                or upper is None
                or lower[0] != lower[1]
                or upper[0] != upper[1]
            ):
                return None

            return (lower[0], upper[0] - 1)

        if (
            grid_bounds is not None
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "program_id"
            and len(node.args) == 1
        ):
            axis_interval = _static_interval(node.args[0], grid_bounds, loop_bounds)

            if (
                axis_interval is None
                or axis_interval[0] != axis_interval[1]
                or not 0 <= axis_interval[0] < len(grid_bounds)
            ):
                return None

            axis = int(axis_interval[0])

            return (0, grid_bounds[axis] - 1)

    return None
