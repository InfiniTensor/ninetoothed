import ast
import math


class Ascendifier(ast.NodeTransformer):
    _BROADCAST_ANCHOR_NAMES = {"qk", "accumulator", "acc"}
    _BROADCAST_VECTOR_NAMES = {"next_max", "next_row_max", "m_ij", "alpha", "l_i"}

    def __init__(self):
        super().__init__()
        self.max_axes = None
        try:
            from triton.backends.ascend.runtime.utils import valid_axis_names
            self.max_axes = len(valid_axis_names)
        except ImportError:
            pass

    @staticmethod
    def _is_tl_name(node):
        return isinstance(node, ast.Name) and node.id == "triton.language"

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
            and (cls._is_tl_name(node.value) or cls._is_triton_language(node.value))
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
    def _name_id(node):
        return node.id if isinstance(node, ast.Name) else None

    @classmethod
    def _assign_target_name(cls, node):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            return node.targets[0].id

        return None

    @classmethod
    def _is_where_call(cls, node):
        return (
            isinstance(node, ast.Call)
            and cls._is_triton_language_member(node.func, "where")
        )

    @classmethod
    def _is_dot_call(cls, node):
        return (
            isinstance(node, ast.Call)
            and cls._is_triton_language_member(node.func, "dot")
        )

    @staticmethod
    def _is_supported_broadcast_anchor(node):
        return Ascendifier._name_id(node) in Ascendifier._BROADCAST_ANCHOR_NAMES

    @staticmethod
    def _is_column_broadcast_view(node):
        return (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and isinstance(node.slice, ast.Tuple)
            and len(node.slice.elts) == 2
            and isinstance(node.slice.elts[0], ast.Slice)
            and isinstance(node.slice.elts[1], ast.Constant)
            and node.slice.elts[1].value is None
        )

    @classmethod
    def _is_supported_broadcast_vector(cls, node):
        return (
            cls._is_column_broadcast_view(node)
            and cls._name_id(node.value) in cls._BROADCAST_VECTOR_NAMES
        )

    @classmethod
    def _broadcast_to_shape_of(cls, node, anchor):
        return cls._make_member_call(
            cls._triton_language_namespace(),
            "broadcast_to",
            node,
            ast.Attribute(value=cls._clone(anchor), attr="shape"),
        )

    @classmethod
    def _rewrite_sub_with_column_broadcast(cls, node):
        if (
            isinstance(node.op, ast.Sub)
            and cls._name_id(node.left) == "qk"
            and cls._is_supported_broadcast_vector(node.right)
        ):
            node.right = cls._broadcast_to_shape_of(node.right, node.left)
            return True

        return False

    @classmethod
    def _rewrite_mul_with_alpha_column_broadcast(cls, node):
        if (
            isinstance(node.op, ast.Mult)
            and cls._is_supported_broadcast_anchor(node.left)
            and cls._is_supported_broadcast_vector(node.right)
            and cls._name_id(node.right.value) == "alpha"
        ):
            node.right = cls._broadcast_to_shape_of(node.right, node.left)
            return True

        if (
            isinstance(node.op, ast.Mult)
            and cls._is_supported_broadcast_anchor(node.right)
            and cls._is_supported_broadcast_vector(node.left)
            and cls._name_id(node.left.value) == "alpha"
        ):
            node.left = cls._broadcast_to_shape_of(node.left, node.right)
            return True

        return False

    @classmethod
    def _rewrite_div_with_li_column_broadcast(cls, node):
        if (
            isinstance(node.op, ast.Div)
            and cls._is_supported_broadcast_anchor(node.left)
            and cls._is_supported_broadcast_vector(node.right)
            and cls._name_id(node.right.value) == "l_i"
        ):
            node.right = cls._broadcast_to_shape_of(node.right, node.left)
            return True

        return False

    @classmethod
    def _rewrite_binop_broadcast_operand(cls, node):
        cls._rewrite_sub_with_column_broadcast(node)
        cls._rewrite_mul_with_alpha_column_broadcast(node)
        cls._rewrite_div_with_li_column_broadcast(node)

        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)

        if type(self)._is_triton_language_member(node, "float64"):
            node.attr = "float32"

        return node

    def visit_ImportFrom(self, node):
        self.generic_visit(node)

        if node.module == "triton.language.extra":
            for alias in node.names:
                if alias.name == "libdevice":
                    node.module = "triton.language.extra.cann"

        return node

    def visit_Call(self, node):
        self.generic_visit(node)

        is_autotune = (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "triton"
            and node.func.attr == "autotune"
        )

        if is_autotune:
            for kw in node.keywords:
                if kw.arg != "key" or not isinstance(kw.value, (ast.List, ast.Tuple)):
                    continue

                filtered_keys = [
                    elt
                    for elt in kw.value.elts
                    if isinstance(elt, ast.Constant) and "size" in str(elt.value)
                ][: self.max_axes]
                kw.value.elts = filtered_keys

        is_load = type(self)._is_triton_language_member(node.func, "load")

        if is_load:
            for kw in node.keywords:
                if (
                    kw.arg == "other"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is None
                ):
                    kw.value.value = 0.0

        is_clamp = type(self)._is_triton_language_member(node.func, "clamp")

        if is_clamp and len(node.args) >= 3 and not node.keywords:
            maximum = type(self)._make_member_call(
                node.func.value, "maximum", node.args[0], node.args[1]
            )
            return type(self)._make_member_call(
                node.func.value, "minimum", maximum, node.args[2]
            )

        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)

        return type(self)._rewrite_binop_broadcast_operand(node)

    def visit_AugAssign(self, node):
        self.generic_visit(node)

        if (
            isinstance(node.op, ast.Div)
            and type(self)._is_supported_broadcast_anchor(node.target)
            and type(self)._is_supported_broadcast_vector(node.value)
            and type(self)._name_id(node.value.value) == "l_i"
        ):
            node.value = type(self)._broadcast_to_shape_of(node.value, node.target)

        return node

    @staticmethod
    def _is_constant_true(node):
        return isinstance(node, ast.Constant) and node.value is True

    @staticmethod
    def _is_negative_inf_literal(node):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, float)
            and math.isinf(node.value)
            and node.value < 0
        ):
            return True

        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "float"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value.strip().lower() in {"-inf", "-infinity"}
        ):
            return True

        return False

    @classmethod
    def _sanitize_where_false_expr(cls, false_expr):
        if cls._is_negative_inf_literal(false_expr):
            return ast.Constant(value=-1.0e9)

        return cls._clone(false_expr)

    @classmethod
    def _arith_mask_where_expr(cls, mask_expr, true_expr, false_expr):
        true_dtype = ast.Attribute(value=cls._clone(true_expr), attr="dtype", ctx=ast.Load())
        mask_float = ast.Call(
            func=ast.Attribute(value=cls._clone(mask_expr), attr="to", ctx=ast.Load()),
            args=[true_dtype],
            keywords=[],
        )
        sanitized_false_expr = cls._sanitize_where_false_expr(false_expr)
        one_minus_mask = ast.BinOp(
            left=ast.Constant(value=1),
            op=ast.Sub(),
            right=cls._clone(mask_float),
        )
        return ast.BinOp(
            left=ast.BinOp(
                left=cls._clone(mask_float),
                op=ast.Mult(),
                right=cls._clone(true_expr),
            ),
            op=ast.Add(),
            right=ast.BinOp(
                left=one_minus_mask,
                op=ast.Mult(),
                right=sanitized_false_expr,
            ),
        )

    @classmethod
    def _is_dot_origin_where_assign(cls, stmt, prior_stmts):
        if not (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Call)
            and cls._is_where_call(stmt.value)
            and len(stmt.value.args) >= 3
        ):
            return False

        where_target = stmt.targets[0].id
        where_true_expr = stmt.value.args[1]
        if cls._name_id(where_true_expr) != where_target:
            return False

        if cls._is_constant_true(stmt.value.args[0]):
            return False

        for prior in reversed(prior_stmts):
            if cls._assign_target_name(prior) != where_target:
                continue

            if isinstance(prior.value, ast.Call) and cls._is_dot_call(prior.value):
                return True

            return False

        return False

    @classmethod
    def _rewrite_loop_dot_where_hazards(cls, module):
        for stmt in module.body:
            if not isinstance(stmt, ast.FunctionDef):
                continue

            for node in ast.walk(stmt):
                if not isinstance(node, ast.For):
                    continue

                new_loop_body = []
                for loop_stmt in node.body:
                    if cls._is_dot_origin_where_assign(loop_stmt, new_loop_body):
                        where_call = loop_stmt.value
                        loop_stmt.value = cls._arith_mask_where_expr(
                            where_call.args[0], where_call.args[1], where_call.args[2]
                        )

                    new_loop_body.append(loop_stmt)

                node.body = new_loop_body

        return module

    def visit_Module(self, node):
        self.generic_visit(node)

        # Keep Ascendifier focused on generic loop-dot-where hazard rewrite.
        return type(self)._rewrite_loop_dot_where_hazards(node)
