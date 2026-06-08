import ast


class Ascendifier(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.max_axes = None
        try:
            from triton.backends.ascend.runtime.utils import valid_axis_names

            self.max_axes = len(valid_axis_names)
        except ImportError:
            pass

    @staticmethod
    def _is_triton_language_name(node):
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

    @classmethod
    def _broadcast_to_shape_of(cls, node, anchor):
        return cls._make_member_call(
            cls._triton_language_namespace(),
            "broadcast_to",
            node,
            ast.Attribute(value=cls._clone(anchor), attr="shape"),
        )

    @classmethod
    def _is_where_call(cls, node):
        return isinstance(node, ast.Call) and cls._is_triton_language_member(
            node.func, "where"
        )

    @classmethod
    def _is_dot_call(cls, node):
        return isinstance(node, ast.Call) and cls._is_triton_language_member(
            node.func, "dot"
        )

    @staticmethod
    def _assign_target_name(node):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            return node.targets[0].id

        return None

    @staticmethod
    def _is_negative_inf_literal(node):
        if isinstance(node, ast.Constant) and node.value == float("-inf"):
            return True

        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "float"
            and len(node.args) == 1
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
            and node.args[0].value.strip().lower() in {"-inf", "-infinity"}
        )

    # Keep this SDPA workaround narrow: it handles only the key-boundary
    # mask on qk-like dot results, then masks stable_qk after exp2.
    @staticmethod
    def _is_sdpa_score_name(name):
        return name == "qk" or name.endswith("_qk")

    @classmethod
    def _name_is_last_defined_by_dot(cls, name, previous_statements):
        for previous_statement in reversed(previous_statements):
            if cls._assign_target_name(previous_statement) != name:
                continue

            return isinstance(previous_statement.value, ast.Call) and cls._is_dot_call(
                previous_statement.value
            )

        return False

    @classmethod
    def _is_block_arange_call(cls, node):
        return (
            isinstance(node, ast.Call)
            and cls._is_triton_language_member(node.func, "arange")
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == 0
        )

    @classmethod
    def _split_block_offset(cls, node):
        if cls._is_block_arange_call(node):
            return ast.Constant(value=0), cls._clone(node.args[1])

        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
            return None

        if cls._is_block_arange_call(node.left):
            return cls._clone(node.right), cls._clone(node.left.args[1])

        if cls._is_block_arange_call(node.right):
            return cls._clone(node.left), cls._clone(node.right.args[1])

        return None

    @staticmethod
    def _is_dynamic_limit_expr(node):
        return not isinstance(node, ast.Constant)

    @classmethod
    def _make_tail_key_boundary_test(cls, mask_expression):
        if not (
            isinstance(mask_expression, ast.Compare)
            and len(mask_expression.ops) == 1
            and isinstance(mask_expression.ops[0], ast.Lt)
            and len(mask_expression.comparators) == 1
        ):
            return None

        limit_expression = mask_expression.comparators[0]
        if not cls._is_dynamic_limit_expr(limit_expression):
            return None

        split = cls._split_block_offset(mask_expression.left)
        if split is None:
            return None

        offset_base, block_expression = split
        tail_left_hand_side = ast.BinOp(
            left=offset_base, op=ast.Add(), right=block_expression
        )
        return ast.Compare(
            left=tail_left_hand_side,
            ops=[ast.Gt()],
            comparators=[cls._clone(limit_expression)],
        )

    @classmethod
    def _make_row_broadcast_view(cls, node):
        return ast.Subscript(
            value=cls._clone(node),
            slice=ast.Tuple(elts=[ast.Constant(value=None), ast.Slice()]),
            ctx=ast.Load(),
        )

    @classmethod
    def _make_to_dtype_call(cls, value_expr, dtype_source):
        dtype = ast.Attribute(
            value=cls._clone(dtype_source), attr="dtype", ctx=ast.Load()
        )
        return ast.Call(
            func=ast.Attribute(value=cls._clone(value_expr), attr="to", ctx=ast.Load()),
            args=[dtype],
            keywords=[],
        )

    @classmethod
    def _make_tail_key_boundary_bias_assign(cls, target_name, mask_expression):
        target = ast.Name(id=target_name, ctx=ast.Load())
        full_mask = cls._broadcast_to_shape_of(
            cls._make_row_broadcast_view(mask_expression), target
        )
        mask_float = cls._make_to_dtype_call(full_mask, target)
        inverse_mask = ast.BinOp(
            left=ast.Constant(value=1.0),
            op=ast.Sub(),
            right=mask_float,
        )
        bias = ast.BinOp(
            left=inverse_mask,
            op=ast.Mult(),
            right=ast.Constant(value=-1.0e9),
        )
        return ast.Assign(
            targets=[ast.Name(id=target_name, ctx=ast.Store())],
            value=ast.BinOp(left=target, op=ast.Add(), right=bias),
        )

    @classmethod
    def _make_tail_key_boundary_stable_mask_assign(cls, target_name, mask_expression):
        target = ast.Name(id=target_name, ctx=ast.Load())
        full_mask = cls._broadcast_to_shape_of(
            cls._make_row_broadcast_view(mask_expression), target
        )
        return ast.Assign(
            targets=[ast.Name(id=target_name, ctx=ast.Store())],
            value=ast.BinOp(
                left=target,
                op=ast.Mult(),
                right=cls._make_to_dtype_call(full_mask, target),
            ),
        )

    @classmethod
    def _is_exp2_assign_using_name(cls, statement, name):
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.Call)
            and cls._is_triton_language_member(statement.value.func, "exp2")
            and len(statement.value.args) == 1
        ):
            return False

        return any(
            isinstance(node, ast.Name) and node.id == name
            for node in ast.walk(statement.value.args[0])
        )

    @classmethod
    def _rewrite_tail_key_boundary_where_assign(cls, statement, previous_statements):
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.Call)
            and cls._is_where_call(statement.value)
            and len(statement.value.args) >= 3
        ):
            return None

        target_name = statement.targets[0].id
        where_call = statement.value
        if not (
            cls._is_sdpa_score_name(target_name)
            and isinstance(where_call.args[1], ast.Name)
            and where_call.args[1].id == target_name
            and cls._is_negative_inf_literal(where_call.args[2])
            and cls._name_is_last_defined_by_dot(target_name, previous_statements)
        ):
            return None

        mask_expression = where_call.args[0]
        tail_condition = cls._make_tail_key_boundary_test(mask_expression)
        if tail_condition is None:
            return None

        rewritten = ast.If(
            test=tail_condition,
            body=[
                cls._make_tail_key_boundary_bias_assign(target_name, mask_expression)
            ],
            orelse=[],
        )
        return rewritten, target_name, tail_condition, mask_expression

    # BiShengIR cannot compile SDPA's original key-boundary where on qk.
    # Keep this rewrite narrow: qk-like dot result, -inf false value,
    # and a provable block-offset < runtime-length mask.
    @classmethod
    def _rewrite_tail_key_boundary_masks(cls, module):
        for statement in module.body:
            if not isinstance(statement, ast.FunctionDef):
                continue

            for node in ast.walk(statement):
                if not isinstance(node, ast.For):
                    continue

                new_loop_body = []
                pending_tail_masks = {}
                for loop_stmt in node.body:
                    rewrite = cls._rewrite_tail_key_boundary_where_assign(
                        loop_stmt, new_loop_body
                    )
                    if rewrite is not None:
                        (
                            rewritten,
                            target_name,
                            tail_condition,
                            mask_expression,
                        ) = rewrite
                        new_loop_body.append(rewritten)
                        pending_tail_masks[target_name] = (
                            tail_condition,
                            mask_expression,
                        )
                        continue

                    new_loop_body.append(loop_stmt)
                    for (
                        masked_name,
                        (tail_condition, mask_expression),
                    ) in pending_tail_masks.items():
                        if not cls._is_exp2_assign_using_name(loop_stmt, masked_name):
                            continue

                        stable_name = cls._assign_target_name(loop_stmt)
                        if stable_name is None:
                            continue

                        new_loop_body.append(
                            ast.If(
                                test=cls._clone(tail_condition),
                                body=[
                                    cls._make_tail_key_boundary_stable_mask_assign(
                                        stable_name, mask_expression
                                    )
                                ],
                                orelse=[],
                            )
                        )
                        break

                node.body = new_loop_body

        return module

    @classmethod
    def _config_block_size_values(cls, node):
        is_triton_config = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Config"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "triton"
        )
        if not (
            isinstance(node, ast.Call)
            and is_triton_config
            and node.args
            and isinstance(node.args[0], ast.Dict)
        ):
            return None

        block_entries = []
        for key, value in zip(node.args[0].keys, node.args[0].values):
            if not (
                isinstance(key, ast.Constant)
                and isinstance(key.value, str)
                and "BLOCK_SIZE" in key.value
                and isinstance(value, ast.Constant)
                and isinstance(value.value, int)
            ):
                continue

            block_entries.append((key.value, value))

        if len(block_entries) != 2:
            return None

        return block_entries

    @classmethod
    def _rewrite_square_block_autotune_configs(cls, configs):
        square_configs = []
        for config in configs:
            block_entries = cls._config_block_size_values(config)
            if block_entries is None:
                return

            values = [value.value for _, value in block_entries]
            if values[0] != values[1]:
                return

            square_configs.append((values[0], block_entries))

        if [value for value, _ in square_configs] != [32, 64, 128]:
            return

        replacement_values = [(32, 64), (64, 128), (128, 64)]
        for (_, block_entries), replacement in zip(square_configs, replacement_values):
            for (_, value_node), new_value in zip(block_entries, replacement):
                value_node.value = new_value

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
        if keyword.arg == "configs" and cls._is_sequence_literal(keyword.value):
            cls._rewrite_square_block_autotune_configs(keyword.value.elts)
            return

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

        if type(self)._is_triton_autotune_call(node):
            type(self)._rewrite_autotune_call(node, self.max_axes)

        type(self)._rewrite_load_call(node)
        return type(self)._rewrite_clamp_call(node)

    def visit_Module(self, node):
        self.generic_visit(node)

        node = type(self)._rewrite_tail_key_boundary_masks(node)

        return node
