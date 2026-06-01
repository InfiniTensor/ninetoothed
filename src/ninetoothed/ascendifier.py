import ast


class Ascendifier(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.max_axes = None
        self._needs_ascend_config_prune = False
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
    def _name_is_last_defined_by_dot(cls, name, prior_stmts):
        for prior in reversed(prior_stmts):
            if cls._assign_target_name(prior) != name:
                continue

            return isinstance(prior.value, ast.Call) and cls._is_dot_call(prior.value)

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
    def _make_tail_key_boundary_test(cls, mask_expr):
        if not (
            isinstance(mask_expr, ast.Compare)
            and len(mask_expr.ops) == 1
            and isinstance(mask_expr.ops[0], ast.Lt)
            and len(mask_expr.comparators) == 1
        ):
            return None

        limit_expr = mask_expr.comparators[0]
        if not cls._is_dynamic_limit_expr(limit_expr):
            return None

        split = cls._split_block_offset(mask_expr.left)
        if split is None:
            return None

        offset_base, block_expr = split
        tail_lhs = ast.BinOp(left=offset_base, op=ast.Add(), right=block_expr)
        return ast.Compare(
            left=tail_lhs,
            ops=[ast.Gt()],
            comparators=[cls._clone(limit_expr)],
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
        dtype = ast.Attribute(value=cls._clone(dtype_source), attr="dtype", ctx=ast.Load())
        return ast.Call(
            func=ast.Attribute(value=cls._clone(value_expr), attr="to", ctx=ast.Load()),
            args=[dtype],
            keywords=[],
        )

    @classmethod
    def _make_tail_key_boundary_bias_assign(cls, target_name, mask_expr):
        target = ast.Name(id=target_name, ctx=ast.Load())
        full_mask = cls._broadcast_to_shape_of(
            cls._make_row_broadcast_view(mask_expr), target
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
    def _make_tail_key_boundary_stable_mask_assign(cls, target_name, mask_expr):
        target = ast.Name(id=target_name, ctx=ast.Load())
        full_mask = cls._broadcast_to_shape_of(
            cls._make_row_broadcast_view(mask_expr), target
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
    def _is_exp2_assign_using_name(cls, stmt, name):
        if not (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Call)
            and cls._is_triton_language_member(stmt.value.func, "exp2")
            and len(stmt.value.args) == 1
        ):
            return False

        return any(
            isinstance(node, ast.Name) and node.id == name
            for node in ast.walk(stmt.value.args[0])
        )

    @classmethod
    def _rewrite_tail_key_boundary_where_assign(cls, stmt, prior_stmts):
        if not (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.Call)
            and cls._is_where_call(stmt.value)
            and len(stmt.value.args) >= 3
        ):
            return None

        target_name = stmt.targets[0].id
        where_call = stmt.value
        if not (
            cls._is_sdpa_score_name(target_name)
            and isinstance(where_call.args[1], ast.Name)
            and where_call.args[1].id == target_name
            and cls._is_negative_inf_literal(where_call.args[2])
            and cls._name_is_last_defined_by_dot(target_name, prior_stmts)
        ):
            return None

        mask_expr = where_call.args[0]
        tail_condition = cls._make_tail_key_boundary_test(mask_expr)
        if tail_condition is None:
            return None

        rewritten = ast.If(
            test=tail_condition,
            body=[cls._make_tail_key_boundary_bias_assign(target_name, mask_expr)],
            orelse=[],
        )
        return rewritten, target_name, tail_condition, mask_expr

    # BiShengIR cannot compile SDPA's original key-boundary where on qk.
    # Keep this rewrite narrow: qk-like dot result, -inf false value,
    # and a provable block-offset < runtime-length mask.
    @classmethod
    def _rewrite_tail_key_boundary_masks(cls, module):
        for stmt in module.body:
            if not isinstance(stmt, ast.FunctionDef):
                continue

            for node in ast.walk(stmt):
                if not isinstance(node, ast.For):
                    continue

                new_loop_body = []
                pending_tail_masks = {}
                for loop_stmt in node.body:
                    rewrite = cls._rewrite_tail_key_boundary_where_assign(
                        loop_stmt, new_loop_body
                    )
                    if rewrite is not None:
                        rewritten, target_name, tail_condition, mask_expr = rewrite
                        new_loop_body.append(rewritten)
                        pending_tail_masks[target_name] = (tail_condition, mask_expr)
                        continue

                    new_loop_body.append(loop_stmt)
                    for masked_name, (tail_condition, mask_expr) in pending_tail_masks.items():
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
                                        stable_name, mask_expr
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
    def _ascend_config_prune_function_name():
        return "_ninetoothed_prune_ascend_configs"

    @classmethod
    def _ascend_config_prune_function_def(cls):
        source = """
def {name}(configs, nargs, **kwargs):
    max_core_dim = 65535
    max_ub_bits = 1572864
    max_cc_bits = 1048576

    try:
        grid = kwargs["grid"]
    except KeyError:
        grid = None

    def _as_positive_int(value):
        try:
            value = int(value)
        except Exception:
            return None

        if value <= 0:
            return None

        return value

    def _is_block_key(key):
        key = str(key)
        upper_key = key.upper()
        lower_key = key.lower()
        return (
            "BLOCK_SIZE" in upper_key
            or upper_key.startswith("BLOCK_")
            or lower_key.startswith("block_size")
        )

    def _not_none(value):
        return value is not None

    def _block_items(meta):
        items = []
        for key, value in meta.items():
            if not _is_block_key(key):
                continue

            value = _as_positive_int(value)
            if _not_none(value):
                items.append((str(key), value))

        return items

    def _block_values(meta):
        return [value for _, value in _block_items(meta)]

    def _feature_dim(meta, fallback):
        candidates = []
        for key, value in meta.items():
            key = str(key)
            upper_key = key.upper()
            lower_key = key.lower()
            if (
                "EMB_DIM" not in upper_key
                and "HEAD_DIM" not in upper_key
                and "D_MODEL" not in upper_key
                and "size_3" not in lower_key
            ):
                continue

            value = _as_positive_int(value)
            if value is not None and value <= 4096:
                candidates.append(value)

        if candidates:
            return max(candidates)

        return fallback

    def _estimate_memory_bits(meta):
        block_items = _block_items(meta)
        blocks = [value for _, value in block_items]
        if not blocks:
            return None

        sub_blocks = [value for key, value in block_items if "SUB" in key.upper()]
        if len(blocks) == 1 or sub_blocks:
            block = max(sub_blocks or blocks)
            # Generic elementwise-like estimate: input, output, mask/temp, plus
            # one extra temporary. Kernels that need larger blocks generally
            # require explicit sub-blocking instead of a larger single tile; when
            # such a sub-block exists, UB pressure is bounded by that sub-block.
            ub_bits = block * 16 * 4
            return ub_bits, 0

        block_m = blocks[0]
        block_n = blocks[1]
        feature = _feature_dim(meta, max(block_n, 64))

        dot_bits = block_m * block_n * 32
        qk_like_bits = 3 * dot_bits
        acc_like_bits = block_m * feature * 32
        input_like_bits = (block_m * feature + 2 * block_n * feature) * 16
        vector_like_bits = (4 * block_m + block_n + feature) * 32

        ub_bits = qk_like_bits + acc_like_bits + input_like_bits + vector_like_bits
        cc_bits = 2 * dot_bits

        return ub_bits, cc_bits

    def _core_dim(meta):
        try:
            grid_dims = grid(meta) if callable(grid) else grid
        except Exception:
            return None

        if not isinstance(grid_dims, tuple):
            grid_dims = (grid_dims,)

        core_dim = 1
        try:
            for dim in grid_dims:
                core_dim *= int(dim)
        except Exception:
            return None

        return core_dim

    safe_records = []
    pruned_for_core_dim = False
    for config in configs:
        meta = dict(kwargs)
        meta.update(config.all_kwargs())

        core_dim = _core_dim(meta)
        if core_dim is not None and core_dim > max_core_dim:
            pruned_for_core_dim = True
            continue

        memory_bits = _estimate_memory_bits(meta)
        if _not_none(memory_bits):
            ub_bits, cc_bits = memory_bits
            if ub_bits > max_ub_bits or cc_bits > max_cc_bits:
                continue

        blocks = _block_values(meta)
        block_footprint = 0
        for block in blocks:
            block_footprint = max(block_footprint, block)
        safe_records.append((config, core_dim, block_footprint))

    if safe_records:
        if pruned_for_core_dim:
            safe_records = sorted(
                safe_records,
                key=lambda item: (
                    max_core_dim if item[1] is None else item[1],
                    -item[2],
                ),
            )

        return [config for config, _, _ in safe_records]

    raise RuntimeError(
        "No Ascend-safe Triton autotune configs: all candidates exceed "
        "coreDim, UB, or CC limits."
    )
""".format(name=cls._ascend_config_prune_function_name())
        return ast.parse(source).body[0]

    @classmethod
    def _has_ascend_config_prune_function(cls, module):
        return any(
            isinstance(stmt, ast.FunctionDef)
            and stmt.name == cls._ascend_config_prune_function_name()
            for stmt in module.body
        )

    @classmethod
    def _inject_ascend_config_prune_function(cls, module):
        if cls._has_ascend_config_prune_function(module):
            return module

        insert_at = 0
        while insert_at < len(module.body) and isinstance(
            module.body[insert_at], (ast.Import, ast.ImportFrom)
        ):
            insert_at += 1

        module.body.insert(insert_at, cls._ascend_config_prune_function_def())
        return module

    @classmethod
    def _chain_ascend_config_prune(cls, existing):
        args = ast.arguments(
            posonlyargs=[],
            args=[
                ast.arg(arg="configs"),
                ast.arg(arg="nargs"),
            ],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=ast.arg(arg="kwargs"),
            defaults=[],
        )
        chained_configs = ast.Call(
            func=cls._clone(existing),
            args=[
                ast.Name(id="configs", ctx=ast.Load()),
                ast.Name(id="nargs", ctx=ast.Load()),
            ],
            keywords=[
                ast.keyword(arg=None, value=ast.Name(id="kwargs", ctx=ast.Load()))
            ],
        )
        return ast.Lambda(
            args=args,
            body=ast.Call(
                func=ast.Name(
                    id=cls._ascend_config_prune_function_name(),
                    ctx=ast.Load(),
                ),
                args=[
                    chained_configs,
                    ast.Name(id="nargs", ctx=ast.Load()),
                ],
                keywords=[
                    ast.keyword(arg=None, value=ast.Name(id="kwargs", ctx=ast.Load()))
                ],
            ),
        )

    @classmethod
    def _add_ascend_config_prune_to_autotune(cls, node):
        for kw in node.keywords:
            if kw.arg != "prune_configs_by":
                continue

            if isinstance(kw.value, ast.Dict):
                for index, key_node in enumerate(kw.value.keys):
                    if not (
                        isinstance(key_node, ast.Constant)
                        and key_node.value == "early_config_prune"
                    ):
                        continue

                    kw.value.values[index] = cls._chain_ascend_config_prune(
                        kw.value.values[index]
                    )
                    return True

                kw.value.keys.append(ast.Constant(value="early_config_prune"))
                kw.value.values.append(
                    ast.Name(id=cls._ascend_config_prune_function_name(), ctx=ast.Load())
                )
                return True

            return False

        node.keywords.append(
            ast.keyword(
                arg="prune_configs_by",
                value=ast.Dict(
                    keys=[ast.Constant(value="early_config_prune")],
                    values=[
                        ast.Name(
                            id=cls._ascend_config_prune_function_name(),
                            ctx=ast.Load(),
                        )
                    ],
                ),
            )
        )
        return True

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
        index, elt = item
        value = str(elt.value)
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
            elt
            for elt in key_nodes
            if isinstance(elt, ast.Constant) and "size" in str(elt.value)
        ]
        return [
            elt
            for _, elt in sorted(enumerate(size_keys), key=cls._autotune_key_priority)
        ][:max_axes]

    @classmethod
    def _rewrite_autotune_keyword(cls, kw, max_axes):
        if kw.arg == "configs" and cls._is_sequence_literal(kw.value):
            cls._rewrite_square_block_autotune_configs(kw.value.elts)
            return

        if kw.arg == "key" and cls._is_sequence_literal(kw.value):
            kw.value.elts = cls._filter_autotune_keys(kw.value.elts, max_axes)

    @classmethod
    def _rewrite_autotune_call(cls, node, max_axes):
        for kw in node.keywords:
            cls._rewrite_autotune_keyword(kw, max_axes)

        return cls._add_ascend_config_prune_to_autotune(node)

    @classmethod
    def _rewrite_load_call(cls, node):
        if not cls._is_triton_language_member(node.func, "load"):
            return

        for kw in node.keywords:
            if (
                kw.arg == "other"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is None
            ):
                kw.value.value = 0.0

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
            if type(self)._rewrite_autotune_call(node, self.max_axes):
                self._needs_ascend_config_prune = True

        type(self)._rewrite_load_call(node)
        return type(self)._rewrite_clamp_call(node)

    def visit_Module(self, node):
        self.generic_visit(node)

        node = type(self)._rewrite_tail_key_boundary_masks(node)
        if self._needs_ascend_config_prune:
            node = type(self)._inject_ascend_config_prune_function(node)

        return node
