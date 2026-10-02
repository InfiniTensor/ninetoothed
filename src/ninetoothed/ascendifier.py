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
