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
