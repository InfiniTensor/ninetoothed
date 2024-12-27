import ast

import ninetoothed.naming as naming
from ninetoothed.tensor import Tensor


class Torchifier(ast.NodeTransformer):
    def visit_Name(self, node):
        self.generic_visit(node)

        source = node.id

        if naming.is_constexpr(source):
            return node

        def repl(match):
            return f"{match.group(1)}"

        source = Tensor.pointer_pattern().sub(repl, source)

        def repl(match):
            return f"{match.group(1)}.{match.group(2)}({match.group(3)})"

        source = Tensor.size_pattern().sub(repl, source)
        source = Tensor.stride_pattern().sub(repl, source)

        if source != node.id:
            return ast.parse(source, mode="eval").body

        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)

        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "ninetoothed"
            and node.attr == "language"
        ):
            return node.value

        return node
