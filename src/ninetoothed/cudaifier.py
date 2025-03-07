import ast

import ninetoothed.naming as naming
from ninetoothed.tensor import Tensor


class Cudaifier(ast.NodeTransformer):
    def visit_Name(self, node):
        self.generic_visit(node)

        source = node.id

        if naming.is_constexpr(source):
            return node

        def repl(match):
            return f"{match.group(1)}.data"

        source = Tensor.pointer_pattern().sub(repl, source)

        def repl(match):
            return f"{match.group(1)}.shape[{match.group(3)}]"

        source = Tensor.size_pattern().sub(repl, source)

        def repl(match):
            return f"{match.group(1)}.strides[{match.group(3)}]"

        source = Tensor.stride_pattern().sub(repl, source)

        source = source.removesuffix("_with_auto_tuning")

        if source != node.id:
            return ast.parse(source, mode="eval").body

        return node
