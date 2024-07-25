import ast
import re


class Torchifier(ast.NodeTransformer):
    def visit_Name(self, node):
        self.generic_visit(node)

        pattern = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)_(size|stride)_(.+)")

        node.id = node.id.replace("_ptr", "")

        if re.fullmatch(pattern, node.id):
            return ast.parse(
                pattern.sub(
                    lambda match: f"{match.group(1)}.{match.group(2)}({match.group(3)})",
                    node.id,
                ),
                mode="eval",
            ).body

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
