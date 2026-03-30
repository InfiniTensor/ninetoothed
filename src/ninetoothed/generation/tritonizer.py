import ast


class Tritonizer(ast.NodeTransformer):
    def visit_Module(self, node):
        self.generic_visit(node)

        node.body.insert(0, ast.Import(names=[ast.alias(name="triton.language")]))
        node.body.insert(0, ast.Import(names=[ast.alias(name="triton")]))

        return node

    def visit_Name(self, node):
        self.generic_visit(node)

        if node.id == "ninetoothed" or "ninetoothed." in node.id:
            node.id = node.id.replace("ninetoothed", "triton")

        return node

    def visit_Call(self, node):
        self.generic_visit(node)

        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "triton"
            and node.func.attr == "jit"
        ):
            return ast.Attribute(
                value=ast.Name(id="triton", ctx=ast.Load()), attr="jit", ctx=ast.Load()
            )

        return node
