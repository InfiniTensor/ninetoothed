import ast
import copy

import ninetoothed.naming as naming


class CSETransformer(ast.NodeTransformer):
    """A transformer that does common subexpression elimination (CSE)."""

    def visit_Module(self, node):
        node.body = self._process_block(node.body)

        return node

    def visit_FunctionDef(self, node):
        node.body = self._process_block(node.body)

        return node

    def visit_Lambda(self, node):
        return node

    def visit_If(self, node):
        node.body = self._process_block(node.body)

        node.orelse = self._process_block(node.orelse)

        return node

    def visit_For(self, node):
        node.body = self._process_block(node.body)

        return node

    def visit_BinOp(self, node):
        return self._process_expr(node)

    def visit_Compare(self, node):
        return self._process_expr(node)

    def _process_block(self, block):
        self._exprs = {}
        self._assignments = []

        new_block = []

        for stmt in block:
            stmt = self.visit(stmt)

            for assignment in self._assignments:
                new_block.append(assignment)

            self._assignments = []

            new_block.append(stmt)

        self._exprs = {}

        return new_block

    def _process_expr(self, expr):
        unparsed = ast.unparse(expr)

        if unparsed in self._exprs:
            return self._exprs[unparsed][0]

        expr = self.generic_visit(copy.deepcopy(expr))

        id = self._make_temporary()
        name = ast.Name(id=id, ctx=ast.Load())
        assignment = ast.Assign(targets=[ast.Name(id=id, ctx=ast.Store())], value=expr)

        self._exprs[unparsed] = (name, expr)
        self._assignments.append(assignment)

        return name

    def _make_temporary(self):
        if not hasattr(self, "_count"):
            self._count = 0

        prefix = f"{naming.auto_generate(f'cse_{self._count}')}_"
        self._count += 1

        return prefix
