import ast


class LICMTransformer(ast.NodeTransformer):
    """A transformer that does loop-invariant code motion (LICM)."""

    def visit_Module(self, node):
        while self._process_block(node.body):
            pass

        return node

    def visit_FunctionDef(self, node):
        while self._process_block(node.body):
            pass

        return node

    def _process_block(self, block):
        changed = False

        for stmt in block:
            if not hasattr(stmt, "body"):
                continue

            changed = changed or self._process_block(stmt.body)

        for i, stmt in enumerate(block):
            if not isinstance(stmt, ast.For):
                continue

            changing_vars = self._get_changing_vars(stmt)

            invariants = []
            new_body = []

            for stmt_ in stmt.body:
                if not self._is_invariant(stmt_, changing_vars):
                    new_body.append(stmt_)

                    continue

                invariants.append(stmt_)
                changed = True

            if invariants:
                stmt.body = new_body

                for invariant in reversed(invariants):
                    block.insert(i, invariant)

        return changed

    def _get_changing_vars(self, for_):
        changing_vars = set()

        if isinstance(for_.target, ast.Name):
            changing_vars.add(for_.target.id)

        for node in ast.walk(for_):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        changing_vars.add(target.id)
            elif isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name):
                    changing_vars.add(node.target.id)

        return changing_vars

    def _is_invariant(self, stmt, changing_vars):
        if not isinstance(stmt, ast.Assign):
            return False

        used_vars = set()

        for node in ast.walk(stmt.value):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_vars.add(node.id)

        return not (used_vars & changing_vars)
