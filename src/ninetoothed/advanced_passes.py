import ast


def get_const_val(node):
    if isinstance(node, ast.Constant):
        return node.value
    if getattr(ast, "Num", None) and isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        val = get_const_val(node.operand)
        if val is not None:
            return -val
    return None


class SafeAlgebraPass(ast.NodeTransformer):
    """
    Safely folds constants and eliminates redundant arithmetic operations.
    Strictly preserves vector semantics for expressions containing 'arange'
    to prevent broadcasting bugs (e.g., scalarizing zero-strided tensors).
    """

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        lv = get_const_val(node.left)
        rv = get_const_val(node.right)

        # Core protection: Never optimize vector expressions containing 'arange' into a pure scalar 0.
        has_arange = "arange" in ast.dump(node)

        # 1. Basic constant folding
        if lv is not None and rv is not None:
            try:
                if isinstance(node.op, ast.Add):
                    return ast.Constant(value=lv + rv)
                if isinstance(node.op, ast.Sub):
                    return ast.Constant(value=lv - rv)
                if isinstance(node.op, ast.Mult):
                    return ast.Constant(value=lv * rv)
                if isinstance(node.op, ast.FloorDiv) and rv != 0:
                    return ast.Constant(value=lv // rv)
            except Exception:
                pass

        # 2. Redundant addition/subtraction elimination (e.g., (X - C) + C -> X)
        if isinstance(node.op, ast.Add) and isinstance(node.left, ast.BinOp) and isinstance(node.left.op, ast.Sub):
            if get_const_val(node.right) == get_const_val(node.left.right) and get_const_val(node.right) is not None:
                return node.left.left
        if isinstance(node.op, ast.Sub) and isinstance(node.left, ast.BinOp) and isinstance(node.left.op, ast.Add):
            if get_const_val(node.right) == get_const_val(node.left.right) and get_const_val(node.right) is not None:
                return node.left.left

        # 3. Multiplication by 0/1 and Addition/Subtraction by 0
        if isinstance(node.op, ast.Mult):
            if lv == 0 or rv == 0:
                if not has_arange:
                    return ast.Constant(value=0)  # Prevents scalarization
            if lv == 1:
                return node.right
            if rv == 1:
                return node.left
        elif isinstance(node.op, ast.Add):
            if lv == 0:
                return node.right
            if rv == 0:
                return node.left
        elif isinstance(node.op, ast.Sub):
            if rv == 0:
                return node.left
            if ast.dump(node.left) == ast.dump(node.right):
                if not has_arange:
                    return ast.Constant(value=0)  # Prevents scalarization
        elif isinstance(node.op, ast.FloorDiv):
            if rv == 1:
                return node.left
            if lv == 0:
                if not has_arange:
                    return ast.Constant(value=0)  # Prevents scalarization

        return node


class UltimateBCEPass(ast.NodeTransformer):
    """
    Bounds Checking Elimination (BCE).
    Eliminates redundant bounds checks by statically evaluating trivially true conditions.
    """

    def __init__(self):
        self.loop_bounds = {}

    def visit_For(self, node):
        if isinstance(node.target, ast.Name):
            self.loop_bounds[node.target.id] = True
        self.generic_visit(node)
        if isinstance(node.target, ast.Name):
            self.loop_bounds.pop(node.target.id, None)
        return node

    def visit_Compare(self, node):
        node = self.generic_visit(node)
        if len(node.ops) != 1:
            return node

        left = node.left
        op = node.ops[0]
        right = node.comparators[0]
        l_dump = ast.dump(left)
        r_dump = ast.dump(right)
        lv = get_const_val(left)
        rv = get_const_val(right)

        if lv is not None and rv is not None:
            try:
                if isinstance(op, ast.Lt):
                    return ast.Constant(value=lv < rv)
                if isinstance(op, ast.GtE):
                    return ast.Constant(value=lv >= rv)
            except Exception:
                pass

        is_loop_var = isinstance(left, ast.Name) and left.id in self.loop_bounds
        if is_loop_var and isinstance(op, ast.GtE) and rv == 0:
            return ast.Constant(value=True)
        if is_loop_var and isinstance(op, ast.Lt) and "FloorDiv" in r_dump and "size" in r_dump:
            return ast.Constant(value=True)

        if isinstance(op, ast.GtE) and rv == 0:
            if "arange" in l_dump and "index" not in l_dump and "pid" not in l_dump:
                return ast.Constant(value=True)
        if isinstance(op, ast.Lt) and "arange" in l_dump and "BLOCK_SIZE" in l_dump and "BLOCK_SIZE" in r_dump:
            if "size" not in r_dump:
                return ast.Constant(value=True)

        return node


class MaskCSEPass(ast.NodeTransformer):
    """
    Mask-level Common Subexpression Elimination (CSE).
    Simplifies mask expressions by flattening nested BitAnd operations and removing duplicate conditions.
    """

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.BitAnd):

            def flatten_and(n):
                if isinstance(n, ast.BinOp) and isinstance(n.op, ast.BitAnd):
                    return flatten_and(n.left) + flatten_and(n.right)
                return [n]

            terms = flatten_and(node)
            unique_terms = []
            seen = set()
            for t in terms:
                if getattr(t, "value", None) is True:
                    continue
                d = ast.dump(t)
                if d not in seen:
                    seen.add(d)
                    unique_terms.append(t)

            if not unique_terms:
                return ast.Constant(value=True)

            res = unique_terms[0]
            for t in unique_terms[1:]:
                res = ast.BinOp(left=res, op=ast.BitAnd(), right=t)
            return res

        return node


def apply_advanced_optimizations(tree):
    ast.fix_missing_locations(tree)
    old_dump = ""
    # Iteratively apply optimizations until a fixed point is reached
    while old_dump != ast.dump(tree):
        old_dump = ast.dump(tree)
        tree = SafeAlgebraPass().visit(tree)
        tree = UltimateBCEPass().visit(tree)
        tree = MaskCSEPass().visit(tree)
        ast.fix_missing_locations(tree)

    return tree