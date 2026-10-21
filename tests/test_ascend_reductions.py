import ast
import unittest
from types import SimpleNamespace

import numpy as np

from ninetoothed.ascend_reductions import split_large_load_reductions


class _Pointer:
    def __init__(self, data, offsets=0):
        self.data = data
        self.offsets = offsets

    def __add__(self, offsets):
        return type(self)(self.data, self.offsets + offsets)


def _load(pointer, mask=True, other=0):
    offsets, mask = np.broadcast_arrays(pointer.offsets, mask)
    result = np.full(offsets.shape, other, dtype=pointer.data.dtype)
    result[mask] = pointer.data[offsets[mask]]

    return result


def _store(pointer, value):
    pointer.data[...] = value


class TestSplitLargeReductions(unittest.TestCase):
    def _execute(self, operation, values, *, padded_width, dynamic, rank=1):
        axis = 0 if rank == 1 else 1
        suffix = "" if rank == 1 else "[None, :]"
        width = "width" if dynamic else str(padded_width)
        other = {"sum": "0", "min": "float('inf')", "max": "float('-inf')"}[operation]
        source = f"""
def kernel(x, output, length, width: triton.language.constexpr):
    triton.language.store(output, triton.language.{operation}(
        triton.language.load(x + triton.language.arange(0, {width}){suffix},
                             mask=triton.language.arange(0, {width}){suffix} < length,
                             other={other}), axis={axis}))
"""
        tree = split_large_load_reductions(
            ast.parse(source), max_elements=32, chunk_size=8
        )
        language = SimpleNamespace(
            constexpr=int,
            arange=np.arange,
            load=_load,
            store=_store,
            sum=lambda x, axis: np.sum(x, axis=axis, dtype=np.float32),
            min=np.min,
            max=np.max,
            minimum=np.minimum,
            maximum=np.maximum,
        )
        namespace = {"triton": SimpleNamespace(language=language)}
        exec(compile(tree, "<split-reduction>", "exec"), namespace)
        result = np.empty((1,), dtype=np.float32)
        namespace["kernel"](
            _Pointer(values), _Pointer(result), len(values), padded_width
        )
        expected = getattr(np, operation)(values)
        np.testing.assert_allclose(result[0], expected, rtol=1e-5, atol=1e-5)

        return tree

    def test_constant_and_specialized_widths_preserve_values_and_tail_masks(self):
        values = np.random.default_rng(3).normal(size=61).astype(np.float32)

        for operation in ("sum", "min", "max"):
            for dynamic in (False, True):
                for rank in (1, 2):
                    with self.subTest(operation=operation, dynamic=dynamic, rank=rank):
                        tree = self._execute(
                            operation,
                            values,
                            padded_width=64,
                            dynamic=dynamic,
                            rank=rank,
                        )
                        self.assertTrue(
                            any(isinstance(node, ast.For) for node in ast.walk(tree))
                        )

    def test_small_runtime_specialization_uses_original_reduction(self):
        values = np.arange(13, dtype=np.float32)
        self._execute("sum", values, padded_width=16, dynamic=True)

    def test_small_constant_width_does_not_add_a_loop(self):
        values = np.arange(13, dtype=np.float32)
        tree = self._execute("sum", values, padded_width=16, dynamic=False)
        self.assertFalse(any(isinstance(node, ast.For) for node in ast.walk(tree)))

    def test_materialized_operands_and_other_axes_are_not_rewritten(self):
        source = """
def kernel(x, output):
    value = triton.language.load(x + triton.language.arange(0, 65536))
    result = triton.language.sum(value, axis=0)
    other_axis = triton.language.sum(triton.language.load(
        x + triton.language.arange(0, 65536)[None, :]), axis=0)
"""
        tree = ast.parse(source)
        before = ast.dump(tree)
        after = split_large_load_reductions(tree)
        self.assertEqual(ast.dump(after), before)

    def test_conditional_loads_are_not_hoisted(self):
        source = """
def kernel(x, enabled):
    value = triton.language.sum(triton.language.load(
        x + triton.language.arange(0, 65536)), axis=0) if enabled else 0
    condition = enabled and triton.language.sum(triton.language.load(
        x + triton.language.arange(0, 65536)), axis=0)
"""
        tree = ast.parse(source)
        before = ast.dump(tree)
        self.assertEqual(ast.dump(split_large_load_reductions(tree)), before)

    def test_transformed_or_strided_index_layouts_are_not_rewritten(self):
        for index in (
            "triton.language.trans(triton.language.arange(0, 65536)[None, :])",
            "triton.language.arange(0, 65536)[1::3]",
            "triton.language.arange(0, 65536).reshape((65536, 1))",
        ):
            with self.subTest(index=index):
                tree = ast.parse(f"""
def kernel(x):
    value = triton.language.sum(triton.language.load(x + {index}), axis=0)
""")
                before = ast.dump(tree)
                self.assertEqual(ast.dump(split_large_load_reductions(tree)), before)

    def test_explicit_none_slice_bounds_match_codegen_representation(self):
        tree = ast.parse("""
def kernel(x):
    value = triton.language.sum(triton.language.load(
        x + triton.language.arange(0, 65536)[None, :]), axis=1)
""")

        for node in ast.walk(tree):
            if isinstance(node, ast.Slice):
                node.lower = ast.Constant(value=None)
                node.upper = ast.Constant(value=None)
                node.step = ast.Constant(value=None)

        tree = split_large_load_reductions(tree)
        self.assertTrue(any(isinstance(node, ast.For) for node in ast.walk(tree)))


if __name__ == "__main__":
    unittest.main()
