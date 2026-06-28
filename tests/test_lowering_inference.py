# ruff: noqa: F841
import inspect
import unittest

import ninetoothed.language as ntl
from ninetoothed.ir import SSAOperationIR, SSAProgramIR, TensorTypeIR
from ninetoothed.ssa import application_to_ssa, render_ssa_program


def fill_statement(out):
    ntl.fill(out, 2.5)


def negative_fill_assignment(out):
    out = ntl.fill(-7.25)


def copy_statement(x, out):
    ntl.copy(x, out)


def sum_statement(x, out):
    ntl.reduce_sum(x, out)


def max_assignment(x, out):
    out = ntl.reduce_max(x)


def transpose_assignment(x, out):
    out = x.T


def matmul_statement(a, b, out):
    ntl.matmul(a, b, out)


def matmul_assignment(a, b, out):
    out = a @ b


def flash_attention_call_name(q, k, v, out):
    ntl.flash_attention(q, k, v, out, 0.125)


def method_sum_assignment(x, out):
    out = x.sum()


def dot_reduction_assignment(x, y, out):
    out = ntl.sum(x * y)


def plain_copy_assignment(x, out):
    out = x


def multi_output_elementwise(x0, x1, cos, sin, out0, out1):
    out0 = x0 * cos - x1 * sin
    out1 = x0 * sin + x1 * cos


def bitwise_shift(x, y, out):
    out = x << y


def compare_float_inf(x, out):
    out = (x == x) & (x != float("inf")) & (x != -float("inf"))


def eye_offsets(out):
    out = out.offsets(0) == out.offsets(1)


def axis_zero_call(x, out):
    out = ntl.sum(x, axis=0)


def rowwise_sum(x, out):
    out = ntl.sum(x, axis=1)


def rowwise_mean(x, out):
    out = ntl.sum(x, axis=1) / 32.0


def rowwise_aminmax(x, out0, out1):
    out0 = ntl.min(x, axis=1)
    out1 = ntl.max(x, axis=1)


def rowwise_var_mean(x, out0, out1):
    mean = ntl.sum(x, axis=1) / 32.0
    var = ntl.sum(x * x, axis=1) / 32.0 - mean * mean
    out0 = var
    out1 = mean


def rowwise_addmv(bias, a, x, out):
    out = bias + ntl.sum(a * x, axis=1)


def rowwise_softmax(x, out):
    m = ntl.max(x, axis=1)
    e = ntl.exp(x - m[:, None])
    out = e / ntl.sum(e, axis=1)[:, None]


def rowwise_layernorm(x, weight, bias, out):
    mean = ntl.sum(x, axis=1) / 32.0
    mean_square = ntl.sum(x * x, axis=1) / 32.0
    var = mean_square - mean * mean
    out = (x - mean[:, None]) * ntl.rsqrt(var[:, None] + 1.0e-5) * weight + bias


def _ssa(
    func,
    tensors: tuple[TensorTypeIR, ...] | None = None,
) -> SSAProgramIR:
    if tensors is None:
        tensors = tuple(
            TensorTypeIR(name, 1, dtype="float32", shape=("n",))
            for name in inspect.signature(func).parameters
        )
    program = application_to_ssa(func, tensor_irs=tensors, kind=func.__name__)
    assert program is not None
    return program


def _walk(program: SSAProgramIR) -> tuple[SSAOperationIR, ...]:
    ops: list[SSAOperationIR] = []

    def visit(operation: SSAOperationIR) -> None:
        ops.append(operation)
        for region in operation.regions:
            for inner in region.operations:
                visit(inner)

    for block in program.blocks:
        for operation in block.operations:
            visit(operation)
    return tuple(ops)


def _opcodes(program: SSAProgramIR) -> tuple[str, ...]:
    return tuple(operation.opcode for operation in _walk(program))


class LoweringInferenceTest(unittest.TestCase):
    def assertNoCoarseProgramIR(self, program: SSAProgramIR) -> None:
        rendered = render_ssa_program(program)
        self.assertNotIn("ProgramIR", rendered)
        self.assertNotIn("ReductionOpIR", rendered)
        self.assertNotIn("MatmulOpIR", rendered)
        self.assertNotIn("FlashAttentionOpIR", rendered)
        self.assertEqual(program.metadata["source"], "application_ast")
        self.assertFalse(program.metadata["coarse_operator_nodes"])

    def test_fill_copy_and_assignment_calls_lower_to_ssa_effects(self):
        for func in (
            fill_statement,
            negative_fill_assignment,
            copy_statement,
            plain_copy_assignment,
        ):
            with self.subTest(func=func.__name__):
                program = _ssa(func)

                self.assertIn("mem.store", _opcodes(program))
                self.assertNoCoarseProgramIR(program)

    def test_reductions_lower_to_ssa_reduce_ops(self):
        cases = (
            (sum_statement, "reduce.sum"),
            (method_sum_assignment, "reduce.sum"),
            (max_assignment, "reduce.max"),
            (dot_reduction_assignment, "reduce.sum"),
        )

        for func, opcode in cases:
            with self.subTest(func=func.__name__):
                program = _ssa(func)
                opcodes = _opcodes(program)

                self.assertIn(opcode, opcodes)
                self.assertIn("mem.store", opcodes)
                self.assertNoCoarseProgramIR(program)

    def test_transpose_and_matmul_lower_to_ssa_compute_ops(self):
        cases = (
            (transpose_assignment, "linalg.transpose"),
            (matmul_statement, "linalg.matmul"),
            (matmul_assignment, "linalg.matmul"),
        )

        for func, opcode in cases:
            with self.subTest(func=func.__name__):
                program = _ssa(func)
                opcodes = _opcodes(program)

                self.assertIn(opcode, opcodes)
                self.assertIn("mem.store", opcodes)
                self.assertNoCoarseProgramIR(program)

    def test_unknown_intrinsic_names_stay_as_call_ops_not_coarse_attention_ir(self):
        tensors = tuple(
            TensorTypeIR(name, 2, dtype="float32", shape=("rows", "cols"))
            for name in ("q", "k", "v", "out")
        )
        program = _ssa(flash_attention_call_name, tensors)

        self.assertIn("call.flash_attention", _opcodes(program))
        self.assertNoCoarseProgramIR(program)

    def test_multi_output_and_scalar_math_are_generic_ssa(self):
        for func, fragments in (
            (multi_output_elementwise, ("arith.mul", "arith.sub", "arith.add")),
            (bitwise_shift, ("arith.bitwise_left_shift",)),
            (compare_float_inf, ("cmp.eq", "cmp.ne", "arith.bitwise_and")),
        ):
            with self.subTest(func=func.__name__):
                program = _ssa(func)
                opcodes = _opcodes(program)

                for fragment in fragments:
                    self.assertIn(fragment, opcodes)
                self.assertIn("mem.store", opcodes)
                self.assertNoCoarseProgramIR(program)

    def test_offsets_lower_to_explicit_index_ops(self):
        program = _ssa(
            eye_offsets,
            (TensorTypeIR("out", 2, dtype="float32", shape=("rows", "cols")),),
        )
        opcodes = _opcodes(program)

        self.assertEqual(opcodes.count("index.offset"), 2)
        self.assertIn("cmp.eq", opcodes)
        self.assertIn("mem.store", opcodes)
        self.assertNoCoarseProgramIR(program)

    def test_axis_reductions_are_not_shape_special_cased(self):
        for func, axis in ((axis_zero_call, 0), (rowwise_sum, 1), (rowwise_mean, 1)):
            with self.subTest(func=func.__name__):
                program = _ssa(func)
                reduce_ops = [op for op in _walk(program) if op.opcode == "reduce.sum"]

                self.assertTrue(reduce_ops)
                self.assertEqual(reduce_ops[0].attrs.get("axis"), axis)
                self.assertNoCoarseProgramIR(program)

    def test_axis_reduction_fusions_lower_to_generic_dataflow(self):
        tensors = (
            TensorTypeIR("x", 2, dtype="float32", shape=("rows", "cols")),
            TensorTypeIR("out0", 1, dtype="float32", shape=("rows",)),
            TensorTypeIR("out1", 1, dtype="float32", shape=("rows",)),
        )
        program = _ssa(rowwise_aminmax, tensors)
        opcodes = _opcodes(program)

        self.assertIn("reduce.min", opcodes)
        self.assertIn("reduce.max", opcodes)
        self.assertEqual(opcodes.count("mem.store"), 2)
        self.assertNoCoarseProgramIR(program)

    def test_rowwise_softmax_and_layernorm_are_dataflow_not_kernel_nodes(self):
        softmax_tensors = (
            TensorTypeIR("x", 2, dtype="float32", shape=("rows", "cols")),
            TensorTypeIR("out", 2, dtype="float32", shape=("rows", "cols")),
        )
        layernorm_tensors = (
            TensorTypeIR("x", 2, dtype="float32", shape=("rows", "cols")),
            TensorTypeIR("weight", 1, dtype="float32", shape=("cols",)),
            TensorTypeIR("bias", 1, dtype="float32", shape=("cols",)),
            TensorTypeIR("out", 2, dtype="float32", shape=("rows", "cols")),
        )

        for func, tensors, fragments in (
            (
                rowwise_softmax,
                softmax_tensors,
                ("reduce.max", "math.exp", "reduce.sum", "arith.div"),
            ),
            (
                rowwise_layernorm,
                layernorm_tensors,
                ("reduce.sum", "math.rsqrt", "arith.mul", "arith.add"),
            ),
        ):
            with self.subTest(func=func.__name__):
                program = _ssa(func, tensors)
                opcodes = _opcodes(program)

                for fragment in fragments:
                    self.assertIn(fragment, opcodes)
                self.assertIn("mem.store", opcodes)
                self.assertNoCoarseProgramIR(program)

    def test_ssa_textual_rendering_is_the_audit_format(self):
        rendered = render_ssa_program(_ssa(rowwise_addmv))

        self.assertTrue(rendered.startswith("ssa @rowwise_addmv {"))
        self.assertIn("reduce.sum", rendered)
        self.assertIn("mem.store", rendered)


if __name__ == "__main__":
    unittest.main()
