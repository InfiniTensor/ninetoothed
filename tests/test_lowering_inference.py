import inspect
import math

import triton.language as tl

import ninetoothed.language as ntl
from ninetoothed.frontend.python import from_application
from ninetoothed.ir import TensorSpec, ssa


def fill_statement(out):
    ntl.fill(out, 2.5)


def negative_fill_assignment(out):
    out = ntl.fill(-7.25)  # noqa: F841


def copy_statement(x, out):
    ntl.copy(x, out)


def sum_statement(x, out):
    ntl.reduce_sum(x, out)


def max_assignment(x, out):
    out = ntl.reduce_max(x)  # noqa: F841


def transpose_assignment(x, out):
    out = x.T  # noqa: F841


def matmul_statement(a, b, out):
    ntl.matmul(a, b, out)


def matmul_assignment(a, b, out):
    out = a @ b  # noqa: F841


def flash_attention_call_name(q, k, v, out):
    ntl.flash_attention(q, k, v, out, 0.125)


def method_sum_assignment(x, out):
    out = x.sum()  # noqa: F841


def dot_reduction_assignment(x, y, out):
    out = ntl.sum(x * y)  # noqa: F841


def plain_copy_assignment(x, out):
    out = x  # noqa: F841


def multi_output_elementwise(x0, x1, cos, sin, out0, out1):
    out0 = x0 * cos - x1 * sin  # noqa: F841
    out1 = x0 * sin + x1 * cos  # noqa: F841


def bitwise_shift(x, y, out):
    out = x << y  # noqa: F841


def compare_float_inf(x, out):
    out = (x == x) & (x != float("inf")) & (x != -float("inf"))  # noqa: F841


def method_math_and_dim_alias(x, out):
    denom = x.sqrt().sum(dim=0)
    out = x.exp() / denom  # noqa: F841


def namespace_math_alias(x, out):
    out = math.exp(x) + tl.sqrt(x)  # noqa: F841


def eye_offsets(out):
    out = out.offsets(0) == out.offsets(1)


def axis_zero_call(x, out):
    out = ntl.sum(x, axis=0)  # noqa: F841


def rowwise_sum(x, out):
    out = ntl.sum(x, axis=1)  # noqa: F841


def rowwise_mean(x, out):
    out = ntl.sum(x, axis=1) / 32.0  # noqa: F841


def rowwise_aminmax(x, out0, out1):
    out0 = ntl.min(x, axis=1)  # noqa: F841
    out1 = ntl.max(x, axis=1)  # noqa: F841


def rowwise_var_mean(x, out0, out1):
    mean = ntl.sum(x, axis=1) / 32.0
    var = ntl.sum(x * x, axis=1) / 32.0 - mean * mean
    out0 = var  # noqa: F841
    out1 = mean  # noqa: F841


def rowwise_addmv(bias, a, x, out):
    out = bias + ntl.sum(a * x, axis=1)  # noqa: F841


def rowwise_softmax(x, out):
    m = ntl.max(x, axis=1)
    e = ntl.exp(x - m[:, None])
    out = e / ntl.sum(e, axis=1)[:, None]  # noqa: F841


def rowwise_layernorm(x, weight, bias, out):
    mean = ntl.sum(x, axis=1) / 32.0
    mean_square = ntl.sum(x * x, axis=1) / 32.0
    var = mean_square - mean * mean
    out = (  # noqa: F841
        (x - mean[:, None]) * ntl.rsqrt(var[:, None] + 1e-05) * weight + bias
    )


def _ssa(func, tensors: tuple[TensorSpec, ...] | None = None) -> ssa.Program:
    if tensors is None:
        tensors = tuple(
            (
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name=name)
                for name in inspect.signature(func).parameters
            )
        )

    program = from_application(func, tensor_irs=tensors, kind=func.__name__)
    assert program is not None

    return program


def _walk(program: ssa.Program) -> tuple[ssa.Operation, ...]:
    ops: list[ssa.Operation] = []

    def visit(operation: ssa.Operation) -> None:
        ops.append(operation)

        for region in operation.regions:
            for inner in region.operations:
                visit(inner)

    for block in program.blocks:
        for operation in block.operations:
            visit(operation)
    return tuple(ops)


def _opcodes(program: ssa.Program) -> tuple[str, ...]:
    return tuple((operation.opcode for operation in _walk(program)))


class TestLoweringInference:
    def _assert_no_coarse_ir_nodes(self, program: ssa.Program) -> None:
        rendered = ssa.render(program)
        assert "ReductionOpIR" not in rendered
        assert "MatmulOpIR" not in rendered
        assert "FlashAttentionOpIR" not in rendered
        assert program.metadata["source"] == "application_ast"
        assert not program.metadata["coarse_operator_nodes"]

    def test_fill_copy_and_assignment_calls_lower_to_ssa_effects(self):
        for func in (
            fill_statement,
            negative_fill_assignment,
            copy_statement,
            plain_copy_assignment,
        ):
            program = _ssa(func)
            assert "mem.store" in _opcodes(program)
            self._assert_no_coarse_ir_nodes(program)

    def test_reductions_lower_to_ssa_reduce_ops(self):
        cases = (
            (sum_statement, "reduce.sum"),
            (method_sum_assignment, "reduce.sum"),
            (max_assignment, "reduce.max"),
            (dot_reduction_assignment, "reduce.sum"),
        )

        for func, opcode in cases:
            program = _ssa(func)
            opcodes = _opcodes(program)
            assert opcode in opcodes
            assert "mem.store" in opcodes
            self._assert_no_coarse_ir_nodes(program)

    def test_transpose_and_matmul_lower_to_ssa_compute_ops(self):
        cases = (
            (transpose_assignment, "linalg.transpose"),
            (matmul_statement, "linalg.matmul"),
            (matmul_assignment, "linalg.matmul"),
        )

        for func, opcode in cases:
            program = _ssa(func)
            opcodes = _opcodes(program)
            assert opcode in opcodes
            assert "mem.store" in opcodes
            self._assert_no_coarse_ir_nodes(program)

    def test_unknown_intrinsic_names_stay_as_call_ops_not_coarse_attention_ir(self):
        tensors = tuple(
            (
                TensorSpec(ndim=2, shape=("rows", "cols"), dtype="float32", name=name)
                for name in ("q", "k", "v", "out")
            )
        )
        program = _ssa(flash_attention_call_name, tensors)
        assert "call.flash_attention" in _opcodes(program)
        self._assert_no_coarse_ir_nodes(program)

    def test_multi_output_and_scalar_math_are_generic_ssa(self):
        for func, fragments in (
            (multi_output_elementwise, ("arith.mul", "arith.sub", "arith.add")),
            (bitwise_shift, ("arith.bitwise_left_shift",)),
            (compare_float_inf, ("cmp.eq", "cmp.ne", "arith.bitwise_and")),
        ):
            program = _ssa(func)
            opcodes = _opcodes(program)

            for fragment in fragments:
                assert fragment in opcodes

            assert "mem.store" in opcodes
            self._assert_no_coarse_ir_nodes(program)

    def test_method_namespace_and_dim_aliases_normalize_to_generic_ssa(self):
        cases = (
            (
                method_math_and_dim_alias,
                ("math.sqrt", "reduce.sum", "math.exp", "arith.div", "mem.store"),
            ),
            (
                namespace_math_alias,
                ("math.exp", "math.sqrt", "arith.add", "mem.store"),
            ),
        )

        for func, expected_opcodes in cases:
            program = _ssa(func)
            assert _opcodes(program) == expected_opcodes
            self._assert_no_coarse_ir_nodes(program)

            if func is method_math_and_dim_alias:
                reduction = next(
                    op for op in _walk(program) if op.opcode == "reduce.sum"
                )
                assert reduction.attrs["axis"] == 0

    def test_offsets_lower_to_explicit_index_ops(self):
        program = _ssa(
            eye_offsets,
            (TensorSpec(ndim=2, shape=("rows", "cols"), dtype="float32", name="out"),),
        )
        opcodes = _opcodes(program)
        assert opcodes.count("index.offset") == 2
        assert "cmp.eq" in opcodes
        assert "mem.store" in opcodes
        self._assert_no_coarse_ir_nodes(program)

    def test_axis_reductions_are_not_shape_special_cased(self):
        for func, axis in ((axis_zero_call, 0), (rowwise_sum, 1), (rowwise_mean, 1)):
            program = _ssa(func)
            reduce_ops = [op for op in _walk(program) if op.opcode == "reduce.sum"]
            assert reduce_ops
            assert reduce_ops[0].attrs.get("axis") == axis
            self._assert_no_coarse_ir_nodes(program)

    def test_axis_reduction_fusions_lower_to_generic_dataflow(self):
        tensors = (
            TensorSpec(ndim=2, shape=("rows", "cols"), dtype="float32", name="x"),
            TensorSpec(ndim=1, shape=("rows",), dtype="float32", name="out0"),
            TensorSpec(ndim=1, shape=("rows",), dtype="float32", name="out1"),
        )
        program = _ssa(rowwise_aminmax, tensors)
        opcodes = _opcodes(program)
        assert "reduce.min" in opcodes
        assert "reduce.max" in opcodes
        assert opcodes.count("mem.store") == 2
        self._assert_no_coarse_ir_nodes(program)

    def test_rowwise_softmax_and_layernorm_are_dataflow_not_kernel_nodes(self):
        softmax_tensors = (
            TensorSpec(ndim=2, shape=("rows", "cols"), dtype="float32", name="x"),
            TensorSpec(ndim=2, shape=("rows", "cols"), dtype="float32", name="out"),
        )
        layernorm_tensors = (
            TensorSpec(ndim=2, shape=("rows", "cols"), dtype="float32", name="x"),
            TensorSpec(ndim=1, shape=("cols",), dtype="float32", name="weight"),
            TensorSpec(ndim=1, shape=("cols",), dtype="float32", name="bias"),
            TensorSpec(ndim=2, shape=("rows", "cols"), dtype="float32", name="out"),
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
            program = _ssa(func, tensors)
            opcodes = _opcodes(program)

            for fragment in fragments:
                assert fragment in opcodes

            assert "mem.store" in opcodes
            self._assert_no_coarse_ir_nodes(program)

    def test_ssa_textual_rendering_is_the_audit_format(self):
        rendered = ssa.render(_ssa(rowwise_addmv))
        assert rendered.startswith("ssa @rowwise_addmv {")
        assert "reduce.sum" in rendered
        assert "mem.store" in rendered
