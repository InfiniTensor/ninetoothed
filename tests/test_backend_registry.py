import unittest

from ninetoothed.backends import (
    BackendName,
    backend_capabilities,
    lower,
    normalize_backend_name,
    normalize_backend_options,
)
from ninetoothed.ir import (
    AxisReductionAssignOpIR,
    CopyOpIR,
    ElementwiseAssignOpIR,
    ElementwiseBinaryOpIR,
    ExprIR,
    FillOpIR,
    KernelIR,
    LaunchIR,
    MatmulOpIR,
    ProgramIR,
    ReductionOpIR,
    RowwiseAssignOpIR,
    TensorTypeIR,
    TransposeOpIR,
    program_to_ssa,
)


def _kernel_ir():
    return KernelIR(
        kernel_name="add",
        source="@triton.jit\ndef add(x, y, out):\n    return\n",
        entrypoint="add",
        launch=LaunchIR(
            name="launch_add", args=("x", "y", "out"), grid="lambda meta: (1,)"
        ),
        tensors=(
            TensorTypeIR("x", 1, dtype="float32", shape=("n",)),
            TensorTypeIR("y", 1, dtype="float32", shape=("n",)),
            TensorTypeIR("out", 1, dtype="float32", shape=("n",)),
        ),
        compiler_options={"num_warps": 4, "num_stages": 3},
    )


def _structured_binary_ir(operator="add"):
    return _with_ssa(
        KernelIR(
            kernel_name=operator,
            source=f"@triton.jit\ndef {operator}(x, y, out):\n    return\n",
            entrypoint=operator,
            launch=LaunchIR(
                name="launch_add", args=("x", "y", "out"), grid="lambda meta: (1,)"
            ),
            tensors=(
                TensorTypeIR("x", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("y", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("out", 1, dtype="float32", shape=("n",)),
            ),
            program=ProgramIR(
                kind="elementwise",
                operations=(
                    ElementwiseBinaryOpIR(
                        operator=operator, lhs="x", rhs="y", output="out", extent="n"
                    ),
                ),
            ),
        )
    )


def _structured_add_ir():
    return _structured_binary_ir("add")


def _structured_float16_add_ir():
    return _with_ssa(
        KernelIR(
            kernel_name="add_fp16",
            source="@triton.jit\ndef add_fp16(x, y, out):\n    return\n",
            entrypoint="add_fp16",
            launch=LaunchIR(
                name="launch_add_fp16",
                args=("x", "y", "out"),
                grid="lambda meta: (1,)",
            ),
            tensors=(
                TensorTypeIR("x", 1, dtype="float16", shape=("n",)),
                TensorTypeIR("y", 1, dtype="float16", shape=("n",)),
                TensorTypeIR("out", 1, dtype="float16", shape=("n",)),
            ),
            program=ProgramIR(
                kind="elementwise",
                operations=(
                    ElementwiseBinaryOpIR(
                        operator="add", lhs="x", rhs="y", output="out", extent="n"
                    ),
                ),
            ),
        )
    )


def _with_ssa(kernel: KernelIR) -> KernelIR:
    if kernel.program is None:
        raise ValueError("test fixture requires ProgramIR before attaching SSA")
    return type(kernel)(
        kernel_name=kernel.kernel_name,
        source=kernel.source,
        source_path=kernel.source_path,
        source_language=kernel.source_language,
        entrypoint=kernel.entrypoint,
        launch=kernel.launch,
        tensors=kernel.tensors,
        compiler_options=kernel.compiler_options,
        metadata=kernel.metadata,
        program=kernel.program,
        ssa=program_to_ssa(kernel.program, kernel.tensors),
    )


def _without_ssa(kernel: KernelIR) -> KernelIR:
    return type(kernel)(
        kernel_name=kernel.kernel_name,
        source=kernel.source,
        source_path=kernel.source_path,
        source_language=kernel.source_language,
        entrypoint=kernel.entrypoint,
        launch=kernel.launch,
        tensors=kernel.tensors,
        compiler_options=kernel.compiler_options,
        metadata=kernel.metadata,
        program=kernel.program,
        ssa=None,
    )


def _structured_expression_ir():
    expression = ExprIR(
        kind="call",
        value="where",
        args=(
            ExprIR(
                kind="binary",
                value="gt",
                args=(ExprIR(kind="var", value="x"), ExprIR(kind="const", value=0.0)),
            ),
            ExprIR(
                kind="call",
                value="exp",
                args=(
                    ExprIR(
                        kind="unary", value="neg", args=(ExprIR(kind="var", value="x"),)
                    ),
                ),
            ),
            ExprIR(kind="call", value="sqrt", args=(ExprIR(kind="var", value="y"),)),
        ),
    )
    return _with_ssa(
        KernelIR(
            kernel_name="expr",
            source="@triton.jit\ndef expr(x, y, out):\n    return\n",
            entrypoint="expr",
            launch=LaunchIR(
                name="launch_expr", args=("x", "y", "out"), grid="lambda meta: (1,)"
            ),
            tensors=(
                TensorTypeIR("x", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("y", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("out", 1, dtype="float32", shape=("n",)),
            ),
            program=ProgramIR(
                kind="elementwise",
                operations=(
                    ElementwiseAssignOpIR(output="out", expression=expression),
                ),
            ),
        )
    )


def _structured_multi_output_ir():
    out0_expr = ExprIR(
        kind="binary",
        value="sub",
        args=(
            ExprIR(
                kind="binary",
                value="mul",
                args=(ExprIR(kind="var", value="x0"), ExprIR(kind="var", value="cos")),
            ),
            ExprIR(
                kind="binary",
                value="mul",
                args=(ExprIR(kind="var", value="x1"), ExprIR(kind="var", value="sin")),
            ),
        ),
    )
    out1_expr = ExprIR(
        kind="binary",
        value="add",
        args=(
            ExprIR(
                kind="binary",
                value="mul",
                args=(ExprIR(kind="var", value="x0"), ExprIR(kind="var", value="sin")),
            ),
            ExprIR(
                kind="binary",
                value="mul",
                args=(ExprIR(kind="var", value="x1"), ExprIR(kind="var", value="cos")),
            ),
        ),
    )
    return _with_ssa(
        KernelIR(
            kernel_name="multi_output",
            source="@triton.jit\ndef multi_output(x0, x1, cos, sin, out0, out1):\n    return\n",
            tensors=(
                TensorTypeIR("x0", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("x1", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("cos", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("sin", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("out0", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("out1", 1, dtype="float32", shape=("n",)),
            ),
            program=ProgramIR(
                kind="elementwise",
                operations=(
                    ElementwiseAssignOpIR(output="out0", expression=out0_expr),
                    ElementwiseAssignOpIR(output="out1", expression=out1_expr),
                ),
            ),
        )
    )


def _structured_offsets_ir():
    expression = ExprIR(
        kind="binary",
        value="eq",
        args=(
            ExprIR(kind="offset", value={"tensor": "out", "dim": 0}),
            ExprIR(kind="offset", value={"tensor": "out", "dim": 1}),
        ),
    )
    return _with_ssa(
        KernelIR(
            kernel_name="eye_offsets",
            source="@triton.jit\ndef eye_offsets(out):\n    return\n",
            tensors=(TensorTypeIR("out", 2, dtype="float32", shape=("rows", "cols")),),
            program=ProgramIR(
                kind="elementwise",
                operations=(
                    ElementwiseAssignOpIR(output="out", expression=expression),
                ),
            ),
        )
    )


def _structured_special_calls_ir():
    expression = ExprIR(
        kind="binary",
        value="add",
        args=(
            ExprIR(kind="call", value="exp2", args=(ExprIR(kind="var", value="x"),)),
            ExprIR(
                kind="call",
                value="_atan2_approx",
                args=(ExprIR(kind="var", value="y"), ExprIR(kind="const", value=1.0)),
            ),
        ),
    )
    return _with_ssa(
        KernelIR(
            kernel_name="special_calls",
            source="@triton.jit\ndef special_calls(x, y, out):\n    return\n",
            tensors=(
                TensorTypeIR("x", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("y", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("out", 1, dtype="float32", shape=("n",)),
            ),
            program=ProgramIR(
                kind="elementwise",
                operations=(
                    ElementwiseAssignOpIR(output="out", expression=expression),
                ),
            ),
        )
    )


def _structured_fill_ir():
    return _with_ssa(
        KernelIR(
            kernel_name="fill",
            source="@triton.jit\ndef fill(out):\n    return\n",
            tensors=(TensorTypeIR("out", 1, dtype="float32", shape=("n",)),),
            program=ProgramIR(
                kind="fill",
                operations=(FillOpIR(output="out", value=3.25, extent="n"),),
            ),
        )
    )


def _structured_copy_ir():
    return _with_ssa(
        KernelIR(
            kernel_name="copy",
            source="@triton.jit\ndef copy(x, out):\n    return\n",
            tensors=(
                TensorTypeIR("x", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("out", 1, dtype="float32", shape=("n",)),
            ),
            program=ProgramIR(
                kind="copy",
                operations=(CopyOpIR(input="x", output="out", extent="n"),),
            ),
        )
    )


def _structured_reduction_ir(operator="sum"):
    return _with_ssa(
        KernelIR(
            kernel_name=f"reduce_{operator}",
            source=f"@triton.jit\ndef reduce_{operator}(x, out):\n    return\n",
            tensors=(
                TensorTypeIR("x", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("out", 1, dtype="float32", shape=("1",)),
            ),
            program=ProgramIR(
                kind="reduction",
                operations=(
                    ReductionOpIR(
                        operator=operator, input="x", output="out", extent="n"
                    ),
                ),
            ),
        )
    )


def _structured_dot_reduction_ir():
    expression = ExprIR(
        kind="binary",
        value="mul",
        args=(ExprIR(kind="var", value="x"), ExprIR(kind="var", value="y")),
    )
    return _with_ssa(
        KernelIR(
            kernel_name="dot_reduce",
            source="@triton.jit\ndef dot_reduce(x, y, out):\n    return\n",
            tensors=(
                TensorTypeIR("x", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("y", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("out", 1, dtype="float32", shape=("1",)),
            ),
            program=ProgramIR(
                kind="reduction",
                operations=(
                    ReductionOpIR(
                        operator="sum",
                        input="",
                        output="out",
                        extent="n",
                        expression=expression,
                    ),
                ),
            ),
        )
    )


def _structured_axis_reduction_ir():
    sum_expr = ExprIR(
        kind="axis_reduce",
        value={"operator": "sum", "axis": 1},
        args=(
            ExprIR(
                kind="binary",
                value="mul",
                args=(ExprIR(kind="var", value="a"), ExprIR(kind="var", value="x")),
            ),
        ),
    )
    expression = ExprIR(
        kind="binary",
        value="add",
        args=(ExprIR(kind="var", value="bias"), sum_expr),
    )
    return _with_ssa(
        KernelIR(
            kernel_name="axis_addmv",
            source="@triton.jit\ndef axis_addmv(bias, a, x, out):\n    return\n",
            tensors=(
                TensorTypeIR("bias", 1, dtype="float32", shape=("rows",)),
                TensorTypeIR("a", 2, dtype="float32", shape=("rows", "cols")),
                TensorTypeIR("x", 1, dtype="float32", shape=("cols",)),
                TensorTypeIR("out", 1, dtype="float32", shape=("rows",)),
            ),
            program=ProgramIR(
                kind="axis_reduction",
                operations=(
                    AxisReductionAssignOpIR(output="out", expression=expression),
                ),
            ),
        )
    )


def _structured_axis_multi_reduction_ir():
    min_expr = ExprIR(
        kind="axis_reduce",
        value={"operator": "min", "axis": 1},
        args=(ExprIR(kind="var", value="x"),),
    )
    max_expr = ExprIR(
        kind="axis_reduce",
        value={"operator": "max", "axis": 1},
        args=(ExprIR(kind="var", value="x"),),
    )
    return _with_ssa(
        KernelIR(
            kernel_name="axis_aminmax",
            source="@triton.jit\ndef axis_aminmax(x, out0, out1):\n    return\n",
            tensors=(
                TensorTypeIR("x", 2, dtype="float32", shape=("rows", "cols")),
                TensorTypeIR("out0", 1, dtype="float32", shape=("rows",)),
                TensorTypeIR("out1", 1, dtype="float32", shape=("rows",)),
            ),
            program=ProgramIR(
                kind="axis_reduction",
                operations=(
                    AxisReductionAssignOpIR(output="out0", expression=min_expr),
                    AxisReductionAssignOpIR(output="out1", expression=max_expr),
                ),
            ),
        )
    )


def _structured_rowwise_softmax_ir():
    x = ExprIR(kind="var", value="x")
    max_expr = ExprIR(
        kind="axis_reduce",
        value={"operator": "max", "axis": 1},
        args=(x,),
    )
    shifted = ExprIR(kind="binary", value="sub", args=(x, max_expr))
    exp_expr = ExprIR(kind="call", value="exp", args=(shifted,))
    sum_expr = ExprIR(
        kind="axis_reduce",
        value={"operator": "sum", "axis": 1},
        args=(exp_expr,),
    )
    output = ExprIR(kind="binary", value="div", args=(exp_expr, sum_expr))
    return _with_ssa(
        KernelIR(
            kernel_name="rowwise_softmax",
            source="@triton.jit\ndef rowwise_softmax(x, out):\n    return\n",
            tensors=(
                TensorTypeIR("x", 2, dtype="float32", shape=("rows", "cols")),
                TensorTypeIR("out", 2, dtype="float32", shape=("rows", "cols")),
            ),
            program=ProgramIR(
                kind="rowwise",
                operations=(RowwiseAssignOpIR(output="out", expression=output),),
            ),
        )
    )


def _structured_rowwise_layernorm_ir():
    x = ExprIR(kind="var", value="x")
    mean = ExprIR(
        kind="binary",
        value="div",
        args=(
            ExprIR(
                kind="axis_reduce",
                value={"operator": "sum", "axis": 1},
                args=(x,),
            ),
            ExprIR(kind="const", value=32.0),
        ),
    )
    centered = ExprIR(kind="binary", value="sub", args=(x, mean))
    variance = ExprIR(
        kind="binary",
        value="div",
        args=(
            ExprIR(
                kind="axis_reduce",
                value={"operator": "sum", "axis": 1},
                args=(ExprIR(kind="binary", value="mul", args=(centered, centered)),),
            ),
            ExprIR(kind="const", value=32.0),
        ),
    )
    inv_std = ExprIR(
        kind="call",
        value="rsqrt",
        args=(
            ExprIR(
                kind="binary",
                value="add",
                args=(variance, ExprIR(kind="const", value=1.0e-5)),
            ),
        ),
    )
    normalized = ExprIR(kind="binary", value="mul", args=(centered, inv_std))
    scaled = ExprIR(
        kind="binary",
        value="mul",
        args=(normalized, ExprIR(kind="var", value="weight")),
    )
    output = ExprIR(
        kind="binary",
        value="add",
        args=(scaled, ExprIR(kind="var", value="bias")),
    )
    return _with_ssa(
        KernelIR(
            kernel_name="rowwise_layernorm",
            source="@triton.jit\ndef rowwise_layernorm(x, weight, bias, out):\n    return\n",
            tensors=(
                TensorTypeIR("x", 2, dtype="float32", shape=("rows", "cols")),
                TensorTypeIR("weight", 1, dtype="float32", shape=("cols",)),
                TensorTypeIR("bias", 1, dtype="float32", shape=("cols",)),
                TensorTypeIR("out", 2, dtype="float32", shape=("rows", "cols")),
            ),
            program=ProgramIR(
                kind="rowwise",
                operations=(RowwiseAssignOpIR(output="out", expression=output),),
            ),
        )
    )


def _structured_matmul_ir():
    return _with_ssa(
        KernelIR(
            kernel_name="matmul",
            source="@triton.jit\ndef matmul(a, b, out):\n    return\n",
            tensors=(
                TensorTypeIR("a", 2, dtype="float32", shape=("m", "k")),
                TensorTypeIR("b", 2, dtype="float32", shape=("k", "n")),
                TensorTypeIR("out", 2, dtype="float32", shape=("m", "n")),
            ),
            program=ProgramIR(
                kind="matmul",
                operations=(
                    MatmulOpIR(lhs="a", rhs="b", output="out", m="m", n="n", k="k"),
                ),
            ),
        )
    )


def _structured_transpose_ir():
    return _with_ssa(
        KernelIR(
            kernel_name="transpose",
            source="@triton.jit\ndef transpose(x, out):\n    return\n",
            tensors=(
                TensorTypeIR("x", 2, dtype="float32", shape=("rows", "cols")),
                TensorTypeIR("out", 2, dtype="float32", shape=("cols", "rows")),
            ),
            program=ProgramIR(
                kind="transpose",
                operations=(
                    TransposeOpIR(input="x", output="out", rows="rows", cols="cols"),
                ),
            ),
        )
    )


class BackendRegistryTest(unittest.TestCase):
    def test_backend_aliases_are_normalized(self):
        self.assertEqual(normalize_backend_name(None), BackendName.TRITON)
        self.assertEqual(normalize_backend_name("tl"), BackendName.TRITON)
        self.assertEqual(normalize_backend_name("tile-lang"), BackendName.TILELANG)
        self.assertEqual(normalize_backend_name("cu"), BackendName.CUDA)
        self.assertEqual(normalize_backend_name("tvm-script"), BackendName.TVM)

    def test_backend_options_keep_caller_and_extra_values(self):
        options = normalize_backend_options(
            "cuda", caller="cuda", emit_only=False, arch="sm_90"
        )

        self.assertEqual(options.name, BackendName.CUDA)
        self.assertEqual(options.caller, "cuda")
        self.assertFalse(options.emit_only)
        self.assertEqual(options.extra["arch"], "sm_90")

    def test_default_registry_reports_four_backends(self):
        names = {capability.name for capability in backend_capabilities()}

        self.assertEqual(
            names,
            {
                BackendName.TRITON,
                BackendName.TILELANG,
                BackendName.CUDA,
                BackendName.TVM,
            },
        )

    def test_triton_backend_rejects_source_only_kernel_without_ssa(self):
        with self.assertRaisesRegex(ValueError, "requires SSAProgramIR"):
            lower(_kernel_ir(), "triton")

    def test_triton_backend_lowers_structured_elementwise_add_from_ssa(self):
        artifact = lower(_structured_add_ir(), "triton")

        self.assertTrue(artifact.executable)
        self.assertEqual(artifact.language, "python/triton")
        self.assertIn("Lowering IR: SSAProgramIR", artifact.primary_source)
        self.assertIn(
            "tl.load(x + index, mask=mask, other=0.0)", artifact.primary_source
        )
        self.assertIn(
            "tl.load(y + index, mask=mask, other=0.0)", artifact.primary_source
        )
        self.assertIn(
            "v0 = (tl.load(x + index, mask=mask, other=0.0) + tl.load(y + index, mask=mask, other=0.0))",
            artifact.primary_source,
        )
        self.assertIn("tl.store(out + index, v0, mask=mask)", artifact.primary_source)
        self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
        self.assertEqual(
            artifact.metadata["source_route"], "ssa-unified-triton-emitter"
        )
        self.assertEqual(artifact.metadata["ssa_metadata"]["target_backend"], "triton")
        self.assertNotIn("def add(x, y, out):\n    return", artifact.primary_source)

    def test_triton_backend_can_consume_ssa_elementwise_add(self):
        artifact = lower(_with_ssa(_structured_add_ir()), "triton")

        self.assertTrue(artifact.executable)
        self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
        self.assertIn("Lowering IR: SSAProgramIR", artifact.primary_source)
        self.assertIn("tl.store(out + index, v0, mask=mask)", artifact.primary_source)

    def test_backend_registry_does_not_convert_program_ir_without_explicit_ssa(self):
        for backend in ("triton", "cuda", "tilelang", "tvm"):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, "requires SSAProgramIR"):
                    lower(_without_ssa(_structured_add_ir()), backend)

    def test_triton_backend_lowers_structured_memory_and_compute_ops(self):
        kernels = [
            (
                "fill",
                _structured_fill_ir(),
                ("v0 = 3.25", "tl.store(out + index, v0, mask=mask)"),
            ),
            (
                "copy",
                _structured_copy_ir(),
                ("tl.load(x + index", "tl.store(out + index"),
            ),
            (
                "reduction",
                _structured_reduction_ir("sum"),
                ("for v0_i in range(0, n, 1):", "tl.store(out + index, v0"),
            ),
            (
                "transpose",
                _structured_transpose_ir(),
                ("tl.load(x + (v2) * (cols) + (v1)", "tl.store(out + index"),
            ),
            (
                "matmul",
                _structured_matmul_ir(),
                (
                    "for v10_i in range(0, k, 1):",
                    "tl.load(a + (v1_v10_body) * (k) + (v10_i)",
                ),
            ),
        ]

        for name, kernel, fragments in kernels:
            with self.subTest(name=name):
                artifact = lower(kernel, "triton")

                self.assertTrue(artifact.executable)
                self.assertEqual(artifact.language, "python/triton")
                self.assertIn("Lowering IR: SSAProgramIR", artifact.primary_source)
                self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
                self.assertIn(
                    artifact.metadata["source_route"],
                    {
                        "ssa-unified-triton-emitter",
                        "ssa-unified-triton-emitter",
                        "ssa-unified-triton-emitter",
                        "ssa-unified-triton-emitter",
                        "ssa-unified-triton-emitter",
                    },
                )
                self.assertIn("ssa_optimization", artifact.metadata)
                for fragment in fragments:
                    self.assertIn(fragment, artifact.primary_source)

    def test_cuda_backend_rejects_source_only_kernel_without_ssa(self):
        with self.assertRaisesRegex(ValueError, "requires SSAProgramIR"):
            lower(_kernel_ir(), "cuda")

    def test_cuda_backend_lowers_structured_elementwise_add(self):
        artifact = lower(_structured_add_ir(), "cuda")

        self.assertTrue(artifact.executable)
        self.assertEqual(artifact.language, "cuda/c++")
        self.assertIn("add_kernel", artifact.primary_source)
        self.assertIn("float v0 = (x[index] + y[index]);", artifact.primary_source)
        self.assertIn("out[index] = v0;", artifact.primary_source)
        self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
        self.assertEqual(artifact.metadata["ssa_metadata"]["target_backend"], "cuda")
        self.assertNotIn("return -1", artifact.primary_source)

    def test_cuda_backend_includes_fp16_header_for_half_artifacts(self):
        artifact = lower(_structured_float16_add_ir(), "cuda")

        self.assertIn("#include <cuda_fp16.h>", artifact.primary_source)
        self.assertIn("const half* __restrict__ x", artifact.primary_source)
        self.assertIn("half* __restrict__ out", artifact.primary_source)

    def test_cuda_backend_can_consume_ssa_elementwise_add(self):
        artifact = lower(_with_ssa(_structured_add_ir()), "cuda")

        self.assertTrue(artifact.executable)
        self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
        self.assertIn("Lowering IR: SSAProgramIR", artifact.primary_source)
        self.assertIn("float v0 = (x[index] + y[index]);", artifact.primary_source)
        self.assertIn("out[index] = v0;", artifact.primary_source)

    def test_cuda_backend_omits_trailing_signature_comma_without_shape_params(self):
        kernel = _with_ssa(
            KernelIR(
                kernel_name="static_add",
                source="@triton.jit\ndef static_add(x, y, out):\n    return\n",
                entrypoint="static_add",
                launch=LaunchIR(
                    name="launch_static_add",
                    args=("x", "y", "out"),
                    grid="lambda meta: (1,)",
                ),
                tensors=(
                    TensorTypeIR("x", 1, dtype="float32", shape=("16",)),
                    TensorTypeIR("y", 1, dtype="float32", shape=("16",)),
                    TensorTypeIR("out", 1, dtype="float32", shape=("16",)),
                ),
                program=ProgramIR(
                    kind="elementwise",
                    operations=(
                        ElementwiseBinaryOpIR(
                            operator="add",
                            lhs="x",
                            rhs="y",
                            output="out",
                            extent="16",
                        ),
                    ),
                ),
            )
        )
        artifact = lower(kernel, "cuda")

        self.assertEqual(artifact.metadata["shape_params"], ())
        self.assertNotIn(",\n)", artifact.primary_source)
        self.assertIn(
            "float* __restrict__ out\n) {",
            artifact.primary_source,
        )
        self.assertIn("float* out,\n    cudaStream_t stream", artifact.primary_source)

    def test_tilelang_backend_rejects_source_only_kernel_without_ssa(self):
        with self.assertRaisesRegex(ValueError, "requires SSAProgramIR"):
            lower(_kernel_ir(), "tilelang")

    def test_tilelang_backend_lowers_structured_elementwise_add(self):
        artifact = lower(_structured_add_ir(), "tilelang")

        self.assertTrue(artifact.executable)
        self.assertEqual(artifact.language, "python/tilelang")
        self.assertIn("@T.prim_func", artifact.primary_source)
        self.assertIn(
            'T.thread_binding((n + 255) // 256, thread="blockIdx.x")',
            artifact.primary_source,
        )
        self.assertIn(
            'T.thread_binding(256, thread="threadIdx.x")', artifact.primary_source
        )
        self.assertIn("v0 = (x_buf[index] + y_buf[index])", artifact.primary_source)
        self.assertIn("out_buf[index] = v0", artifact.primary_source)
        self.assertEqual(
            artifact.metadata["source_route"], "ssa-unified-tilelang-emitter"
        )
        self.assertNotIn("NotImplementedError", artifact.primary_source)

    def test_tvm_backend_rejects_source_only_kernel_without_ssa(self):
        with self.assertRaisesRegex(ValueError, "requires SSAProgramIR"):
            lower(_kernel_ir(), "tvm")

    def test_tvm_backend_lowers_structured_elementwise_add(self):
        artifact = lower(_structured_add_ir(), "tvm")

        self.assertTrue(artifact.executable)
        self.assertEqual(artifact.language, "python/tvm-script")
        self.assertIn("@tvm.script.ir_module", artifact.primary_source)
        self.assertIn(
            "T.thread_binding(((n + T.int64(255)) // T.int64(256))",
            artifact.primary_source,
        )
        self.assertIn("'global_symbol': 'main'", artifact.primary_source)
        self.assertIn("v0 = (x_buf[index] + y_buf[index])", artifact.primary_source)
        self.assertIn("out_buf[index] = v0", artifact.primary_source)
        self.assertEqual(artifact.metadata["source_route"], "ssa-unified-tvm-emitter")
        self.assertNotIn("NotImplementedError", artifact.primary_source)

    def test_structured_elementwise_binary_operators_are_executable(self):
        expected = {
            "sub": "-",
            "mul": "*",
            "div": "/",
        }

        for backend in ("cuda", "tilelang", "tvm"):
            for operator, symbol in expected.items():
                with self.subTest(backend=backend, operator=operator):
                    artifact = lower(_structured_binary_ir(operator), backend)

                    self.assertTrue(artifact.executable)
                    self.assertIn(symbol, artifact.primary_source)
                    self.assertNotIn("NotImplementedError", artifact.primary_source)

    def test_structured_elementwise_expression_is_executable(self):
        expected_fragments = {
            "cuda": ("expf", "sqrtf", "?"),
            "tilelang": ("T.exp", "T.sqrt", "T.if_then_else"),
            "tvm": ("T.exp", "T.sqrt", "T.if_then_else"),
        }

        for backend, fragments in expected_fragments.items():
            with self.subTest(backend=backend):
                artifact = lower(_structured_expression_ir(), backend)

                self.assertTrue(artifact.executable)
                for fragment in fragments:
                    self.assertIn(fragment, artifact.primary_source)
                self.assertNotIn("NotImplementedError", artifact.primary_source)

    def test_structured_multi_output_elementwise_is_executable(self):
        expected_fragments = {
            "cuda": ("out0[index] =", "out1[index] ="),
            "tilelang": ("out0_buf[index] =", "out1_buf[index] ="),
            "tvm": ("out0_buf[index] =", "out1_buf[index] ="),
        }

        for backend, fragments in expected_fragments.items():
            with self.subTest(backend=backend):
                artifact = lower(_structured_multi_output_ir(), backend)

                self.assertTrue(artifact.executable)
                for fragment in fragments:
                    self.assertIn(fragment, artifact.primary_source)
                self.assertNotIn("NotImplementedError", artifact.primary_source)

    def test_structured_offsets_elementwise_is_executable(self):
        expected_fragments = {
            "cuda": ("int64_t rows", "int64_t cols", "index / (cols)", "index % cols"),
            "tilelang": (
                "rows: T.int64",
                "cols: T.int64",
                "index // (cols)",
                "index % cols",
            ),
            "tvm": (
                "rows: T.int64",
                "cols: T.int64",
                "index // (cols)",
                "index % cols",
            ),
        }

        for backend, fragments in expected_fragments.items():
            with self.subTest(backend=backend):
                artifact = lower(_structured_offsets_ir(), backend)

                self.assertTrue(artifact.executable)
                for fragment in fragments:
                    self.assertIn(fragment, artifact.primary_source)
                self.assertNotIn("Unsupported", artifact.primary_source)

    def test_structured_special_pointwise_calls_are_executable(self):
        expected_fragments = {
            "cuda": ("exp2f", "atan2f"),
            "tilelang": ("T.exp2", "T.atan2"),
            "tvm": ("T.exp2", "T.atan2"),
        }

        for backend, fragments in expected_fragments.items():
            with self.subTest(backend=backend):
                artifact = lower(_structured_special_calls_ir(), backend)

                self.assertTrue(artifact.executable)
                for fragment in fragments:
                    self.assertIn(fragment, artifact.primary_source)
                self.assertNotIn("Unsupported", artifact.primary_source)

    def test_structured_memory_and_shape_ops_are_executable(self):
        kernels = [
            ("fill", _structured_fill_ir(), ("3.25",)),
            ("copy", _structured_copy_ir(), ("out", "x")),
            ("transpose", _structured_transpose_ir(), ("row", "col")),
        ]

        for backend in ("cuda", "tilelang", "tvm"):
            for name, kernel, fragments in kernels:
                with self.subTest(backend=backend, name=name):
                    artifact = lower(kernel, backend)

                    self.assertTrue(artifact.executable)
                    for fragment in fragments:
                        self.assertIn(fragment, artifact.primary_source)
                    self.assertNotIn("NotImplementedError", artifact.primary_source)

    def test_structured_reductions_are_executable(self):
        expected_fragments = {
            "sum": ("0.0", "+"),
            "max": ("3.4028234663852886e+38", "max"),
        }

        for backend in ("cuda", "tilelang", "tvm"):
            for operator, fragments in expected_fragments.items():
                with self.subTest(backend=backend, operator=operator):
                    artifact = lower(_structured_reduction_ir(operator), backend)

                    self.assertTrue(artifact.executable)
                    for fragment in fragments:
                        self.assertIn(fragment, artifact.primary_source)
                    self.assertNotIn("NotImplementedError", artifact.primary_source)

    def test_structured_expression_reduction_is_executable(self):
        expected_fragments = {
            "cuda": (
                "const float* __restrict__ x",
                "const float* __restrict__ y",
                "x[v1_i] * y[v1_i]",
            ),
            "tilelang": ("x_buf[v1_i] * y_buf[v1_i]", "for v1_i in T.serial(n)"),
            "tvm": ("x_buf[v1_i] * y_buf[v1_i]", "for v1_i in T.serial(n)"),
        }

        for backend, fragments in expected_fragments.items():
            with self.subTest(backend=backend):
                artifact = lower(_structured_dot_reduction_ir(), backend)

                self.assertTrue(artifact.executable)
                for fragment in fragments:
                    self.assertIn(fragment, artifact.primary_source)
                self.assertNotIn("Unsupported", artifact.primary_source)

    def test_structured_axis_reductions_are_executable(self):
        expected_fragments = {
            "triton": (
                "for v1_i in range(0, cols, 1):",
                "tl.load(a + (index) * (cols) + (v1_i)",
                "tl.store(out + index",
            ),
            "cuda": (
                "for (int64_t v1_i = 0; v1_i < cols; v1_i += 1)",
                "bias[index]",
                "a[(index) * (cols) + (v1_i)]",
                "x[v1_i]",
            ),
            "tilelang": (
                "for v1_i in T.serial(cols)",
                "bias_buf[index]",
                "a_buf[(index) * (cols) + (v1_i)]",
                "x_buf[v1_i]",
            ),
            "tvm": (
                "for v1_i in T.serial(cols)",
                "bias_buf[index]",
                "a_buf[(index) * (cols) + (v1_i)]",
                "x_buf[v1_i]",
            ),
        }
        expected_routes = {
            "triton": "ssa-unified-triton-emitter",
            "cuda": "ssa-unified-cuda-emitter",
            "tilelang": "ssa-unified-tilelang-emitter",
            "tvm": "ssa-unified-tvm-emitter",
        }

        for backend, fragments in expected_fragments.items():
            with self.subTest(backend=backend):
                artifact = lower(_structured_axis_reduction_ir(), backend)

                self.assertTrue(artifact.executable)
                self.assertEqual(
                    artifact.metadata["source_route"], expected_routes[backend]
                )
                for fragment in fragments:
                    self.assertIn(fragment, artifact.primary_source)
                self.assertNotIn("Unsupported", artifact.primary_source)

    def test_structured_axis_multi_reductions_are_executable(self):
        expected_routes = {
            "triton": "ssa-unified-triton-emitter",
            "cuda": "ssa-unified-cuda-emitter",
            "tilelang": "ssa-unified-tilelang-emitter",
            "tvm": "ssa-unified-tvm-emitter",
        }
        for backend in ("triton", "cuda", "tilelang", "tvm"):
            with self.subTest(backend=backend):
                artifact = lower(_structured_axis_multi_reduction_ir(), backend)

                self.assertTrue(artifact.executable)
                self.assertEqual(
                    artifact.metadata["source_route"], expected_routes[backend]
                )
                self.assertIn("out0", artifact.primary_source)
                self.assertIn("out1", artifact.primary_source)
                self.assertNotIn("Unsupported", artifact.primary_source)

    def test_structured_rowwise_reductions_are_executable(self):
        expected_fragments = {
            "triton": (
                "for v0_i in range(0, cols, 1):",
                "tl.exp",
                "tl.store(out + index",
            ),
            "cuda": (
                "for (int64_t v0_i = 0; v0_i < cols; v0_i += 1)",
                "expf",
                "out[index] =",
            ),
            "tilelang": ("for v0_i in T.serial(cols)", "T.exp", "out_buf[index] ="),
            "tvm": ("for v0_i in T.serial(cols)", "T.exp", "out_buf[index] ="),
        }
        expected_routes = {
            "triton": "ssa-unified-triton-emitter",
            "cuda": "ssa-unified-cuda-emitter",
            "tilelang": "ssa-unified-tilelang-emitter",
            "tvm": "ssa-unified-tvm-emitter",
        }

        for backend, fragments in expected_fragments.items():
            with self.subTest(backend=backend):
                artifact = lower(_structured_rowwise_softmax_ir(), backend)

                self.assertTrue(artifact.executable)
                self.assertEqual(
                    artifact.metadata["source_route"], expected_routes[backend]
                )
                for fragment in fragments:
                    self.assertIn(fragment, artifact.primary_source)
                self.assertNotIn("Unsupported", artifact.primary_source)

    def test_structured_rowwise_vector_parameters_are_column_indexed(self):
        expected_fragments = {
            "triton": (
                "tl.load(weight + (index % cols)",
                "tl.load(bias + (index % cols)",
                "tl.rsqrt",
            ),
            "cuda": ("weight[(index % cols)]", "bias[(index % cols)]", "rsqrtf"),
            "tilelang": (
                "weight_buf[(index % cols)]",
                "bias_buf[(index % cols)]",
                "T.rsqrt",
            ),
            "tvm": (
                "weight_buf[(index % cols)]",
                "bias_buf[(index % cols)]",
                "T.rsqrt",
            ),
        }
        expected_routes = {
            "triton": "ssa-unified-triton-emitter",
            "cuda": "ssa-unified-cuda-emitter",
            "tilelang": "ssa-unified-tilelang-emitter",
            "tvm": "ssa-unified-tvm-emitter",
        }

        for backend, fragments in expected_fragments.items():
            with self.subTest(backend=backend):
                artifact = lower(_structured_rowwise_layernorm_ir(), backend)

                self.assertTrue(artifact.executable)
                self.assertEqual(
                    artifact.metadata["source_route"], expected_routes[backend]
                )
                for fragment in fragments:
                    self.assertIn(fragment, artifact.primary_source)
                self.assertNotIn("Unsupported", artifact.primary_source)

    def test_structured_matmul_is_executable(self):
        expected_fragments = {
            "cuda": (
                "for (int64_t v10_i = 0; v10_i < k; v10_i += 1)",
                "a[(v1_v10_body) * (k) + (v10_i)]",
            ),
            "tilelang": (
                "for v10_i in T.serial(k)",
                "a_buf[(v1_v10_body) * (k) + (v10_i)]",
            ),
            "tvm": ("for v10_i in T.serial(k)", "a_buf[(v1_v10_body) * (k) + (v10_i)]"),
        }

        for backend, fragments in expected_fragments.items():
            with self.subTest(backend=backend):
                artifact = lower(_structured_matmul_ir(), backend)

                self.assertTrue(artifact.executable)
                for fragment in fragments:
                    self.assertIn(fragment, artifact.primary_source)
                self.assertNotIn("NotImplementedError", artifact.primary_source)

    def test_artifact_can_write_all_sources(self):
        import tempfile

        artifact = lower(_structured_add_ir(), "cuda")

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = artifact.write_to(temp_dir)

            self.assertEqual(len(paths), 2)
            self.assertTrue(all(path.exists() for path in paths))


if __name__ == "__main__":
    unittest.main()
