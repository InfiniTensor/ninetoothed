# ruff: noqa: F841
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor, block_size
from ninetoothed.backends import lower as lower_kernel_ir
from ninetoothed.ir import KernelIR, TensorTypeIR
from ninetoothed.ssa import source_to_ssa


def arrangement(x, out, BLOCK_SIZE=block_size()):
    return x[0:BLOCK_SIZE], out[0:BLOCK_SIZE]


def offset_arrangement(x, out, BLOCK_SIZE=block_size()):
    return x[1 : 1 + BLOCK_SIZE], out[1 : 1 + BLOCK_SIZE]


def add_application(x, out):
    out = x + x


def loop_application(x, out):
    acc = x
    for _ in range(2):
        acc = acc + x
    out = acc


def unsupported_application(x, out):
    tmp = {"value": x}
    out = tmp["value"]


def if_application(x, out):
    acc = x
    if x > x:
        acc = acc + x
    out = acc


def fused_expression_application(x, y, z, out):
    cond = (x > y) & (y < z)
    tmp = ntl.where(cond, ntl.exp(x) + ntl.sqrt(ntl.abs(y)), z)
    out = tmp * 2.0 - x


def reduction_arrangement(x, y, out, BLOCK_SIZE=block_size()):
    return x[0:BLOCK_SIZE], y[0:BLOCK_SIZE], out[0:1]


def binary_arrangement(x, y, out, BLOCK_SIZE=block_size()):
    return x[0:BLOCK_SIZE], y[0:BLOCK_SIZE], out[0:BLOCK_SIZE]


def ternary_arrangement(x, y, z, out, BLOCK_SIZE=block_size()):
    return x[0:BLOCK_SIZE], y[0:BLOCK_SIZE], z[0:BLOCK_SIZE], out[0:BLOCK_SIZE]


def dot_reduction_application(x, y, out):
    out = (x * y).sum()


def fused_affine_helper(x, y, scale=2.0):
    product = x * y
    return product * scale + 1.0


def helper_call_application(x, y, out):
    out = fused_affine_helper(x, y, scale=3.0)


def _ssa_kernel(
    source: str, kernel_name: str, tensors: tuple[TensorTypeIR, ...]
) -> KernelIR:
    ssa = source_to_ssa(source, tensors, kind=kernel_name)
    assert ssa is not None
    return KernelIR(
        kernel_name=kernel_name,
        source=source,
        source_language="ninetoothed-python",
        entrypoint=kernel_name,
        tensors=tensors,
        ssa=ssa,
    )


class SSAFirstBackendLoweringTest(unittest.TestCase):
    def test_backend_entrypoints_do_not_contain_kernel_specialized_lowering(self):
        backend_dir = (
            Path(__file__).resolve().parents[1] / "src" / "ninetoothed" / "backends"
        )
        forbidden = (
            "lower_matmul",
            "lower_reduction",
            "lower_flash",
            "build_ssa_linear_plan",
            "build_ssa_reduction_plan",
            "build_scf_elementwise_plan",
            "FlashAttentionOpIR",
            "MatmulOpIR",
            "ReductionOpIR",
            "detect_online",
            "_single_op",
            "source_passthrough",
            "existing_triton",
        )
        for filename in ("triton.py", "cuda.py", "tilelang.py", "tvm.py"):
            with self.subTest(filename=filename):
                source = (backend_dir / filename).read_text(encoding="utf-8")
                self.assertIn("lower_unified_ssa_artifact", source)
                for token in forbidden:
                    self.assertNotIn(token, source)

    def test_public_lower_generates_fused_expression_without_kernel_template(self):
        expected = {
            "triton": ("ssa-unified-triton-emitter", ("tl.where", "tl.exp", "tl.sqrt")),
            "cuda": ("ssa-unified-cuda-emitter", ("?", "expf", "sqrtf")),
            "tilelang": (
                "ssa-unified-tilelang-emitter",
                ("T.if_then_else", "T.exp", "T.sqrt"),
            ),
            "tvm": ("ssa-unified-tvm-emitter", ("T.if_then_else", "T.exp", "T.sqrt")),
        }
        for backend, (route, fragments) in expected.items():
            with self.subTest(backend=backend):
                artifact = ninetoothed.lower(
                    ternary_arrangement,
                    fused_expression_application,
                    (Tensor(1), Tensor(1), Tensor(1), Tensor(1)),
                    backend=backend,
                    kernel_name=f"ssa_fused_expr_{backend}",
                )

                self.assertTrue(artifact.executable)
                self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
                self.assertEqual(artifact.metadata["source_route"], route)
                self.assertFalse(artifact.metadata["program_ir_compat"])
                self.assertIn("select.where", str(artifact.metadata["ssa"]))
                for fragment in fragments:
                    self.assertIn(fragment, artifact.primary_source)

    def test_public_lower_does_not_use_program_ir_by_default_for_linear_backends(self):
        expected = {
            "triton": ("ssa-unified-triton-emitter", "@triton.jit"),
            "cuda": ("ssa-unified-cuda-emitter", "out[index] = v0;"),
            "tilelang": (
                "ssa-unified-tilelang-emitter",
                "v0 = (x_buf[index] + x_buf[index])",
            ),
            "tvm": ("ssa-unified-tvm-emitter", "v0 = (x_buf[index] + x_buf[index])"),
        }
        for backend, (route, source_fragment) in expected.items():
            with self.subTest(backend=backend):
                artifact = ninetoothed.lower(
                    arrangement,
                    add_application,
                    (Tensor(1), Tensor(1)),
                    backend=backend,
                    kernel_name=f"ssa_first_add_{backend}",
                )

                self.assertTrue(artifact.executable)
                self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
                self.assertNotIn("program_kind", artifact.metadata)
                self.assertEqual(artifact.metadata["source_route"], route)
                self.assertFalse(artifact.metadata["program_ir_compat"])
                self.assertIn("ssa", artifact.metadata)
                self.assertIn(source_fragment, artifact.primary_source)

    def test_public_lower_uses_arrangement_view_shapes_for_ssa_backends(self):
        for backend in ("triton", "cuda", "tilelang", "tvm"):
            with self.subTest(backend=backend):
                artifact = ninetoothed.lower(
                    arrangement,
                    add_application,
                    (Tensor(1), Tensor(1)),
                    backend=backend,
                    kernel_name=f"ssa_arrangement_view_shape_{backend}",
                )

                self.assertEqual(
                    artifact.metadata["kernel_metadata"]["ssa_tensor_ir_source"],
                    "arrangement_views",
                )
                self.assertNotIn("dim0", artifact.primary_source)
                self.assertIn("BLOCK_SIZE", artifact.primary_source)
                self.assertTrue(
                    any(
                        "BLOCK_SIZE" in dim
                        for tensor in artifact.metadata["tensors"]
                        for dim in tensor["shape"]
                    )
                )

    def test_public_lower_preserves_arrangement_view_offsets_for_ssa_backends(self):
        expected = {
            "triton": "x + (",
            "cuda": "x[",
            "tilelang": "x_buf[",
            "tvm": "x_buf[",
        }
        for backend, load_fragment in expected.items():
            with self.subTest(backend=backend):
                artifact = ninetoothed.lower(
                    offset_arrangement,
                    add_application,
                    (Tensor(1), Tensor(1)),
                    backend=backend,
                    kernel_name=f"ssa_arrangement_offset_{backend}",
                )

                source = artifact.primary_source
                self.assertIn(load_fragment, source)
                source_offset = "+ T.int64(1)" if backend == "tvm" else "+ 1"
                self.assertIn(source_offset, source)
                self.assertTrue(
                    any(
                        "+ 1" in tensor["attrs"].get("view_linear_offset", "")
                        for tensor in artifact.metadata["tensors"]
                    )
                )

    def test_cuda_lower_materializes_long_view_index_expressions(self):
        BLOCK_SIZE_M = Symbol("BLOCK_SIZE_M", meta=True)
        BLOCK_SIZE_N = Symbol("BLOCK_SIZE_N", meta=True)

        def tiled_arrangement(
            x, out, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N
        ):
            return x.tile((BLOCK_SIZE_M, BLOCK_SIZE_N)), out.tile(
                (BLOCK_SIZE_M, BLOCK_SIZE_N)
            )

        def tiled_application(x, out):
            out = x + x

        artifact = ninetoothed.lower(
            tiled_arrangement,
            tiled_application,
            (Tensor(2), Tensor(2)),
            backend="cuda",
            kernel_name="ssa_cuda_index_cse_tiled_add",
        )
        source = artifact.primary_source

        self.assertIn("int64_t nt_idx_", source)
        self.assertIn("bool nt_pred_", source)
        self.assertIn("x[(nt_idx_", source)
        self.assertIn("out[(nt_idx_", source)
        self.assertLess(source.count("floor("), 4)

    def test_public_triton_lower_uses_unified_backend_for_structured_shapes(self):
        BLOCK_SIZE_M = Symbol("BLOCK_SIZE_M", meta=True)
        BLOCK_SIZE_N = Symbol("BLOCK_SIZE_N", meta=True)
        BLOCK_SIZE_K = Symbol("BLOCK_SIZE_K", meta=True)

        def matmul_arrangement(
            lhs,
            rhs,
            output,
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        ):
            output_tiled = output.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))
            lhs_tiled = (
                lhs.tile((BLOCK_SIZE_M, BLOCK_SIZE_K))
                .tile((1, -1))
                .expand((-1, output_tiled.shape[1]))
            )
            lhs_tiled.dtype = lhs_tiled.dtype.squeeze(0)
            rhs_tiled = (
                rhs.tile((BLOCK_SIZE_K, BLOCK_SIZE_N))
                .tile((-1, 1))
                .expand((output_tiled.shape[0], -1))
            )
            rhs_tiled.dtype = rhs_tiled.dtype.squeeze(1)
            return lhs_tiled, rhs_tiled, output_tiled

        def matmul_application(lhs, rhs, output):
            accumulator = ntl.zeros(output.shape, dtype=ntl.float32)
            for k in range(lhs.shape[0]):
                accumulator += ntl.dot(lhs[k], rhs[k])
            output = accumulator.to(ntl.float16)

        artifact = ninetoothed.lower(
            matmul_arrangement,
            matmul_application,
            (Tensor(2), Tensor(2), Tensor(2)),
            backend="triton",
            kernel_name="public_triton_codegen_matmul_process",
        )

        self.assertTrue(artifact.executable)
        self.assertEqual(
            artifact.metadata["source_route"], "ssa-unified-triton-emitter"
        )
        self.assertNotIn("linalg.matmul", str(artifact.metadata.get("ssa", "")))
        self.assertIn("@triton.jit", artifact.primary_source)
        self.assertIn("SSAProgramIR", artifact.primary_source)

    def test_public_triton_lower_raises_when_ssa_is_unavailable(self):
        with self.assertRaisesRegex(
            Exception, "Cannot lower `unsupported_application`"
        ):
            ninetoothed.lower(
                arrangement,
                unsupported_application,
                (Tensor(1), Tensor(1)),
                backend="triton",
                kernel_name="ssa_unavailable_no_generation_fallback",
            )

    def test_public_lower_does_not_use_legacy_program_ir_flag_for_artifact_backend_path(
        self,
    ):
        with self.assertRaisesRegex(
            Exception, "Cannot lower `unsupported_application`"
        ):
            ninetoothed.lower(
                arrangement,
                unsupported_application,
                (Tensor(1), Tensor(1)),
                backend="cuda",
                kernel_name="ssa_unavailable_no_program_ir_fallback",
                legacy_program_ir=True,
            )

    def test_source_to_ssa_generates_extended_linear_tensor_ops_for_native_backends(
        self,
    ):
        cases = {
            "where": (
                """
def where_application(x, y, out):
    out = where(x > y, x, y)
""",
                (
                    TensorTypeIR("x", 1, dtype="float32", shape=("n",)),
                    TensorTypeIR("y", 1, dtype="float32", shape=("n",)),
                    TensorTypeIR("out", 1, dtype="float32", shape=("n",)),
                ),
                {
                    "triton": "tl.where",
                    "cuda": " ? ",
                    "tilelang": "T.if_then_else",
                    "tvm": "T.if_then_else",
                },
            ),
            "full": (
                """
def full_application(out):
    out = full((n,), 2.5)
""",
                (TensorTypeIR("out", 1, dtype="float32", shape=("n",)),),
                {
                    "triton": "v1 = v0",
                    "cuda": "float v1 = v0;",
                    "tilelang": "v1 = v0",
                    "tvm": "v1 = v0",
                },
            ),
            "zeros": (
                """
def zeros_application(out):
    out = zeros((n,))
""",
                (TensorTypeIR("out", 1, dtype="float32", shape=("n",)),),
                {
                    "triton": "v0 = 0.0",
                    "cuda": "float v0 = 0.0;",
                    "tilelang": "v0 = 0.0",
                    "tvm": "v0 = 0.0",
                },
            ),
            "view": (
                """
def view_application(x, out):
    out = x[:, :]
""",
                (
                    TensorTypeIR("x", 2, dtype="float32", shape=("rows", "cols")),
                    TensorTypeIR("out", 2, dtype="float32", shape=("rows", "cols")),
                ),
                {
                    "triton": "tl.load(x + ((index // (cols)))",
                    "cuda": "x[((index / (cols)))",
                    "tilelang": "x_buf[((index // (cols)))",
                    "tvm": "x_buf[((index // (cols)))",
                },
            ),
            "extract": (
                """
def extract_application(x, out):
    out = x[0]
""",
                (
                    TensorTypeIR("x", 1, dtype="float32", shape=("n",)),
                    TensorTypeIR("out", 1, dtype="float32", shape=("n",)),
                ),
                {
                    "triton": "tl.load(x + v0",
                    "cuda": "x[v0]",
                    "tilelang": "x_buf[v0]",
                    "tvm": "x_buf[v0]",
                },
            ),
            "tanh": (
                """
def tanh_application(x, out):
    out = tanh(x)
""",
                (
                    TensorTypeIR("x", 1, dtype="float32", shape=("n",)),
                    TensorTypeIR("out", 1, dtype="float32", shape=("n",)),
                ),
                {
                    "triton": "tl.tanh",
                    "cuda": "tanhf",
                    "tilelang": "T.tanh",
                    "tvm": "T.tanh",
                },
            ),
        }
        routes = {
            "triton": "ssa-unified-triton-emitter",
            "cuda": "ssa-unified-cuda-emitter",
            "tilelang": "ssa-unified-tilelang-emitter",
            "tvm": "ssa-unified-tvm-emitter",
        }
        for case_name, (source, tensors, fragments) in cases.items():
            kernel = _ssa_kernel(source, f"ssa_linear_{case_name}", tensors)
            for backend, route in routes.items():
                with self.subTest(case=case_name, backend=backend):
                    artifact = lower_kernel_ir(kernel, backend)

                    self.assertTrue(artifact.executable)
                    self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
                    self.assertEqual(artifact.metadata["source_route"], route)
                    self.assertFalse(artifact.metadata["program_ir_compat"])
                    self.assertIn(fragments[backend], artifact.primary_source)
                    self.assertNotEqual(
                        artifact.metadata["source_route"], "generic-ssa-emitter"
                    )
                    self.assertNotEqual(
                        artifact.metadata["source_route"], "existing-triton-generator"
                    )

    def test_source_to_ssa_generates_shape_dim_for_native_backends(self):
        kernel = _ssa_kernel(
            """
def shape_dim_application(x, out):
    out = x.shape[0]
""",
            "ssa_shape_dim",
            (
                TensorTypeIR("x", 2, dtype="float32", shape=("rows", "cols")),
                TensorTypeIR("out", 1, dtype="float32", shape=("n",)),
            ),
        )
        operations = kernel.ssa.blocks[0].operations
        self.assertEqual(
            [operation.opcode for operation in operations], ["shape.dim", "mem.store"]
        )
        self.assertEqual(operations[0].attrs["dim"], 0)
        expected = {
            "triton": (
                "ssa-unified-triton-emitter",
                ("rows: tl.constexpr", "v0 = rows"),
            ),
            "cuda": (
                "ssa-unified-cuda-emitter",
                ("int64_t rows", "int64_t v0 = rows;"),
            ),
            "tilelang": (
                "ssa-unified-tilelang-emitter",
                ("rows: T.int64", "v0 = rows"),
            ),
            "tvm": ("ssa-unified-tvm-emitter", ("rows: T.int64", "v0 = rows")),
        }
        for backend, (route, source_fragments) in expected.items():
            with self.subTest(backend=backend):
                artifact = lower_kernel_ir(kernel, backend)

                self.assertTrue(artifact.executable)
                self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
                self.assertEqual(artifact.metadata["source_route"], route)
                self.assertFalse(artifact.metadata["program_ir_compat"])
                self.assertNotEqual(
                    artifact.metadata["source_route"], "generic-ssa-emitter"
                )
                for source_fragment in source_fragments:
                    self.assertIn(source_fragment, artifact.primary_source)

    def test_source_to_ssa_generates_tensor_stride_for_native_backends(self):
        kernel = _ssa_kernel(
            """
def stride_application(x, out):
    out = x.stride(0) + x.stride(1)
""",
            "ssa_tensor_stride",
            (
                TensorTypeIR("x", 2, dtype="float32", shape=("rows", "cols")),
                TensorTypeIR("out", 1, dtype="int64", shape=("n",)),
            ),
        )
        operations = kernel.ssa.blocks[0].operations
        self.assertEqual(
            [operation.opcode for operation in operations],
            ["tensor.stride", "tensor.stride", "arith.add", "mem.store"],
        )
        self.assertEqual(operations[0].attrs["dim"], 0)
        self.assertEqual(operations[1].attrs["dim"], 1)
        expected = {
            "triton": (
                "ssa-unified-triton-emitter",
                ("v0 = cols", "v1 = 1", "v2 = (v0 + v1)"),
            ),
            "cuda": (
                "ssa-unified-cuda-emitter",
                ("int64_t v0 = cols;", "int64_t v1 = 1;", "int64_t v2 = (v0 + v1);"),
            ),
            "tilelang": (
                "ssa-unified-tilelang-emitter",
                ("v0 = cols", "v1 = 1", "v2 = (v0 + v1)"),
            ),
            "tvm": (
                "ssa-unified-tvm-emitter",
                ("v0 = cols", "v1 = 1", "v2 = (v0 + v1)"),
            ),
        }
        for backend, (route, source_fragments) in expected.items():
            with self.subTest(backend=backend):
                artifact = lower_kernel_ir(kernel, backend)

                self.assertTrue(artifact.executable)
                self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
                self.assertEqual(artifact.metadata["source_route"], route)
                self.assertFalse(artifact.metadata["program_ir_compat"])
                self.assertNotIn("lower_stride", artifact.primary_source)
                for source_fragment in source_fragments:
                    self.assertIn(source_fragment, artifact.primary_source)

    def test_source_to_ssa_generates_maximum_minimum_for_native_backends(self):
        kernel = _ssa_kernel(
            """
def max_min_application(x, y, out):
    tmp = maximum(x, y)
    out = minimum(tmp, y)
""",
            "ssa_max_min",
            (
                TensorTypeIR("x", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("y", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("out", 1, dtype="float32", shape=("n",)),
            ),
        )
        self.assertEqual(
            [operation.opcode for operation in kernel.ssa.blocks[0].operations],
            ["arith.maximum", "arith.minimum", "mem.store"],
        )
        expected = {
            "triton": ("ssa-unified-triton-emitter", ("tl.maximum", "tl.minimum")),
            "cuda": ("ssa-unified-cuda-emitter", ("fmaxf", "fminf")),
            "tilelang": ("ssa-unified-tilelang-emitter", ("T.max", "T.min")),
            "tvm": ("ssa-unified-tvm-emitter", ("T.max", "T.min")),
        }
        for backend, (route, source_fragments) in expected.items():
            with self.subTest(backend=backend):
                artifact = lower_kernel_ir(kernel, backend)

                self.assertTrue(artifact.executable)
                self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
                self.assertEqual(artifact.metadata["source_route"], route)
                self.assertFalse(artifact.metadata["program_ir_compat"])
                self.assertNotEqual(
                    artifact.metadata["source_route"], "generic-ssa-emitter"
                )
                for source_fragment in source_fragments:
                    self.assertIn(source_fragment, artifact.primary_source)

    def test_source_to_ssa_generates_common_math_calls_for_native_backends(self):
        kernel = _ssa_kernel(
            """
def common_math_application(x, y, out):
    out = log1p(abs(x)) + atan2(x, y) + pow(abs(y) + 0.25, 0.5)
""",
            "ssa_common_math",
            (
                TensorTypeIR("x", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("y", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("out", 1, dtype="float32", shape=("n",)),
            ),
        )
        opcodes = [operation.opcode for operation in kernel.ssa.blocks[0].operations]
        self.assertIn("math.log1p", opcodes)
        self.assertIn("math.atan2", opcodes)
        self.assertIn("math.pow", opcodes)
        expected = {
            "triton": (
                "ssa-unified-triton-emitter",
                ("tl.log(1.0 +", "tl.atan2", "tl.pow"),
            ),
            "cuda": ("ssa-unified-cuda-emitter", ("log1pf", "atan2f", "powf")),
            "tilelang": (
                "ssa-unified-tilelang-emitter",
                ("T.log1p", "T.atan2", "T.pow"),
            ),
            "tvm": ("ssa-unified-tvm-emitter", ("T.log1p", "T.atan2", "T.pow")),
        }
        for backend, (route, source_fragments) in expected.items():
            with self.subTest(backend=backend):
                artifact = lower_kernel_ir(kernel, backend)

                self.assertTrue(artifact.executable)
                self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
                self.assertEqual(artifact.metadata["source_route"], route)
                self.assertFalse(artifact.metadata["program_ir_compat"])
                self.assertNotEqual(
                    artifact.metadata["source_route"], "generic-ssa-emitter"
                )
                for source_fragment in source_fragments:
                    self.assertIn(source_fragment, artifact.primary_source)

    def test_source_to_ssa_generates_python_expression_syntax_for_native_backends(self):
        kernel = _ssa_kernel(
            """
def python_expression_syntax_application(x, y, z, out):
    tmp: float = x if 0 < 1 < 2 else y
    pass
    out = tmp + z
    return out
""",
            "ssa_python_expression_syntax",
            (
                TensorTypeIR("x", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("y", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("z", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("out", 1, dtype="float32", shape=("n",)),
            ),
        )
        opcodes = [operation.opcode for operation in kernel.ssa.blocks[0].operations]
        self.assertEqual(
            opcodes,
            [
                "arith.constant",
                "arith.constant",
                "cmp.lt",
                "arith.constant",
                "cmp.lt",
                "arith.and",
                "select.where",
                "arith.add",
                "mem.store",
            ],
        )
        expected = {
            "triton": (
                "ssa-unified-triton-emitter",
                ("tl.where", "(v2 & v4)", "tl.store(out + index"),
            ),
            "cuda": (
                "ssa-unified-cuda-emitter",
                ("bool v5 = (v2 & v4);", " ? ", "out[index] = v7;"),
            ),
            "tilelang": (
                "ssa-unified-tilelang-emitter",
                ("T.if_then_else", "(v2 & v4)", "out_buf[index] = v7"),
            ),
            "tvm": (
                "ssa-unified-tvm-emitter",
                ("T.if_then_else", "(v2 & v4)", "out_buf[index] = v7"),
            ),
        }
        for backend, (route, source_fragments) in expected.items():
            with self.subTest(backend=backend):
                artifact = lower_kernel_ir(kernel, backend)

                self.assertTrue(artifact.executable)
                self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
                self.assertEqual(artifact.metadata["source_route"], route)
                self.assertFalse(artifact.metadata["program_ir_compat"])
                for source_fragment in source_fragments:
                    self.assertIn(source_fragment, artifact.primary_source)

    def test_source_to_ssa_generates_method_math_and_dim_alias_for_native_backends(
        self,
    ):
        kernel = _ssa_kernel(
            """
def method_math_application(x, out):
    denom = x.sqrt().sum(dim=0)
    out = x.exp() / denom
""",
            "ssa_method_math",
            (
                TensorTypeIR("x", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("out", 1, dtype="float32", shape=("n",)),
            ),
        )
        opcodes = [operation.opcode for operation in kernel.ssa.blocks[0].operations]
        self.assertEqual(
            opcodes, ["math.sqrt", "reduce.sum", "math.exp", "arith.div", "mem.store"]
        )
        self.assertEqual(kernel.ssa.blocks[0].operations[1].attrs["axis"], 0)
        expected = {
            "triton": (
                "ssa-unified-triton-emitter",
                ("tl.sqrt", "for v1_i in range(0, n, 1):", "tl.exp"),
            ),
            "cuda": (
                "ssa-unified-cuda-emitter",
                ("sqrtf", "for (int64_t v1_i = 0; v1_i < n; v1_i += 1)", "expf"),
            ),
            "tilelang": (
                "ssa-unified-tilelang-emitter",
                ("T.sqrt", "for v1_i in T.serial(n):", "T.exp"),
            ),
            "tvm": (
                "ssa-unified-tvm-emitter",
                ("T.sqrt", "for v1_i in T.serial(n):", "T.exp"),
            ),
        }
        for backend, (route, source_fragments) in expected.items():
            with self.subTest(backend=backend):
                artifact = lower_kernel_ir(kernel, backend)

                self.assertTrue(artifact.executable)
                self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
                self.assertEqual(artifact.metadata["source_route"], route)
                self.assertFalse(artifact.metadata["program_ir_compat"])
                for source_fragment in source_fragments:
                    self.assertIn(source_fragment, artifact.primary_source)

    def test_source_to_ssa_generates_namespace_math_calls_for_native_backends(self):
        kernel = _ssa_kernel(
            """
def namespace_math_application(x, out):
    out = math.exp(x) + tl.sqrt(x)
""",
            "ssa_namespace_math",
            (
                TensorTypeIR("x", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("out", 1, dtype="float32", shape=("n",)),
            ),
        )
        opcodes = [operation.opcode for operation in kernel.ssa.blocks[0].operations]
        self.assertEqual(opcodes, ["math.exp", "math.sqrt", "arith.add", "mem.store"])
        expected = {
            "triton": ("ssa-unified-triton-emitter", ("tl.exp", "tl.sqrt")),
            "cuda": ("ssa-unified-cuda-emitter", ("expf", "sqrtf")),
            "tilelang": ("ssa-unified-tilelang-emitter", ("T.exp", "T.sqrt")),
            "tvm": ("ssa-unified-tvm-emitter", ("T.exp", "T.sqrt")),
        }
        for backend, (route, source_fragments) in expected.items():
            with self.subTest(backend=backend):
                artifact = lower_kernel_ir(kernel, backend)

                self.assertTrue(artifact.executable)
                self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
                self.assertEqual(artifact.metadata["source_route"], route)
                self.assertFalse(artifact.metadata["program_ir_compat"])
                self.assertNotIn("math[", artifact.primary_source)
                for source_fragment in source_fragments:
                    self.assertIn(source_fragment, artifact.primary_source)

    def test_source_to_ssa_generates_extended_math_calls_for_native_backends(self):
        kernel = _ssa_kernel(
            """
def extended_math_application(x, y, out):
    out = acos(x) + asin(y) + atan(x) + log10(abs(y) + 1.0) + expm1(x) + sinh(x) + cosh(y)
""",
            "ssa_extended_math",
            (
                TensorTypeIR("x", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("y", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("out", 1, dtype="float32", shape=("n",)),
            ),
        )
        opcodes = [operation.opcode for operation in kernel.ssa.blocks[0].operations]
        for opcode in (
            "math.acos",
            "math.asin",
            "math.atan",
            "math.log10",
            "math.expm1",
            "math.sinh",
            "math.cosh",
        ):
            self.assertIn(opcode, opcodes)
        expected = {
            "triton": (
                "ssa-unified-triton-emitter",
                ("tl.acos", "tl.asin", "tl.atan", "2.302585092994046", "tl.exp"),
            ),
            "cuda": (
                "ssa-unified-cuda-emitter",
                ("acosf", "asinf", "atanf", "log10f", "expm1f", "sinhf", "coshf"),
            ),
            "tilelang": (
                "ssa-unified-tilelang-emitter",
                ("T.acos", "T.asin", "T.atan", "T.log10", "T.exp", "T.sinh", "T.cosh"),
            ),
            "tvm": (
                "ssa-unified-tvm-emitter",
                ("T.acos", "T.asin", "T.atan", "T.log10", "T.exp", "T.sinh", "T.cosh"),
            ),
        }
        for backend, (route, source_fragments) in expected.items():
            with self.subTest(backend=backend):
                artifact = lower_kernel_ir(kernel, backend)

                self.assertTrue(artifact.executable)
                self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
                self.assertEqual(artifact.metadata["source_route"], route)
                self.assertFalse(artifact.metadata["program_ir_compat"])
                self.assertNotEqual(
                    artifact.metadata["source_route"], "generic-ssa-emitter"
                )
                for source_fragment in source_fragments:
                    self.assertIn(source_fragment, artifact.primary_source)

    def test_source_to_ssa_generates_bitwise_shifts_for_native_backends(self):
        kernel = _ssa_kernel(
            """
def bitwise_shift_application(x, y, out):
    out = (x << 1) ^ (y >> 1)
""",
            "ssa_bitwise_shift",
            (
                TensorTypeIR("x", 1, dtype="int64", shape=("n",)),
                TensorTypeIR("y", 1, dtype="int64", shape=("n",)),
                TensorTypeIR("out", 1, dtype="int64", shape=("n",)),
            ),
        )
        opcodes = [operation.opcode for operation in kernel.ssa.blocks[0].operations]
        self.assertIn("arith.bitwise_left_shift", opcodes)
        self.assertIn("arith.bitwise_right_shift", opcodes)
        self.assertIn("arith.bitwise_xor", opcodes)
        expected = {
            "triton": "ssa-unified-triton-emitter",
            "cuda": "ssa-unified-cuda-emitter",
            "tilelang": "ssa-unified-tilelang-emitter",
            "tvm": "ssa-unified-tvm-emitter",
        }
        for backend, route in expected.items():
            with self.subTest(backend=backend):
                artifact = lower_kernel_ir(kernel, backend)

                self.assertTrue(artifact.executable)
                self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
                self.assertEqual(artifact.metadata["source_route"], route)
                self.assertFalse(artifact.metadata["program_ir_compat"])
                self.assertNotEqual(
                    artifact.metadata["source_route"], "generic-ssa-emitter"
                )
                self.assertIn("<<", artifact.primary_source)
                self.assertIn(">>", artifact.primary_source)
                self.assertIn("^", artifact.primary_source)

    def test_source_to_ssa_preserves_subscript_store_indices_for_native_backends(self):
        cases = {
            "one_dimensional": (
                """
def indexed_store_application(x, out):
    i = x.offsets(0)
    out[i] = x
""",
                (
                    TensorTypeIR("x", 1, dtype="float32", shape=("n",)),
                    TensorTypeIR("out", 1, dtype="float32", shape=("n",)),
                ),
                {
                    "triton": "tl.store(out + v0",
                    "cuda": "out[v0] = x[index];",
                    "tilelang": "out_buf[v0] = x_buf[index]",
                    "tvm": "out_buf[v0] = x_buf[index]",
                },
            ),
            "two_dimensional": (
                """
def indexed_store_2d_application(x, out):
    i = x.offsets(0)
    j = x.offsets(1)
    out[i, j] = x
""",
                (
                    TensorTypeIR("x", 2, dtype="float32", shape=("rows", "cols")),
                    TensorTypeIR("out", 2, dtype="float32", shape=("rows", "cols")),
                ),
                {
                    "triton": "tl.store(out + (v0) * (cols) + (v1)",
                    "cuda": "out[(v0) * (cols) + (v1)] = x[((index / (cols)))",
                    "tilelang": "out_buf[(v0) * (cols) + (v1)] = x_buf[((index // (cols)))",
                    "tvm": "out_buf[(v0) * (cols) + (v1)] = x_buf[((index // (cols)))",
                },
            ),
        }
        routes = {
            "triton": "ssa-unified-triton-emitter",
            "cuda": "ssa-unified-cuda-emitter",
            "tilelang": "ssa-unified-tilelang-emitter",
            "tvm": "ssa-unified-tvm-emitter",
        }
        for case_name, (source, tensors, fragments) in cases.items():
            kernel = _ssa_kernel(source, f"ssa_indexed_store_{case_name}", tensors)
            self.assertIn("'indices'", str(kernel.ssa))
            for backend, route in routes.items():
                with self.subTest(case=case_name, backend=backend):
                    artifact = lower_kernel_ir(kernel, backend)

                    self.assertTrue(artifact.executable)
                    self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
                    self.assertEqual(artifact.metadata["source_route"], route)
                    self.assertFalse(artifact.metadata["program_ir_compat"])
                    self.assertIn(fragments[backend], artifact.primary_source)

    def test_source_to_ssa_expands_subscript_augassign_for_native_backends(self):
        kernel = _ssa_kernel(
            """
def indexed_augassign_application(x, out):
    i = x.offsets(0)
    out[i] += x
""",
            "ssa_indexed_augassign",
            (
                TensorTypeIR("x", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("out", 1, dtype="float32", shape=("n",)),
            ),
        )
        operations = kernel.ssa.blocks[0].operations
        self.assertEqual(
            [operation.opcode for operation in operations],
            [
                "index.offset",
                "tensor.extract",
                "arith.add",
                "mem.store",
            ],
        )
        self.assertEqual(operations[-1].attrs["indices"], ("%0",))
        expected = {
            "triton": ("ssa-unified-triton-emitter", "tl.store(out + v0"),
            "cuda": ("ssa-unified-cuda-emitter", "out[v0] = v2;"),
            "tilelang": ("ssa-unified-tilelang-emitter", "out_buf[v0] = v2"),
            "tvm": ("ssa-unified-tvm-emitter", "out_buf[v0] = v2"),
        }
        for backend, (route, source_fragment) in expected.items():
            with self.subTest(backend=backend):
                artifact = lower_kernel_ir(kernel, backend)

                self.assertTrue(artifact.executable)
                self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
                self.assertEqual(artifact.metadata["source_route"], route)
                self.assertFalse(artifact.metadata["program_ir_compat"])
                self.assertIn(source_fragment, artifact.primary_source)

    def test_source_to_ssa_linearizes_multidimensional_extract_by_source_shape(self):
        kernel = _ssa_kernel(
            """
def extract_2d_application(x, out):
    i = x.offsets(0)
    j = x.offsets(1)
    out = x[i, j]
""",
            "ssa_extract_2d",
            (
                TensorTypeIR("x", 2, dtype="float32", shape=("rows", "cols")),
                TensorTypeIR("out", 2, dtype="float32", shape=("rows", "cols")),
            ),
        )
        expected = {
            "triton": (
                "ssa-unified-triton-emitter",
                "tl.load(x + (v0) * (cols) + (v1)",
            ),
            "cuda": ("ssa-unified-cuda-emitter", "x[(v0) * (cols) + (v1)]"),
            "tilelang": ("ssa-unified-tilelang-emitter", "x_buf[(v0) * (cols) + (v1)]"),
            "tvm": ("ssa-unified-tvm-emitter", "x_buf[(v0) * (cols) + (v1)]"),
        }
        extract = kernel.ssa.blocks[0].operations[2]
        self.assertEqual(extract.opcode, "tensor.extract")
        self.assertEqual(extract.operands, ("x", "%0", "%1"))
        for backend, (route, source_fragment) in expected.items():
            with self.subTest(backend=backend):
                artifact = lower_kernel_ir(kernel, backend)

                self.assertTrue(artifact.executable)
                self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
                self.assertEqual(artifact.metadata["source_route"], route)
                self.assertFalse(artifact.metadata["program_ir_compat"])
                self.assertIn(source_fragment, artifact.primary_source)
                self.assertNotIn("x + v0 + v1", artifact.primary_source)
                self.assertNotIn("x[v0 + v1]", artifact.primary_source)
                self.assertNotIn("x_buf[v0 + v1]", artifact.primary_source)

    def test_public_lower_generates_ssa_reduction_for_native_backends(self):
        expected = {
            "triton": ("ssa-unified-triton-emitter", "@triton.jit"),
            "cuda": ("ssa-unified-cuda-emitter", "(x[v1_i] * y[v1_i])"),
            "tilelang": ("ssa-unified-tilelang-emitter", "x_buf[v1_i] * y_buf[v1_i]"),
            "tvm": ("ssa-unified-tvm-emitter", "x_buf[v1_i] * y_buf[v1_i]"),
        }
        for backend, (route, source_fragment) in expected.items():
            with self.subTest(backend=backend):
                artifact = ninetoothed.lower(
                    reduction_arrangement,
                    dot_reduction_application,
                    (Tensor(1), Tensor(1), Tensor(1)),
                    backend=backend,
                    kernel_name=f"ssa_dot_reduce_{backend}",
                )

                self.assertTrue(artifact.executable)
                self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
                self.assertEqual(artifact.metadata["source_route"], route)
                self.assertFalse(artifact.metadata["program_ir_compat"])
                self.assertIn(source_fragment, artifact.primary_source)
                self.assertIn("reduce.sum", str(artifact.metadata["ssa"]))

    def test_public_lower_tvm_reduction_artifact_builds(self):
        kernel_name = "ssa_dot_reduce_tvm_build_regression"
        artifact = ninetoothed.lower(
            reduction_arrangement,
            dot_reduction_application,
            (Tensor(1), Tensor(1), Tensor(1)),
            backend="tvm",
            kernel_name=kernel_name,
        )

        self.assertIn('T.Cast("int64", block_id)', artifact.primary_source)
        self.assertIn("T.int64(0)", artifact.primary_source)
        self.assertIn("T.int64(1)", artifact.primary_source)
        self.assertRegex(
            artifact.primary_source,
            r"\(index\) < ninetoothed_ninetoothed_tensor_[0-9]+_size_0",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "tvm_reduction_artifact.py"
            path.write_text(artifact.primary_source, encoding="utf-8")
            module_name = "tvm_reduction_artifact"
            spec = importlib.util.spec_from_file_location(module_name, path)
            self.assertIsNotNone(spec)
            assert spec is not None
            self.assertIsNotNone(spec.loader)
            assert spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                try:
                    spec.loader.exec_module(module)
                    getattr(module, f"build_{kernel_name}")()
                except ImportError as exc:
                    self.skipTest(f"TVM is not installed: {exc}")
            finally:
                sys.modules.pop(module_name, None)

    def test_public_lower_inlines_user_helper_calls_before_ssa_lowering(self):
        expected = {
            "triton": (
                "ssa-unified-triton-emitter",
                ("v0 = (tl.load(x + index", "v2 = (v0 * v1)", "v4 = (v2 + v3)"),
            ),
            "cuda": (
                "ssa-unified-cuda-emitter",
                (
                    "float v0 = (x[index] * y[index]);",
                    "float v2 = (v0 * v1);",
                    "float v4 = (v2 + v3);",
                ),
            ),
            "tilelang": (
                "ssa-unified-tilelang-emitter",
                (
                    "v0 = (x_buf[index] * y_buf[index])",
                    "v2 = (v0 * v1)",
                    "v4 = (v2 + v3)",
                ),
            ),
            "tvm": (
                "ssa-unified-tvm-emitter",
                (
                    "v0 = (x_buf[index] * y_buf[index])",
                    "v2 = (v0 * v1)",
                    "v4 = (v2 + v3)",
                ),
            ),
        }
        for backend, (route, source_fragments) in expected.items():
            with self.subTest(backend=backend):
                artifact = ninetoothed.lower(
                    binary_arrangement,
                    helper_call_application,
                    (Tensor(1), Tensor(1), Tensor(1)),
                    backend=backend,
                    kernel_name=f"ssa_helper_inline_{backend}",
                )

                self.assertTrue(artifact.executable)
                self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
                self.assertEqual(artifact.metadata["source_route"], route)
                self.assertFalse(artifact.metadata["program_ir_compat"])
                self.assertNotIn(
                    "call.fused_affine_helper", str(artifact.metadata["ssa"])
                )
                self.assertNotIn("fused_affine_helper", artifact.primary_source)
                for source_fragment in source_fragments:
                    self.assertIn(source_fragment, artifact.primary_source)

    def test_source_to_ssa_generates_axis_reduction_for_native_backends(self):
        kernel = _ssa_kernel(
            """
def axis_addmv_application(bias, a, x, out):
    out = bias + sum(a * x, axis=1)
""",
            "ssa_axis_addmv",
            (
                TensorTypeIR("bias", 1, dtype="float32", shape=("rows",)),
                TensorTypeIR("a", 2, dtype="float32", shape=("rows", "cols")),
                TensorTypeIR("x", 1, dtype="float32", shape=("cols",)),
                TensorTypeIR("out", 1, dtype="float32", shape=("rows",)),
            ),
        )
        expected = {
            "triton": ("ssa-unified-triton-emitter", "for v1_i in range(0, cols, 1):"),
            "cuda": (
                "ssa-unified-cuda-emitter",
                "a[(index) * (cols) + (v1_i)] * x[v1_i]",
            ),
            "tilelang": ("ssa-unified-tilelang-emitter", "for v1_i in T.serial(cols):"),
            "tvm": ("ssa-unified-tvm-emitter", "for v1_i in T.serial(cols):"),
        }
        for backend, (route, source_fragment) in expected.items():
            with self.subTest(backend=backend):
                artifact = lower_kernel_ir(kernel, backend)

                self.assertTrue(artifact.executable)
                self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
                self.assertEqual(artifact.metadata["source_route"], route)
                self.assertFalse(artifact.metadata["program_ir_compat"])
                self.assertIn(source_fragment, artifact.primary_source)
                self.assertIn("reduce.sum", str(artifact.metadata["ssa"]))

    def test_source_to_ssa_generates_rowwise_reduction_for_native_backends(self):
        kernel = _ssa_kernel(
            """
def rowwise_norm_application(x, out):
    out = x / sum(x, axis=1)
""",
            "ssa_rowwise_norm",
            (
                TensorTypeIR("x", 2, dtype="float32", shape=("rows", "cols")),
                TensorTypeIR("out", 2, dtype="float32", shape=("rows", "cols")),
            ),
        )
        expected = {
            "triton": ("ssa-unified-triton-emitter", "for v0_i in range(0, cols, 1):"),
            "cuda": ("ssa-unified-cuda-emitter", "out[index] = v1;"),
            "tilelang": ("ssa-unified-tilelang-emitter", "out_buf[index] = v1"),
            "tvm": ("ssa-unified-tvm-emitter", "out_buf[index] = v1"),
        }
        for backend, (route, source_fragment) in expected.items():
            with self.subTest(backend=backend):
                artifact = lower_kernel_ir(kernel, backend)

                self.assertTrue(artifact.executable)
                self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
                self.assertEqual(artifact.metadata["source_route"], route)
                self.assertFalse(artifact.metadata["program_ir_compat"])
                self.assertIn(source_fragment, artifact.primary_source)
                self.assertIn("reduce.sum", str(artifact.metadata["ssa"]))

    def test_source_to_ssa_generates_linalg_matmul_for_native_backends(self):
        kernel = _ssa_kernel(
            """
def matmul_application(a, b, out):
    out = a @ b
""",
            "ssa_matmul",
            (
                TensorTypeIR("a", 2, dtype="float32", shape=("m", "k")),
                TensorTypeIR("b", 2, dtype="float32", shape=("k", "n")),
                TensorTypeIR("out", 2, dtype="float32", shape=("m", "n")),
            ),
        )
        expected = {
            "triton": ("ssa-unified-triton-emitter", "for v10_i in range(0, k, 1):"),
            "cuda": (
                "ssa-unified-cuda-emitter",
                "for (int64_t v10_i = 0; v10_i < k; v10_i += 1)",
            ),
            "tilelang": ("ssa-unified-tilelang-emitter", "for v10_i in T.serial(k):"),
            "tvm": ("ssa-unified-tvm-emitter", "for v10_i in T.serial(k):"),
        }
        for backend, (route, source_fragment) in expected.items():
            with self.subTest(backend=backend):
                artifact = lower_kernel_ir(kernel, backend)

                self.assertTrue(artifact.executable)
                self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
                self.assertEqual(artifact.metadata["source_route"], route)
                self.assertFalse(artifact.metadata["program_ir_compat"])
                self.assertNotIn("linalg.matmul", str(artifact.metadata["ssa"]))
                self.assertIn(source_fragment, artifact.primary_source)

    def test_source_to_ssa_generates_linalg_transpose_for_native_backends(self):
        kernel = _ssa_kernel(
            """
def transpose_application(x, out):
    out = transpose(x)
""",
            "ssa_transpose",
            (
                TensorTypeIR("x", 2, dtype="float32", shape=("rows", "cols")),
                TensorTypeIR("out", 2, dtype="float32", shape=("cols", "rows")),
            ),
        )
        expected = {
            "triton": (
                "ssa-unified-triton-emitter",
                "tl.load(x + (v2) * (cols) + (v1)",
            ),
            "cuda": ("ssa-unified-cuda-emitter", "x[(v2) * (cols) + (v1)]"),
            "tilelang": ("ssa-unified-tilelang-emitter", "x_buf[(v2) * (cols) + (v1)]"),
            "tvm": ("ssa-unified-tvm-emitter", "x_buf[(v2) * (cols) + (v1)]"),
        }
        for backend, (route, source_fragment) in expected.items():
            with self.subTest(backend=backend):
                artifact = lower_kernel_ir(kernel, backend)

                self.assertTrue(artifact.executable)
                self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
                self.assertEqual(artifact.metadata["source_route"], route)
                self.assertFalse(artifact.metadata["program_ir_compat"])
                self.assertNotIn("linalg.transpose", str(artifact.metadata["ssa"]))
                self.assertIn(source_fragment, artifact.primary_source)

    def test_source_to_ssa_generates_linalg_transpose_for_attribute_t(self):
        kernel = _ssa_kernel(
            """
def transpose_attribute_application(x, out):
    out = x.T
""",
            "ssa_transpose_attribute",
            (
                TensorTypeIR("x", 2, dtype="float32", shape=("rows", "cols")),
                TensorTypeIR("out", 2, dtype="float32", shape=("cols", "rows")),
            ),
        )
        expected = {
            "triton": (
                "ssa-unified-triton-emitter",
                "tl.load(x + (v2) * (cols) + (v1)",
            ),
            "cuda": ("ssa-unified-cuda-emitter", "x[(v2) * (cols) + (v1)]"),
            "tilelang": ("ssa-unified-tilelang-emitter", "x_buf[(v2) * (cols) + (v1)]"),
            "tvm": ("ssa-unified-tvm-emitter", "x_buf[(v2) * (cols) + (v1)]"),
        }
        transpose = kernel.ssa.blocks[0].operations[0]
        self.assertEqual(transpose.opcode, "linalg.transpose")
        self.assertEqual(transpose.operands, ("x",))
        for backend, (route, source_fragment) in expected.items():
            with self.subTest(backend=backend):
                artifact = lower_kernel_ir(kernel, backend)

                self.assertTrue(artifact.executable)
                self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
                self.assertEqual(artifact.metadata["source_route"], route)
                self.assertFalse(artifact.metadata["program_ir_compat"])
                self.assertNotIn("linalg.transpose", str(artifact.metadata["ssa"]))
                self.assertIn(source_fragment, artifact.primary_source)

    def test_source_to_ssa_emits_store_inside_scf_for_without_operator_dispatch(self):
        kernel = _ssa_kernel(
            """
def loop_store_application(x, out):
    for i in range(n):
        out[i] = x[i] + 1.0
""",
            "ssa_loop_store",
            (
                TensorTypeIR("x", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("out", 1, dtype="float32", shape=("n",)),
            ),
        )
        expected = {
            "triton": (
                "ssa-unified-triton-emitter",
                ("for loop_i in range(0, n, 1):", "tl.store(out + loop_i"),
            ),
            "cuda": (
                "ssa-unified-cuda-emitter",
                ("for (int64_t loop_i = 0; loop_i < n; loop_i += 1)", "out[loop_i] ="),
            ),
            "tilelang": (
                "ssa-unified-tilelang-emitter",
                ("for loop_i in T.serial(n):", "out_buf[loop_i] ="),
            ),
            "tvm": (
                "ssa-unified-tvm-emitter",
                ("for loop_i in T.serial(n):", "out_buf[loop_i] ="),
            ),
        }
        for backend, (route, source_fragments) in expected.items():
            with self.subTest(backend=backend):
                artifact = lower_kernel_ir(kernel, backend)

                self.assertTrue(artifact.executable)
                self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
                self.assertEqual(artifact.metadata["source_route"], route)
                self.assertFalse(artifact.metadata["program_ir_compat"])
                self.assertIn("scf.for", str(artifact.metadata["ssa"]))
                self.assertNotIn("lower_loop_store", artifact.primary_source)
                for source_fragment in source_fragments:
                    self.assertIn(source_fragment, artifact.primary_source)

    def test_source_to_ssa_emits_store_inside_scf_if_without_operator_dispatch(self):
        kernel = _ssa_kernel(
            """
def if_store_application(x, out):
    if 1 < 2:
        i = x.offsets(0)
        out[i] = x
""",
            "ssa_if_store",
            (
                TensorTypeIR("x", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("out", 1, dtype="float32", shape=("n",)),
            ),
        )
        expected = {
            "triton": (
                "ssa-unified-triton-emitter",
                ("v2 = (v0 < v1)", "if v2:", "tl.store(out + v3"),
            ),
            "cuda": (
                "ssa-unified-cuda-emitter",
                ("bool v2 = (v0 < v1);", "if (v2) {", "out[v3"),
            ),
            "tilelang": (
                "ssa-unified-tilelang-emitter",
                ("v2 = (v0 < v1)", "if v2:", "out_buf[v3"),
            ),
            "tvm": (
                "ssa-unified-tvm-emitter",
                ("v2 = (v0 < v1)", "if v2:", "out_buf[v3"),
            ),
        }
        for backend, (route, source_fragments) in expected.items():
            with self.subTest(backend=backend):
                artifact = lower_kernel_ir(kernel, backend)

                self.assertTrue(artifact.executable)
                self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
                self.assertEqual(artifact.metadata["source_route"], route)
                self.assertFalse(artifact.metadata["program_ir_compat"])
                self.assertIn("scf.if", str(artifact.metadata["ssa"]))
                self.assertNotIn("lower_if_store", artifact.primary_source)
                for source_fragment in source_fragments:
                    self.assertIn(source_fragment, artifact.primary_source)

    def test_source_to_ssa_preserves_else_store_region_for_side_effect_if(self):
        kernel = _ssa_kernel(
            """
def if_else_store_application(x, y, out):
    if 1 < 2:
        i = x.offsets(0)
        out[i] = x
    else:
        j = y.offsets(0)
        out[j] = y
""",
            "ssa_if_else_store",
            (
                TensorTypeIR("x", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("y", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("out", 1, dtype="float32", shape=("n",)),
            ),
        )
        expected = {
            "triton": (
                "ssa-unified-triton-emitter",
                (
                    "if v2:",
                    "tl.store(out + v3",
                    "else:",
                    "tl.store(out + v4",
                    "tl.load(y + index",
                ),
            ),
            "cuda": (
                "ssa-unified-cuda-emitter",
                ("if (v2) {", "out[v3", "} else {", "out[v4"),
            ),
            "tilelang": (
                "ssa-unified-tilelang-emitter",
                ("if v2:", "out_buf[v3", "else:", "out_buf[v4"),
            ),
            "tvm": (
                "ssa-unified-tvm-emitter",
                ("if v2:", "out_buf[v3", "else:", "out_buf[v4"),
            ),
        }
        op = next(
            operation
            for operation in kernel.ssa.blocks[0].operations
            if operation.opcode == "scf.if"
        )
        self.assertEqual(tuple(region.name for region in op.regions), ("then", "else"))
        for backend, (route, source_fragments) in expected.items():
            with self.subTest(backend=backend):
                artifact = lower_kernel_ir(kernel, backend)

                self.assertTrue(artifact.executable)
                self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
                self.assertEqual(artifact.metadata["source_route"], route)
                self.assertFalse(artifact.metadata["program_ir_compat"])
                for source_fragment in source_fragments:
                    self.assertIn(source_fragment, artifact.primary_source)

    def test_source_to_ssa_emits_multi_result_scf_if_once(self):
        kernel = _ssa_kernel(
            """
def multi_result_if_application(x, y, out0, out1):
    a = x
    b = y
    if 1 < 2:
        a = x + y
        b = x - y
    out0 = a
    out1 = b
""",
            "ssa_multi_result_if",
            (
                TensorTypeIR("x", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("y", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("out0", 1, dtype="float32", shape=("n",)),
                TensorTypeIR("out1", 1, dtype="float32", shape=("n",)),
            ),
        )
        expected = {
            "triton": ("ssa-unified-triton-emitter", "if v2:"),
            "cuda": ("ssa-unified-cuda-emitter", "if (v2) {"),
            "tilelang": ("ssa-unified-tilelang-emitter", "if v2:"),
            "tvm": ("ssa-unified-tvm-emitter", "if v2:"),
        }
        for backend, (route, source_fragment) in expected.items():
            with self.subTest(backend=backend):
                artifact = lower_kernel_ir(kernel, backend)

                self.assertTrue(artifact.executable)
                self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
                self.assertEqual(artifact.metadata["source_route"], route)
                self.assertFalse(artifact.metadata["program_ir_compat"])
                self.assertIn(source_fragment, artifact.primary_source)
                self.assertIn("out0", artifact.primary_source)
                self.assertIn("out1", artifact.primary_source)
                self.assertEqual(artifact.primary_source.count(source_fragment), 1)

    def test_public_lower_generates_scf_for_loop_for_native_backends(self):
        expected = {
            "triton": ("ssa-unified-triton-emitter", "@triton.jit"),
            "cuda": ("ssa-unified-cuda-emitter", "for (int64_t"),
            "tilelang": ("ssa-unified-tilelang-emitter", "T.serial(2)"),
            "tvm": ("ssa-unified-tvm-emitter", "T.serial(2)"),
        }
        for backend, (route, source_fragment) in expected.items():
            with self.subTest(backend=backend):
                artifact = ninetoothed.lower(
                    arrangement,
                    loop_application,
                    (Tensor(1), Tensor(1)),
                    backend=backend,
                    kernel_name=f"ssa_loop_{backend}",
                )

                self.assertTrue(artifact.executable)
                self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
                self.assertEqual(artifact.metadata["source_route"], route)
                self.assertFalse(artifact.metadata["program_ir_compat"])
                self.assertIn(source_fragment, artifact.primary_source)
                self.assertIn("scf.for", str(artifact.metadata["ssa"]))

    def test_public_lower_generates_scf_if_for_native_backends(self):
        expected = {
            "triton": ("ssa-unified-triton-emitter", "@triton.jit"),
            "cuda": ("ssa-unified-cuda-emitter", " ? "),
            "tilelang": ("ssa-unified-tilelang-emitter", "T.if_then_else"),
            "tvm": ("ssa-unified-tvm-emitter", "T.if_then_else"),
        }
        for backend, (route, source_fragment) in expected.items():
            with self.subTest(backend=backend):
                artifact = ninetoothed.lower(
                    arrangement,
                    if_application,
                    (Tensor(1), Tensor(1)),
                    backend=backend,
                    kernel_name=f"ssa_if_{backend}",
                )

                self.assertTrue(artifact.executable)
                self.assertEqual(artifact.metadata["lowering_ir"], "SSAProgramIR")
                self.assertEqual(artifact.metadata["source_route"], route)
                self.assertFalse(artifact.metadata["program_ir_compat"])
                self.assertIn(source_fragment, artifact.primary_source)
                self.assertIn("scf.if", str(artifact.metadata["ssa"]))


if __name__ == "__main__":
    unittest.main()
