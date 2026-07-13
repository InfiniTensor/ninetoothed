import pytest

import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor, block_size
from ninetoothed.backends import emit as emit_kernel
from ninetoothed.compiler import lower as lower_application
from ninetoothed.frontend.python import from_source
from ninetoothed.ir import Kernel, TensorSpec


def arrangement(x, out, BLOCK_SIZE=block_size()):
    return (x[0:BLOCK_SIZE], out[0:BLOCK_SIZE])


def offset_arrangement(x, out, BLOCK_SIZE=block_size()):
    return (x[1 : 1 + BLOCK_SIZE], out[1 : 1 + BLOCK_SIZE])


def add_application(x, out):
    out = x + x  # noqa: F841


def control_flow_application(x, out):
    acc = x

    for _ in range(2):
        acc = acc + x

    if x > x:
        acc = acc + x

    out = acc  # noqa: F841


def unsupported_application(x, out):
    tmp = {"value": x}
    out = tmp["value"]  # noqa: F841


def fused_expression_application(x, y, z, out):
    cond = (x > y) & (y < z)
    tmp = ntl.where(cond, ntl.exp(x) + ntl.sqrt(ntl.abs(y)), z)
    out = tmp * 2.0 - x  # noqa: F841


def reduction_arrangement(x, y, out, BLOCK_SIZE=block_size()):
    return (x[0:BLOCK_SIZE], y[0:BLOCK_SIZE], out[0:1])


def binary_arrangement(x, y, out, BLOCK_SIZE=block_size()):
    return (x[0:BLOCK_SIZE], y[0:BLOCK_SIZE], out[0:BLOCK_SIZE])


def ternary_arrangement(x, y, z, out, BLOCK_SIZE=block_size()):
    return (x[0:BLOCK_SIZE], y[0:BLOCK_SIZE], z[0:BLOCK_SIZE], out[0:BLOCK_SIZE])


def dot_reduction_application(x, y, out):
    out = (x * y).sum()  # noqa: F841


def fused_affine_helper(x, y, scale=2.0):
    product = x * y

    return product * scale + 1.0


def helper_call_application(x, y, out):
    out = fused_affine_helper(x, y, scale=3.0)  # noqa: F841


def _ssa_kernel(
    source: str, kernel_name: str, tensors: tuple[TensorSpec, ...]
) -> Kernel:
    ssa = from_source(source, tensors, kind=kernel_name)
    assert ssa is not None

    return Kernel(
        kernel_name=kernel_name,
        source=source,
        source_language="ninetoothed-python",
        entrypoint=kernel_name,
        tensors=tensors,
        ssa=ssa,
    )


def _assert_ssa_artifact(artifact, *, route):
    assert artifact.metadata["lowering_ir"] == "ssa.Program"
    assert artifact.metadata["source_route"] == route


class TestSSAFirstBackendLowering:
    def test_cuda_injects_curand_support_only_for_rand_operations(self):
        add_artifact = lower_application(
            arrangement,
            add_application,
            (Tensor(1), Tensor(1)),
            backend="cuda",
            kernel_name="cuda_without_rand",
        )
        assert "curand_kernel.h" not in add_artifact.primary_source
        assert "ninetoothed_curand_uniform" not in add_artifact.primary_source

        random_kernel = _ssa_kernel(
            """
def random_application(seed, out):
    index = out.offsets(0)
    out = rand(seed, index)
""",
            "cuda_with_rand",
            (
                TensorSpec(ndim=0, shape=(), dtype="int64", name="seed"),
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
            ),
        )
        random_artifact = emit_kernel(random_kernel, "cuda")
        source = random_artifact.primary_source
        assert "#include <curand_kernel.h>" in source
        assert "ninetoothed_curand_uniform" in source
        assert "curandStatePhilox4_32_10_t" in source
        assert "curand_init" in source

    def test_public_lower_generates_fused_expression_without_kernel_template(self):
        expected = {
            "triton": ("ssa-unified-triton-emitter", ("tl.where", "tl.exp", "tl.sqrt")),
            "cuda": ("ssa-unified-cuda-emitter", ("?", "expf", "sqrtf")),
            "tilelang": (
                "ssa-unified-tilelang-emitter",
                ("T.if_then_else", "T.exp", "T.sqrt"),
            ),
        }

        for backend, (route, fragments) in expected.items():
            artifact = lower_application(
                ternary_arrangement,
                fused_expression_application,
                (Tensor(1), Tensor(1), Tensor(1), Tensor(1)),
                backend=backend,
                kernel_name=f"ssa_fused_expr_{backend}",
            )
            _assert_ssa_artifact(artifact, route=route)
            assert "select.where" in str(artifact.metadata["ssa"])

            for fragment in fragments:
                assert fragment in artifact.primary_source

    def test_public_lower_uses_ssa_by_default_for_linear_backends(self):
        expected = {
            "triton": ("ssa-unified-triton-emitter", "@triton.jit"),
            "cuda": ("ssa-unified-cuda-emitter", "out[index] = v0;"),
            "tilelang": (
                "ssa-unified-tilelang-emitter",
                "v0 = (x_buf[index] + x_buf[index])",
            ),
        }

        for backend, (route, source_fragment) in expected.items():
            artifact = lower_application(
                arrangement,
                add_application,
                (Tensor(1), Tensor(1)),
                backend=backend,
                kernel_name=f"ssa_first_add_{backend}",
            )
            assert artifact.metadata["lowering_ir"] == "ssa.Program"
            assert "program_kind" not in artifact.metadata
            assert artifact.metadata["source_route"] == route
            assert "ssa" in artifact.metadata
            assert source_fragment in artifact.primary_source

    def test_public_lower_uses_arrangement_view_shapes_for_ssa_backends(self):
        for backend in ("triton", "cuda", "tilelang"):
            artifact = lower_application(
                arrangement,
                add_application,
                (Tensor(1), Tensor(1)),
                backend=backend,
                kernel_name=f"ssa_arrangement_view_shape_{backend}",
            )
            assert (
                artifact.metadata["kernel_metadata"]["ssa_tensor_ir_source"]
                == "arrangement_views"
            )
            assert "dim0" not in artifact.primary_source
            assert "BLOCK_SIZE" in artifact.primary_source
            assert any(
                (
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
        }

        for backend, load_fragment in expected.items():
            artifact = lower_application(
                offset_arrangement,
                add_application,
                (Tensor(1), Tensor(1)),
                backend=backend,
                kernel_name=f"ssa_arrangement_offset_{backend}",
            )
            source = artifact.primary_source
            assert load_fragment in source
            assert "+ 1" in source
            assert any(
                (
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
            return (
                x.tile((BLOCK_SIZE_M, BLOCK_SIZE_N)),
                out.tile((BLOCK_SIZE_M, BLOCK_SIZE_N)),
            )

        def tiled_application(x, out):
            out = x + x  # noqa: F841

        artifact = lower_application(
            tiled_arrangement,
            tiled_application,
            (Tensor(2), Tensor(2)),
            backend="cuda",
            kernel_name="ssa_cuda_index_cse_tiled_add",
        )
        source = artifact.primary_source
        assert "int64_t nt_idx_" in source
        assert "bool nt_pred_" in source
        assert "x[(nt_idx_" in source
        assert "out[(nt_idx_" in source
        assert source.count("floor(") < 4

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

            return (lhs_tiled, rhs_tiled, output_tiled)

        def matmul_application(lhs, rhs, output):
            accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

            for k in range(lhs.shape[0]):
                accumulator += ntl.dot(lhs[k], rhs[k])

            output = accumulator.to(ntl.float16)

        artifact = lower_application(
            matmul_arrangement,
            matmul_application,
            (Tensor(2), Tensor(2), Tensor(2)),
            backend="triton",
            kernel_name="public_triton_codegen_matmul_process",
        )
        assert artifact.metadata["source_route"] == "ssa-unified-triton-emitter"
        assert "linalg.matmul" not in str(artifact.metadata.get("ssa", ""))
        assert "@triton.jit" in artifact.primary_source
        assert "ssa.Program" in artifact.primary_source

    def test_tilelang_schedule_candidate_is_materialized_in_source_and_launch(self):
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

            return (lhs_tiled, rhs_tiled, output_tiled)

        def matmul_application(lhs, rhs, output):
            accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

            for k in range(lhs.shape[0]):
                accumulator += ntl.dot(lhs[k], rhs[k])

            output = accumulator.to(ntl.float16)  # noqa: F841

        def lower_candidate(candidate):
            return lower_application(
                matmul_arrangement,
                matmul_application,
                (
                    Tensor(2, dtype="float16"),
                    Tensor(2, dtype="float16"),
                    Tensor(2, dtype="float16"),
                ),
                backend="tilelang",
                kernel_name=f"tilelang_schedule_{candidate}",
                pass_options={
                    "ssa.tilelang.optimize_schedule": {"candidate": candidate}
                },
            )

        balanced = lower_candidate("balanced")
        wide = lower_candidate("wide")
        assert "threads=128" in balanced.primary_source
        assert "num_stages=2" in balanced.primary_source
        assert balanced.metadata["launch_block"] == ("128",)
        assert "threads=256" in wide.primary_source
        assert "num_stages=3" in wide.primary_source
        assert wide.metadata["launch_block"] == ("256",)
        assert balanced.metadata["ssa_schedule"]["tile"] == {
            "block_m": 64,
            "block_n": 64,
            "block_k": 32,
        }
        assert wide.metadata["ssa_schedule"]["tile"] == {
            "block_m": 128,
            "block_n": 64,
            "block_k": 32,
        }

    def test_public_triton_lower_raises_when_ssa_is_unavailable(self):
        with pytest.raises(Exception, match="Cannot lower `unsupported_application`"):
            lower_application(
                arrangement,
                unsupported_application,
                (Tensor(1), Tensor(1)),
                backend="triton",
                kernel_name="ssa_unavailable_no_generation_fallback",
            )

    def test_from_source_generates_tensor_construction_and_view_ops(self):
        cases = {
            "full": (
                "\ndef full_application(out):\n    out = full((n,), 2.5)\n",
                (TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),),
                {
                    "triton": "v1 = v0",
                    "cuda": "float v1 = v0;",
                    "tilelang": "v1 = v0",
                },
            ),
            "zeros": (
                "\ndef zeros_application(out):\n    out = zeros((n,))\n",
                (TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),),
                {
                    "triton": "v0 = 0.0",
                    "cuda": "float v0 = 0.0;",
                    "tilelang": "v0 = 0.0",
                },
            ),
            "view": (
                "\ndef view_application(x, out):\n    out = x[:, :]\n",
                (
                    TensorSpec(
                        ndim=2, shape=("rows", "cols"), dtype="float32", name="x"
                    ),
                    TensorSpec(
                        ndim=2, shape=("rows", "cols"), dtype="float32", name="out"
                    ),
                ),
                {
                    "triton": "tl.load(x + ((index // (cols)))",
                    "cuda": "x[((index / (cols)))",
                    "tilelang": "x_buf[((index // (cols)))",
                },
            ),
            "extract": (
                "\ndef extract_application(x, out):\n    out = x[0]\n",
                (
                    TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
                    TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
                ),
                {
                    "triton": "tl.load(x + v0",
                    "cuda": "x[v0]",
                    "tilelang": "x_buf[v0]",
                },
            ),
        }
        routes = {
            "triton": "ssa-unified-triton-emitter",
            "cuda": "ssa-unified-cuda-emitter",
            "tilelang": "ssa-unified-tilelang-emitter",
        }

        for case_name, (source, tensors, fragments) in cases.items():
            kernel = _ssa_kernel(source, f"ssa_linear_{case_name}", tensors)

            for backend, route in routes.items():
                artifact = emit_kernel(kernel, backend)
                assert artifact.metadata["lowering_ir"] == "ssa.Program"
                assert artifact.metadata["source_route"] == route
                assert fragments[backend] in artifact.primary_source

    def test_from_source_generates_shape_dim_for_native_backends(self):
        kernel = _ssa_kernel(
            "\ndef shape_dim_application(x, out):\n    out = x.shape[0]\n",
            "ssa_shape_dim",
            (
                TensorSpec(ndim=2, shape=("rows", "cols"), dtype="float32", name="x"),
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
            ),
        )
        operations = kernel.ssa.blocks[0].operations
        assert [operation.opcode for operation in operations] == [
            "shape.dim",
            "mem.store",
        ]
        assert operations[0].attrs["dim"] == 0
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
        }

        for backend, (route, source_fragments) in expected.items():
            artifact = emit_kernel(kernel, backend)
            _assert_ssa_artifact(artifact, route=route)

            for source_fragment in source_fragments:
                assert source_fragment in artifact.primary_source

    def test_from_source_generates_tensor_stride_for_native_backends(self):
        kernel = _ssa_kernel(
            "\ndef stride_application(x, out):\n    out = x.stride(0) + x.stride(1)\n",
            "ssa_tensor_stride",
            (
                TensorSpec(ndim=2, shape=("rows", "cols"), dtype="float32", name="x"),
                TensorSpec(ndim=1, shape=("n",), dtype="int64", name="out"),
            ),
        )
        operations = kernel.ssa.blocks[0].operations
        assert [operation.opcode for operation in operations] == [
            "tensor.stride",
            "tensor.stride",
            "arith.add",
            "mem.store",
        ]
        assert operations[0].attrs["dim"] == 0
        assert operations[1].attrs["dim"] == 1
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
        }

        for backend, (route, source_fragments) in expected.items():
            artifact = emit_kernel(kernel, backend)
            _assert_ssa_artifact(artifact, route=route)
            assert "lower_stride" not in artifact.primary_source

            for source_fragment in source_fragments:
                assert source_fragment in artifact.primary_source

    def test_canonical_elementwise_math_opcodes_emit_for_all_backends(self):
        kernel = _ssa_kernel(
            """
def canonical_math_application(x, y, out):
    selected = where(x > y, x, y)
    bounded = minimum(maximum(selected, y), x)
    out = (
        tanh(bounded)
        + log1p(abs(x))
        + atan2(x, y)
        + pow(abs(y) + 0.25, 0.5)
        + acos(x)
        + asin(y)
        + atan(x)
        + log10(abs(y) + 1.0)
        + expm1(x)
        + sinh(x)
        + cosh(y)
    )
""",
            "ssa_canonical_math",
            (
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="y"),
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
            ),
        )
        opcodes = {operation.opcode for operation in kernel.ssa.blocks[0].operations}
        required_opcodes = {
            "arith.maximum",
            "arith.minimum",
            "math.acos",
            "math.asin",
            "math.atan",
            "math.atan2",
            "math.cosh",
            "math.expm1",
            "math.log10",
            "math.log1p",
            "math.pow",
            "math.sinh",
            "math.tanh",
            "select.where",
        }
        assert required_opcodes <= opcodes, required_opcodes - opcodes
        expected = {
            "triton": (
                "ssa-unified-triton-emitter",
                (
                    "tl.where",
                    "tl.maximum",
                    "tl.minimum",
                    "tl.tanh",
                    "tl.log(1.0 +",
                    "tl.atan2",
                    "tl.pow",
                    "tl.acos",
                    "tl.asin",
                    "tl.atan",
                    "2.302585092994046",
                    "tl.exp",
                ),
            ),
            "cuda": (
                "ssa-unified-cuda-emitter",
                (
                    "?",
                    "fmaxf",
                    "fminf",
                    "tanhf",
                    "log1pf",
                    "atan2f",
                    "powf",
                    "acosf",
                    "asinf",
                    "atanf",
                    "log10f",
                    "expm1f",
                    "sinhf",
                    "coshf",
                ),
            ),
            "tilelang": (
                "ssa-unified-tilelang-emitter",
                (
                    "T.if_then_else",
                    "T.max",
                    "T.min",
                    "T.tanh",
                    "T.log1p",
                    "T.atan2",
                    "T.pow",
                    "T.acos",
                    "T.asin",
                    "T.atan",
                    "T.log10",
                    "T.exp",
                    "T.sinh",
                    "T.cosh",
                ),
            ),
        }

        for backend, (route, source_fragments) in expected.items():
            artifact = emit_kernel(kernel, backend)
            _assert_ssa_artifact(artifact, route=route)

            for source_fragment in source_fragments:
                assert source_fragment in artifact.primary_source, (
                    backend,
                    source_fragment,
                )

    def test_from_source_generates_python_expression_syntax_for_native_backends(self):
        kernel = _ssa_kernel(
            "\ndef python_expression_syntax_application(x, y, z, out):\n    tmp: float = x if 0 < 1 < 2 else y\n    pass\n    out = tmp + z\n    return out\n",
            "ssa_python_expression_syntax",
            (
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="y"),
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="z"),
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
            ),
        )
        opcodes = [operation.opcode for operation in kernel.ssa.blocks[0].operations]
        assert opcodes == [
            "arith.constant",
            "arith.constant",
            "cmp.lt",
            "arith.constant",
            "cmp.lt",
            "arith.and",
            "select.where",
            "arith.add",
            "mem.store",
        ]
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
        }

        for backend, (route, source_fragments) in expected.items():
            artifact = emit_kernel(kernel, backend)
            _assert_ssa_artifact(artifact, route=route)

            for source_fragment in source_fragments:
                assert source_fragment in artifact.primary_source

    def test_from_source_generates_bitwise_shifts_for_native_backends(self):
        kernel = _ssa_kernel(
            "\ndef bitwise_shift_application(x, y, out):\n    out = (x << 1) ^ (y >> 1)\n",
            "ssa_bitwise_shift",
            (
                TensorSpec(ndim=1, shape=("n",), dtype="int64", name="x"),
                TensorSpec(ndim=1, shape=("n",), dtype="int64", name="y"),
                TensorSpec(ndim=1, shape=("n",), dtype="int64", name="out"),
            ),
        )
        opcodes = [operation.opcode for operation in kernel.ssa.blocks[0].operations]
        assert "arith.bitwise_left_shift" in opcodes
        assert "arith.bitwise_right_shift" in opcodes
        assert "arith.bitwise_xor" in opcodes
        expected = {
            "triton": "ssa-unified-triton-emitter",
            "cuda": "ssa-unified-cuda-emitter",
            "tilelang": "ssa-unified-tilelang-emitter",
        }

        for backend, route in expected.items():
            artifact = emit_kernel(kernel, backend)
            _assert_ssa_artifact(artifact, route=route)
            assert "<<" in artifact.primary_source
            assert ">>" in artifact.primary_source
            assert "^" in artifact.primary_source

    def test_from_source_preserves_subscript_store_indices_for_native_backends(self):
        cases = {
            "one_dimensional": (
                "\ndef indexed_store_application(x, out):\n    i = x.offsets(0)\n    out[i] = x\n",
                (
                    TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
                    TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
                ),
                {
                    "triton": "tl.store(out + v0",
                    "cuda": "out[v0] = x[index];",
                    "tilelang": "out_buf[v0] = x_buf[index]",
                },
            ),
            "two_dimensional": (
                "\ndef indexed_store_2d_application(x, out):\n    i = x.offsets(0)\n    j = x.offsets(1)\n    out[i, j] = x\n",
                (
                    TensorSpec(
                        ndim=2, shape=("rows", "cols"), dtype="float32", name="x"
                    ),
                    TensorSpec(
                        ndim=2, shape=("rows", "cols"), dtype="float32", name="out"
                    ),
                ),
                {
                    "triton": "tl.store(out + (v0) * (cols) + (v1)",
                    "cuda": "out[(v0) * (cols) + (v1)] = x[((index / (cols)))",
                    "tilelang": "out_buf[(v0) * (cols) + (v1)] = x_buf[((index // (cols)))",
                },
            ),
        }
        routes = {
            "triton": "ssa-unified-triton-emitter",
            "cuda": "ssa-unified-cuda-emitter",
            "tilelang": "ssa-unified-tilelang-emitter",
        }

        for case_name, (source, tensors, fragments) in cases.items():
            kernel = _ssa_kernel(source, f"ssa_indexed_store_{case_name}", tensors)
            assert "'indices'" in str(kernel.ssa)

            for backend, route in routes.items():
                artifact = emit_kernel(kernel, backend)
                assert artifact.metadata["lowering_ir"] == "ssa.Program"
                assert artifact.metadata["source_route"] == route
                assert fragments[backend] in artifact.primary_source

    def test_from_source_expands_subscript_augassign_for_native_backends(self):
        kernel = _ssa_kernel(
            "\ndef indexed_augassign_application(x, out):\n    i = x.offsets(0)\n    out[i] += x\n",
            "ssa_indexed_augassign",
            (
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
            ),
        )
        operations = kernel.ssa.blocks[0].operations
        assert [operation.opcode for operation in operations] == [
            "index.offset",
            "tensor.extract",
            "arith.add",
            "mem.store",
        ]
        assert operations[-1].attrs["indices"] == ("%0",)
        expected = {
            "triton": ("ssa-unified-triton-emitter", "tl.store(out + v0"),
            "cuda": ("ssa-unified-cuda-emitter", "out[v0] = v2;"),
            "tilelang": ("ssa-unified-tilelang-emitter", "out_buf[v0] = v2"),
        }

        for backend, (route, source_fragment) in expected.items():
            artifact = emit_kernel(kernel, backend)
            _assert_ssa_artifact(artifact, route=route)
            assert source_fragment in artifact.primary_source

    def test_from_source_linearizes_multidimensional_extract_by_source_shape(self):
        kernel = _ssa_kernel(
            "\ndef extract_2d_application(x, out):\n    i = x.offsets(0)\n    j = x.offsets(1)\n    out = x[i, j]\n",
            "ssa_extract_2d",
            (
                TensorSpec(ndim=2, shape=("rows", "cols"), dtype="float32", name="x"),
                TensorSpec(ndim=2, shape=("rows", "cols"), dtype="float32", name="out"),
            ),
        )
        expected = {
            "triton": (
                "ssa-unified-triton-emitter",
                "tl.load(x + (v0) * (cols) + (v1)",
            ),
            "cuda": ("ssa-unified-cuda-emitter", "x[(v0) * (cols) + (v1)]"),
            "tilelang": ("ssa-unified-tilelang-emitter", "x_buf[(v0) * (cols) + (v1)]"),
        }
        extract = kernel.ssa.blocks[0].operations[2]
        assert extract.opcode == "tensor.extract"
        assert extract.operands == ("x", "%0", "%1")

        for backend, (route, source_fragment) in expected.items():
            artifact = emit_kernel(kernel, backend)
            _assert_ssa_artifact(artifact, route=route)
            assert source_fragment in artifact.primary_source
            assert "x + v0 + v1" not in artifact.primary_source
            assert "x[v0 + v1]" not in artifact.primary_source
            assert "x_buf[v0 + v1]" not in artifact.primary_source

    def test_public_lower_generates_ssa_reduction_for_native_backends(self):
        expected = {
            "triton": ("ssa-unified-triton-emitter", "@triton.jit"),
            "cuda": ("ssa-unified-cuda-emitter", "(x[v1_i] * y[v1_i])"),
            "tilelang": ("ssa-unified-tilelang-emitter", "x_buf[v1_i] * y_buf[v1_i]"),
        }

        for backend, (route, source_fragment) in expected.items():
            artifact = lower_application(
                reduction_arrangement,
                dot_reduction_application,
                (Tensor(1), Tensor(1), Tensor(1)),
                backend=backend,
                kernel_name=f"ssa_dot_reduce_{backend}",
            )
            _assert_ssa_artifact(artifact, route=route)
            assert source_fragment in artifact.primary_source
            assert "reduce.sum" in str(artifact.metadata["ssa"])

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
        }

        for backend, (route, source_fragments) in expected.items():
            artifact = lower_application(
                binary_arrangement,
                helper_call_application,
                (Tensor(1), Tensor(1), Tensor(1)),
                backend=backend,
                kernel_name=f"ssa_helper_inline_{backend}",
            )
            _assert_ssa_artifact(artifact, route=route)
            assert "call.fused_affine_helper" not in str(artifact.metadata["ssa"])
            assert "fused_affine_helper" not in artifact.primary_source

            for source_fragment in source_fragments:
                assert source_fragment in artifact.primary_source

    def test_from_source_generates_axis_reduction_for_native_backends(self):
        kernel = _ssa_kernel(
            "\ndef axis_addmv_application(bias, a, x, out):\n    out = bias + sum(a * x, axis=1)\n",
            "ssa_axis_addmv",
            (
                TensorSpec(ndim=1, shape=("rows",), dtype="float32", name="bias"),
                TensorSpec(ndim=2, shape=("rows", "cols"), dtype="float32", name="a"),
                TensorSpec(ndim=1, shape=("cols",), dtype="float32", name="x"),
                TensorSpec(ndim=1, shape=("rows",), dtype="float32", name="out"),
            ),
        )
        expected = {
            "triton": ("ssa-unified-triton-emitter", "for v1_i in range(0, cols, 1):"),
            "cuda": (
                "ssa-unified-cuda-emitter",
                "a[(index) * (cols) + (v1_i)] * x[v1_i]",
            ),
            "tilelang": ("ssa-unified-tilelang-emitter", "for v1_i in T.serial(cols):"),
        }

        for backend, (route, source_fragment) in expected.items():
            artifact = emit_kernel(kernel, backend)
            _assert_ssa_artifact(artifact, route=route)
            assert source_fragment in artifact.primary_source
            assert "reduce.sum" in str(artifact.metadata["ssa"])

    def test_from_source_generates_rowwise_reduction_for_native_backends(self):
        kernel = _ssa_kernel(
            "\ndef rowwise_norm_application(x, out):\n    out = x / sum(x, axis=1)\n",
            "ssa_rowwise_norm",
            (
                TensorSpec(ndim=2, shape=("rows", "cols"), dtype="float32", name="x"),
                TensorSpec(ndim=2, shape=("rows", "cols"), dtype="float32", name="out"),
            ),
        )
        expected = {
            "triton": ("ssa-unified-triton-emitter", "for v0_i in range(0, cols, 1):"),
            "cuda": ("ssa-unified-cuda-emitter", "out[index] = v1;"),
            "tilelang": ("ssa-unified-tilelang-emitter", "out_buf[index] = v1"),
        }

        for backend, (route, source_fragment) in expected.items():
            artifact = emit_kernel(kernel, backend)
            _assert_ssa_artifact(artifact, route=route)
            assert source_fragment in artifact.primary_source
            assert "reduce.sum" in str(artifact.metadata["ssa"])

    def test_transpose_aliases_share_one_backend_emission_contract(self):
        tensors = (
            TensorSpec(ndim=2, shape=("rows", "cols"), dtype="float32", name="x"),
            TensorSpec(ndim=2, shape=("cols", "rows"), dtype="float32", name="out"),
        )
        kernel = _ssa_kernel(
            "\ndef transpose_application(x, out):\n    out = transpose(x)\n",
            "ssa_transpose",
            tensors,
        )
        attribute_kernel = _ssa_kernel(
            "\ndef transpose_attribute_application(x, out):\n    out = x.T\n",
            "ssa_transpose_attribute",
            tensors,
        )

        for frontend_kernel in (kernel, attribute_kernel):
            transpose = frontend_kernel.ssa.blocks[0].operations[0]
            assert transpose.opcode == "linalg.transpose"
            assert transpose.operands == ("x",)

        expected = {
            "triton": (
                "ssa-unified-triton-emitter",
                "tl.load(x + (v2) * (cols) + (v1)",
            ),
            "cuda": ("ssa-unified-cuda-emitter", "x[(v2) * (cols) + (v1)]"),
            "tilelang": ("ssa-unified-tilelang-emitter", "x_buf[(v2) * (cols) + (v1)]"),
        }

        for backend, (route, source_fragment) in expected.items():
            artifact = emit_kernel(kernel, backend)
            _assert_ssa_artifact(artifact, route=route)
            assert "linalg.transpose" not in str(artifact.metadata["ssa"])
            assert source_fragment in artifact.primary_source

    def test_from_source_emits_store_inside_scf_for_without_operator_dispatch(self):
        kernel = _ssa_kernel(
            "\ndef loop_store_application(x, out):\n    for i in range(n):\n        out[i] = x[i] + 1.0\n",
            "ssa_loop_store",
            (
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
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
        }

        for backend, (route, source_fragments) in expected.items():
            artifact = emit_kernel(kernel, backend)
            _assert_ssa_artifact(artifact, route=route)
            assert "scf.for" in str(artifact.metadata["ssa"])
            assert "lower_loop_store" not in artifact.primary_source

            for source_fragment in source_fragments:
                assert source_fragment in artifact.primary_source

    def test_from_source_preserves_else_store_region_for_side_effect_if(self):
        kernel = _ssa_kernel(
            "\ndef if_else_store_application(x, y, out):\n    if 1 < 2:\n        i = x.offsets(0)\n        out[i] = x\n    else:\n        j = y.offsets(0)\n        out[j] = y\n",
            "ssa_if_else_store",
            (
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="y"),
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
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
        }
        op = next(
            (
                operation
                for operation in kernel.ssa.blocks[0].operations
                if operation.opcode == "scf.if"
            )
        )
        assert tuple((region.name for region in op.regions)) == ("then", "else")

        for backend, (route, source_fragments) in expected.items():
            artifact = emit_kernel(kernel, backend)
            _assert_ssa_artifact(artifact, route=route)

            for source_fragment in source_fragments:
                assert source_fragment in artifact.primary_source

    def test_from_source_emits_multi_result_scf_if_once(self):
        kernel = _ssa_kernel(
            "\ndef multi_result_if_application(x, y, out0, out1):\n    a = x\n    b = y\n    if 1 < 2:\n        a = x + y\n        b = x - y\n    out0 = a\n    out1 = b\n",
            "ssa_multi_result_if",
            (
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="y"),
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out0"),
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out1"),
            ),
        )
        expected = {
            "triton": ("ssa-unified-triton-emitter", "if v2:"),
            "cuda": ("ssa-unified-cuda-emitter", "if (v2) {"),
            "tilelang": ("ssa-unified-tilelang-emitter", "if v2:"),
        }

        for backend, (route, source_fragment) in expected.items():
            artifact = emit_kernel(kernel, backend)
            _assert_ssa_artifact(artifact, route=route)
            assert source_fragment in artifact.primary_source
            assert "out0" in artifact.primary_source
            assert "out1" in artifact.primary_source
            assert artifact.primary_source.count(source_fragment) == 1

    def test_public_lower_generates_loop_and_if_control_flow_for_all_backends(self):
        expected = {
            "triton": ("ssa-unified-triton-emitter", ("@triton.jit", "tl.where")),
            "cuda": ("ssa-unified-cuda-emitter", ("for (int64_t", " ? ")),
            "tilelang": (
                "ssa-unified-tilelang-emitter",
                ("T.serial(2)", "T.if_then_else"),
            ),
        }

        for backend, (route, source_fragments) in expected.items():
            artifact = lower_application(
                arrangement,
                control_flow_application,
                (Tensor(1), Tensor(1)),
                backend=backend,
                kernel_name=f"ssa_control_flow_{backend}",
            )
            _assert_ssa_artifact(artifact, route=route)
            rendered_ssa = str(artifact.metadata["ssa"])
            assert "scf.for" in rendered_ssa
            assert "scf.if" in rendered_ssa

            for source_fragment in source_fragments:
                assert source_fragment in artifact.primary_source, (
                    backend,
                    source_fragment,
                )
