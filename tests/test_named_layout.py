"""S1 tests grouped by shared layout, computation and runtime obligations."""

import ast
import pickle
from dataclasses import replace

import pytest
import torch

import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed.compiler import Compiler, CompileRequest, load_built_artifact
from ninetoothed.compiler.cache import compilation_cache_key
from ninetoothed.compiler.runtime import _launch_abi_from_dict, _runtime_wrapper
from ninetoothed.errors import LoweringError
from ninetoothed.ir.module import verify_module
from ninetoothed.naming import is_meta


def gemm_layout(a, b, out):
    return (
        a.axes("m", "k").tile((16, 32)),
        b.axes("k", "n").tile((32, 64)),
        out.axes("m", "n").tile((16, 64)),
    )


def old_layout(a, b, out):
    out = out.tile((16, 64))
    a = a.tile((16, 32)).tile((1, -1)).expand((-1, out.shape[1]))
    a.dtype = a.dtype.squeeze(0)
    b = b.tile((32, 64)).tile((-1, 1)).expand((out.shape[0], -1))
    b.dtype = b.dtype.squeeze(1)

    return a, b, out


def gemm(a, b, out):
    acc = ntl.zeros(out.shape, dtype=ntl.float32)

    for k in range(a.shape[0]):
        acc = acc + ntl.dot(a[k], b[k])

    out = acc.to(out.dtype)  # noqa: F841


def request(arrangement=gemm_layout, application=gemm, *, dtype="float32"):
    return CompileRequest(
        arrangement=arrangement,
        application=application,
        tensors=tuple(
            nt.Tensor(2, dtype=dtype, name=name) for name in ("a", "b", "out")
        ),
        backend="triton",
        compute_arch="sm_89",
        num_warps=4,
        num_stages=2,
    )


def test_named_annotations_match_the_arrangement_entrypoint():
    def annotated(
        x: nt.Tensor(1, dtype="float32", name="x").axes("n").tile((32,)),
        out: nt.Tensor(1, dtype="float32", name="out").axes("n").tile((32,)),
    ):
        out = x  # noqa: F841

    def layout(x, out):
        return x.axes("n").tile((32,)), out.axes("n").tile((32,))

    def application(x, out):
        out = x  # noqa: F841

    compiler = Compiler()
    annotated_result = compiler.compile(
        CompileRequest(application=annotated, kernel_name="named_copy")
    )
    arranged_result = compiler.compile(
        CompileRequest(
            arrangement=layout,
            application=application,
            tensors=(
                nt.Tensor(1, dtype="float32", name="x"),
                nt.Tensor(1, dtype="float32", name="out"),
            ),
            kernel_name="named_copy",
        )
    )
    assert (
        annotated_result.request.frontend.named_layout
        == arranged_result.request.frontend.named_layout
    )
    assert (
        annotated_result.artifact.primary_source
        == arranged_result.artifact.primary_source
    )


def test_named_schedule_uses_layout_tiles_not_symbol_names_or_presets():
    def layout(a, b, out):
        return (
            a.axes("rows", "inner").tile((block, 32)),
            b.axes("inner", "columns").tile((32, 64)),
            out.axes("rows", "columns").tile((block, 64)),
        )

    for block in (
        128,
        nt.Symbol("BLOCK_M", meta=True, lower_bound=128, upper_bound=128),
        nt.Symbol("BM", meta=True, lower_bound=128, upper_bound=128),
        nt.Symbol("BLOCK_N", constexpr=True, lower_bound=128, upper_bound=128),
    ):
        compiled = Compiler().compile(request(layout))
        metadata = compiled.artifact.metadata["ssa_metadata"]
        expected = {"block_m": 128, "block_n": 64, "block_k": 32}
        assert metadata["schedule"]["tile"] == expected
        assert all(
            item["schedule"]["tile"] == expected
            for item in metadata["schedule_candidates"]
        )
        assert all(
            item["tile"] == expected for item in compiled.launch_plan.tuning_candidates
        )

        if isinstance(block, nt.Symbol) and is_meta(str(block)):
            fixed = Compiler().compile(
                replace(request(layout), specialization_values={str(block): 128})
            )
            assert fixed.artifact.metadata["ssa_schedule"]["tile"] == expected

    for backend in ("cuda", "tilelang"):
        generated = Compiler().compile(replace(request(layout), backend=backend))
        assert generated.artifact.metadata["ssa_schedule"]["tile"] == expected

    block = nt.Symbol("BLOCK_M", meta=True, lower_bound=64, upper_bound=128)
    compiled = Compiler().compile(request(layout))
    assert {
        item["tile"]["block_m"] for item in compiled.launch_plan.tuning_candidates
    } == {64, 128}

    with pytest.raises(LoweringError, match="threads, target limit"):
        Compiler().compile(
            replace(
                request(layout, dtype="float16"),
                backend="cuda",
                backend_options={"max_threads_per_block": 128},
            )
        )


def test_axes_and_public_meta_operation_boundary():
    root = nt.Tensor(2)

    for names in (("m",), ("m", "m"), ("", "n")):
        with pytest.raises(ValueError):
            root.axes(*names)

    named = root.axes("m", "n")
    assert named._named_layout["axes"] == ("m", "n")

    with pytest.raises(ValueError, match="root"):
        root.tile((16, 16)).axes("m", "n")

    with pytest.raises(ValueError, match="one tile"):
        named.tile((16, 16)).tile((1, 1))

    with pytest.raises(ValueError, match="one tile"):
        named.expand((1, 1))

    # The legacy path keeps accepting its original meta-operation composition.
    assert root.tile((16, 16)).tile((1, -1)).ndim == 2


def test_named_gemm_has_the_legacy_layout_and_backend_source():
    compiler = Compiler()
    named = compiler.compile(request())
    old = compiler.compile(request(old_layout))

    for left, right in zip(named.kernel.tensors, old.kernel.tensors):
        for key in (
            "application_shape",
            "dtype_shapes",
            "dtype_target_dims",
            "access_templates",
        ):
            assert left.attrs[key] == right.attrs[key]

    # Equal kernel names, physical symbol names and layouts give equal code.
    assert named.artifact.primary_source == old.artifact.primary_source

    for backend in ("cuda", "tilelang"):
        compiled = compiler.compile(
            CompileRequest(frontend=named.request.frontend, backend=backend)
        )
        assert compiled.kernel.ssa is not None

        if backend == "tilelang":
            ast.parse(compiled.artifact.primary_source)


def wrong_dot(a, b, out):
    acc = ntl.zeros(out.shape, dtype=ntl.float32)

    for k in range(a.shape[0]):
        acc = acc + ntl.dot(b[k], a[k])

    out = acc  # noqa: F841


def test_named_dot_checks_axis_identity_not_equal_numeric_sizes():
    def square(a, b, out):
        return (
            a.axes("m", "k").tile((32, 32)),
            b.axes("k", "n").tile((32, 32)),
            out.axes("m", "n").tile((32, 32)),
        )

    with pytest.raises(LoweringError, match="Dot axes"):
        Compiler().compile(request(square, wrong_dot))


def test_named_axes_conflicts_are_rejected_before_codegen():
    def mismatch(a, b, out):
        return (
            a.axes("m", "k").tile((16, 16)),
            b.axes("k", "n").tile((32, 64)),
            out.axes("m", "n").tile((16, 64)),
        )

    with pytest.raises(LoweringError, match="axis 'k'"):
        Compiler().compile(request(mismatch))


def test_named_contract_survives_freeze_specialization_and_cache_identity():
    from ninetoothed.compiler.specialization import specialize_module
    from ninetoothed.ir.ssa import VerificationError

    compiler = Compiler()
    block = nt.Symbol("BLOCK_M", meta=True, lower_bound=16, upper_bound=32)

    def layout(a, b, out):
        return (
            a.axes("m", "k").tile((block, 32)),
            b.axes("k", "n").tile((32, 64)),
            out.axes("m", "n").tile((block, 64)),
        )

    module = compiler.frontend(request(layout))
    restored = pickle.loads(pickle.dumps(module))
    first = compiler.compile(CompileRequest(frontend=module))
    second = compiler.compile(CompileRequest(frontend=restored))
    assert compilation_cache_key(first) == compilation_cache_key(second)
    fixed = specialize_module(module, values={str(block): 16})
    assert fixed.named_layout["parameters"]["a"]["tile"][0] == "16"

    with pytest.raises(VerificationError):
        specialize_module(module, values={str(block): 0})

    with pytest.raises(LoweringError, match="discard"):
        verify_module(replace(module, named_layout=None))

    with pytest.raises(ValueError, match="Torch"):
        compiler.compile(CompileRequest(frontend=module, caller="c"))


def test_named_runtime_guards_are_retained_in_serialized_abi():
    compiled = Compiler().compile(request())
    abi = _launch_abi_from_dict(compiled.artifact.metadata["launch_abi"])
    specs = compiled.kernel.tensors
    a, b, out = torch.empty(17, 35), torch.empty(35, 19), torch.empty(17, 19)
    launches = []

    def launcher(*args):
        launches.append(args)

    handle = _runtime_wrapper(
        launcher, abi, low_level=False, specs=specs, device_types=("cpu",)
    )
    handle(a, b, out)
    assert len(launches) == 1

    for actual in (
        (torch.empty(18, 35), b, out),
        (a, torch.empty(35, 20), out),
        (a, torch.empty(36, 19), out),
    ):
        with pytest.raises(ValueError, match="Named axis"):
            handle(*actual)

    with pytest.raises(ValueError, match="overlap"):
        handle(a, b, a[:, :19])

    assert len(launches) == 1


def transposed_layout(a, b, out):
    return (
        a.axes("k", "m").permute((1, 0)).tile((16, 32)),
        b.axes("k", "n").tile((32, 64)),
        out.axes("m", "n").tile((16, 64)),
    )


def fused_layout(a, b, bias, residual, out):
    a, b, out = gemm_layout(a, b, out)

    return a, b, bias.axes("n").tile((64,)), residual.axes("m", "n").tile((16, 64)), out


def fused(a, b, bias, residual, out):
    acc = ntl.zeros(out.shape, dtype=ntl.float32)

    for k in range(0, b.shape[0], 1):
        left = a[k]
        right = b[k]
        acc += ntl.dot(left, right)

    out[...] = ntl.maximum(acc + bias + residual, 0).to(out.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_named_gemm_runtime_and_reload(dtype):
    compiler = Compiler()

    def symbolic_layout(a, b, out):
        block = nt.Symbol("BM", meta=True, lower_bound=128, upper_bound=128)

        return (
            a.axes("m", "k").tile((block, 32)),
            b.axes("k", "n").tile((32, 64)),
            out.axes("m", "n").tile((block, 64)),
        )

    layout = gemm_layout

    if dtype == torch.float32:
        layout = transposed_layout
    elif dtype == torch.float16:
        layout = symbolic_layout

    args = request(layout, dtype=str(dtype).split(".")[-1])
    compilation = compiler.compile(args)
    kernel = compiler.materialize(compilation)
    # Noncontiguous positive strides and all three tails, exact input products.
    a = (torch.arange(35 * 17, device="cuda") % 3 - 1).reshape(35, 17).t().to(dtype)
    b = (torch.arange(35 * 19, device="cuda") % 3 - 1).reshape(35, 19).to(dtype)

    if dtype == torch.float32:
        a = a.t()

    out = torch.empty(17, 19, device="cuda", dtype=dtype)
    kernel(a, b, out)
    expected = ((a.t() if dtype == torch.float32 else a).float() @ b.float()).to(dtype)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    reloaded = load_built_artifact(kernel._built_artifact)
    reloaded(a, b, out)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    with pytest.raises(ValueError, match="axis 'k'"):
        reloaded(a, torch.empty(36, 19, device="cuda", dtype=dtype), out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_named_fused_gemm_runtime():
    compilation = Compiler().compile(
        CompileRequest(
            arrangement=fused_layout,
            application=fused,
            tensors=(
                nt.Tensor(2),
                nt.Tensor(2),
                nt.Tensor(1),
                nt.Tensor(2),
                nt.Tensor(2),
            ),
        )
    )
    kernel = Compiler().materialize(compilation)
    a = torch.ones(17, 35, device="cuda", dtype=torch.float16)
    b = torch.ones(35, 19, device="cuda", dtype=torch.float16)
    bias = torch.arange(19, device="cuda", dtype=torch.float16) - 40
    residual = torch.ones(17, 19, device="cuda", dtype=torch.float16)
    out = torch.empty_like(residual)
    kernel(a, b, bias, residual, out)
    torch.testing.assert_close(
        out, (a.float() @ b.float() + bias + residual).relu().half(), rtol=0, atol=0
    )


def pointwise_layout(x, row, column, out):
    return (
        x.axes("n", "m").tile((32, 16)),
        row.axes("m").tile((16,)),
        column.axes("n").tile((32,)),
        out.axes("m", "n").tile((16, 32)),
    )


def pointwise(x, row, column, out):
    out = x * column + row  # noqa: F841


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
def test_named_pointwise_transpose_and_both_broadcast_axes():
    kernel = Compiler().materialize(
        CompileRequest(
            arrangement=pointwise_layout,
            application=pointwise,
            tensors=(
                nt.Tensor(2, dtype="int32"),
                nt.Tensor(1, dtype="int32"),
                nt.Tensor(1, dtype="int32"),
                nt.Tensor(2, dtype="int32"),
            ),
        )
    )
    x = torch.arange(35 * 17, dtype=torch.int32, device="cuda").reshape(35, 17)
    row = torch.arange(17, dtype=torch.int32, device="cuda")
    column = torch.arange(35, dtype=torch.int32, device="cuda")
    out = torch.empty(17, 35, dtype=torch.int32, device="cuda")
    kernel(x, row, column, out)
    torch.testing.assert_close(out, x.t() * column + row[:, None], rtol=0, atol=0)
