"""CTA matrix execution and its explicit scheduling boundary."""

from dataclasses import replace
from functools import partial

import pytest

import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest, load_built_artifact
from tests.test_matmul import arrangement


def _matrix(x, y, out):
    acc = ntl.zeros(out.shape, dtype=ntl.float32)

    for i in range(x.shape[0]):
        acc += ntl.dot(x[i], y[i])

    out = (acc * 0.5 + 1).to(out.dtype)  # noqa: F841


def request(m=64, n=64, k=64, *, candidate="wmma-32x32", arch="sm_89"):
    return CompileRequest(
        arrangement=partial(
            arrangement, BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32
        ),
        application=_matrix,
        tensors=tuple(
            Tensor(shape=shape, dtype=dtype, other=0)
            for shape, dtype in (
                ((m, k), "float16"),
                ((k, n), "float16"),
                ((m, n), "float32"),
            )
        ),
        backend="cuda",
        backend_options={"arch": arch},
        max_num_configs=1,
        pass_options={"ssa.cuda.optimize_schedule": {"candidate": candidate}}
        if candidate
        else None,
    )


def test_cta_collective_scope_and_layout_boundaries(monkeypatch):
    from ninetoothed.backends.emitters.cuda import CudaTarget, _prove_cta_source_loop
    from ninetoothed.backends.emitters.cuda_dot_transfer import prove_dot_transfers

    contexts = []
    original = CudaTarget.emit_reduction_loop

    def capture(self, local, loop, context):
        contexts.append((loop, context))

        return original(self, local, loop, context)

    monkeypatch.setattr(CudaTarget, "emit_reduction_loop", capture)
    compilation = DEFAULT_COMPILER.compile(request())
    source = compilation.artifact.primary_source
    assert source.index("cp.async.wait_group 0") < source.index(
        "wmma::store_matrix_sync"
    )
    assert source.index("wmma::store_matrix_sync") < source.index("nt_cta_item = 0")
    assert (
        "__syncthreads"
        not in source.split("nt_cta_item = 0", 1)[1].split('extern "C" int', 1)[0]
    )
    assert "__fadd_rn" in source and "_arena[5632]" in source
    assert compilation.launch_plan.block[0].value == 128
    loop, context = contexts[-1]
    dot = _prove_cta_source_loop(loop, context)
    output = context.tensor_infos[context.output]
    wrong_grid = replace(
        context,
        tensor_infos={
            **context.tensor_infos,
            context.output: replace(output, shape=("3", "2")),
        },
    )
    assert prove_dot_transfers(loop, dot, wrong_grid, stage="s", outer="o") is None
    operand = context.operations[dot.operands[0]].operands[0]
    info = context.tensor_infos[operand]
    wrong_stride = replace(
        context,
        tensor_infos={
            **context.tensor_infos,
            operand: replace(info, source_strides=("128", "2")),
        },
    )
    assert prove_dot_transfers(loop, dot, wrong_stride, stage="s", outer="o") is None
    extract = context.operations[dot.operands[0]]
    template = dict(extract.results[0].type.attrs["access_templates"][0])
    template.pop("level")
    bad_type = replace(
        extract.results[0].type,
        attrs={
            **extract.results[0].type.attrs,
            "dtype_level": 0,
            "access_templates": (template,),
        },
    )
    bad_extract = replace(
        extract, results=(replace(extract.results[0], type=bad_type),)
    )
    no_level = replace(
        context,
        operations={**context.operations, dot.operands[0]: bad_extract},
        value_types={**context.value_types, dot.operands[0]: bad_type},
        tensor_infos={
            **context.tensor_infos,
            operand: replace(
                info, attrs={**info.attrs, "access_templates": (template,)}
            ),
        },
    )
    assert prove_dot_transfers(loop, dot, no_level, stage="s", outer="o") is None
    initial = loop.attrs["iter_args"][0]["initial"]
    full = replace(
        context.operations[initial], opcode="tensor.full", attrs={"value": 0.0}
    )
    wrong_initial = replace(context, operations={**context.operations, initial: full})

    with pytest.raises(ValueError):
        _prove_cta_source_loop(loop, wrong_initial)

    with pytest.raises(ValueError):
        DEFAULT_COMPILER.compile(request(65, 67, 69))

    with pytest.raises(ValueError):
        DEFAULT_COMPILER.compile(request(arch="sm_75"))

    assert (
        "cp.async"
        not in DEFAULT_COMPILER.compile(
            request(65, 67, 69, candidate=None)
        ).artifact.primary_source
    )


def test_cta_runtime_strides_stream_and_reload(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")

    compilation = DEFAULT_COMPILER.compile(request())
    handle = DEFAULT_COMPILER.materialize(compilation, mode="aot", output_dir=tmp_path)
    reloaded = load_built_artifact(handle._built_artifact)
    jit = DEFAULT_COMPILER.materialize(compilation)
    torch.manual_seed(131)
    host_a, host_b = [torch.randn(64, 64, dtype=torch.float16) / 8 for _ in range(2)]
    expected = (host_a.double() @ host_b.double()) * 0.5 + 1

    for stride, offset in ((1, 0), (2, 1), (1, 1)):
        a_base = torch.zeros(64 * 64 * stride + offset, dtype=torch.float16)
        a_base[offset::stride] = host_a.flatten()
        a = a_base.cuda()[offset::stride].view(64, 64)
        b = host_b.cuda()
        storage = torch.full((64 * 64 + 2,), -123.0, device="cuda")
        out = storage[1:-1].view(64, 64)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())

        for executable in (handle, reloaded, jit):
            with torch.cuda.stream(stream):
                out.fill_(float("nan"))
                executable(a, b, out)

            stream.synchronize()
            torch.testing.assert_close(
                out.cpu().double(), expected, rtol=2e-5, atol=2e-6
            )
            assert storage[0].item() == storage[-1].item() == -123

    overlapping = torch.zeros((64, 64), device="cuda")
    aliased_input = overlapping.view(torch.float16).flatten()[:4096].view(64, 64)

    with pytest.raises(RuntimeError):
        jit(aliased_input, host_b.cuda(), overlapping)
