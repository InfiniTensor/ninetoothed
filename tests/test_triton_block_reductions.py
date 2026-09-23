from dataclasses import replace

import pytest
import torch

import ninetoothed.language as ntl
from ninetoothed import Tensor, float32
from ninetoothed.backends.triton_reductions import TritonBlockReductions
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest
from ninetoothed.compiler.passes import Context, default_spec
from ninetoothed.ir import ssa
from tests.utils import get_available_devices

PASS = "ssa.triton.block_reductions"


def _arrangement(x, out):
    x = x.tile((1, -1, -1))
    x.dtype = x.dtype.squeeze(0)
    out = out.tile((1, -1))
    out.dtype = out.dtype.squeeze(0)

    return x, out


def _application(x, out):
    weights = ntl.sum(x, axis=1)
    out = ntl.sum(x * weights[:, None], axis=0)  # noqa: F841


def _request(shape=(3, 8, 16), enabled=True):
    passes = default_spec("triton").passes

    return CompileRequest(
        arrangement=_arrangement,
        application=_application,
        tensors=(
            Tensor(shape=shape, dtype=float32),
            Tensor(shape=(shape[0], shape[2]), dtype=float32),
        ),
        backend="triton",
        pipeline=(*passes[:-1], PASS, passes[-1]) if enabled else None,
        num_warps=4,
        max_num_configs=1,
    )


def test_block_reduction_guards_preserve_the_original_schedule():
    compilation = DEFAULT_COMPILER.compile(_request(enabled=False))
    program = compilation.kernel.ssa
    context = Context(
        backend=compilation.target.backend,
        compiler_options={},
        kernel_metadata={},
        tensors=compilation.kernel.tensors,
    )
    transform = TritonBlockReductions()
    limited = replace(context, pass_options={PASS: {"max_block_elements": 32}})
    operations = program.blocks[0].operations
    unsupported = replace(
        program,
        blocks=(
            replace(
                program.blocks[0],
                operations=(
                    ssa.Operation(opcode="call.unknown"),
                    *operations,
                ),
            ),
        ),
    )

    for candidate, candidate_context in ((program, limited), (unsupported, context)):
        rejected = transform.run(candidate, candidate_context)
        schedule = dict(rejected.metadata["schedule"])
        assert not schedule.pop("block_reductions")["enabled"]
        assert schedule == dict(candidate.metadata.get("schedule", {}))
        assert replace(rejected, metadata=candidate.metadata) == candidate


@pytest.mark.parametrize("device", get_available_devices())
def test_block_reduction_runtime(device):
    shape = (3, 8, 16)
    x = torch.randn((*shape[:-1], shape[-1] * 2), device=device)[..., ::2]
    output = torch.empty((shape[0], shape[2]), device=device)
    compilation = DEFAULT_COMPILER.compile(_request(shape))
    source = compilation.artifact.primary_source

    assert PASS not in default_spec("triton").passes
    assert PASS in compilation.pass_trace
    assert source.count("tl.sum(") == 2
    assert "for " not in source
    assert compilation.artifact.metadata["ssa_metadata"]["schedule"][
        "block_reductions"
    ]["enabled"]

    DEFAULT_COMPILER.materialize(compilation)(x, output)
    expected = (x.double() * x.double().sum(dim=2, keepdim=True)).sum(dim=1)
    torch.testing.assert_close(output, expected.float(), rtol=2e-5, atol=2e-5)


def _independent_arrangement(x, y, out):
    return x.tile((-1,)), y.tile((-1,)), out.tile((-1,))


def _independent_application(x, y, out):
    out = x + ntl.sum(y, axis=0)  # noqa: F841


@pytest.mark.parametrize("device", get_available_devices())
def test_block_reduction_loads_each_input_domain(device):
    request = replace(
        _request(),
        arrangement=_independent_arrangement,
        application=_independent_application,
        tensors=(
            Tensor(shape=(8,), dtype=float32),
            Tensor(shape=(16,), dtype=float32),
            Tensor(shape=(8,), dtype=float32),
        ),
    )
    x = torch.zeros(8, device=device)
    y = torch.arange(16, device=device, dtype=torch.float32)
    out = torch.empty_like(x)

    for enabled in (False, True):
        compilation = DEFAULT_COMPILER.compile(
            request if enabled else replace(request, pipeline=None)
        )

        if enabled:
            assert compilation.artifact.metadata["ssa_metadata"]["schedule"][
                "block_reductions"
            ]["enabled"]

        DEFAULT_COMPILER.materialize(compilation)(x, y, out)
        torch.testing.assert_close(out, torch.full_like(out, 120), rtol=0, atol=0)
