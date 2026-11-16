"""Pure vector fusion, its explicit boundary and native execution."""

from dataclasses import replace

import pytest

import ninetoothed as nt
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest, load_built_artifact


def _arrangement(x, y, out):
    return tuple(t.tile((128,)) for t in (x, y, out))


def _application(x, y, out):
    shared = x + y
    square = shared * shared
    out = (square - shared) * 0.5 + y  # noqa: F841


def request(n=1031, dtype="float32", candidate="vector-1024"):
    return CompileRequest(
        arrangement=_arrangement,
        application=_application,
        tensors=tuple(nt.Tensor(shape=(n,), dtype=dtype) for _ in range(3)),
        backend="bangc",
        backend_options={"arch": "compute_50"},
        max_num_configs=1,
        pass_options={"ssa.bangc.optimize_schedule": {"candidate": candidate}}
        if candidate
        else None,
    )


def test_vector_schedule_and_typed_storage(monkeypatch):
    from ninetoothed.backends.emitters import bangc_vector

    contexts = []
    original = bangc_vector.prove_flat_domain

    def capture(context):
        contexts.append(context)

        return original(context)

    monkeypatch.setattr(bangc_vector, "prove_flat_domain", capture)
    compilation = DEFAULT_COMPILER.compile(request())
    source = compilation.artifact.primary_source
    assert "__bang_add(" in source and "__bang_mul(" in source
    assert "__bang_mul_scalar(" in source
    assert "GDRAM2NRAM" in source and "NRAM2GDRAM" in source
    assert compilation.artifact.metadata["bangc_vector"]["storage_bytes"] <= 384 * 1024
    assert (
        "__bang_add("
        not in DEFAULT_COMPILER.compile(request(candidate=None)).artifact.primary_source
    )
    assert (
        "nt_vec_start"
        in DEFAULT_COMPILER.compile(
            request(candidate="vector-4096")
        ).artifact.primary_source
    )

    assert (
        "__bang_float2half_rn"
        in DEFAULT_COMPILER.compile(request(dtype="float16")).artifact.primary_source
    )
    context = contexts[0]
    changed = replace(
        context.value_types["x"],
        attrs={**context.value_types["x"].attrs, "dtype_level": 1},
    )
    assert (
        original(replace(context, value_types={**context.value_types, "x": changed}))
        is None
    )
    assert (
        original(
            replace(
                context,
                value_types={
                    **context.value_types,
                    "x": replace(context.value_types["x"], shape=("1",)),
                },
            )
        )
        is None
    )

    with pytest.raises(ValueError):
        DEFAULT_COMPILER.compile(request(dtype="int32"))


def test_vector_native_tail_reload_and_fallback(tmp_path):
    import torch

    pytest.importorskip("torch_mlu")

    if not torch.mlu.is_available():
        pytest.skip("MLU device required")

    for dtype in (torch.float32, torch.float16):
        for n in (128, 129, 1031):
            torch.manual_seed(n)
            host = [torch.randn(n, dtype=dtype) / 8 for _ in range(2)]
            first = (host[0].double() + host[1].double()).to(dtype)
            square = (first.double() * first.double()).to(dtype)
            difference = (square.double() - first.double()).to(dtype)
            scale = (difference.double() * 0.5).float()
            expected = (scale.double() + host[1].double()).float().to(dtype)
            compilation = DEFAULT_COMPILER.compile(request(n, str(dtype)))
            handle = DEFAULT_COMPILER.materialize(compilation)
            a, b = [v.to("mlu") for v in host]
            storage = torch.full((n + 2,), -123.0, device="mlu", dtype=dtype)
            output = storage[1:-1]
            output.fill_(float("nan"))
            handle(a, b, output)
            torch.mlu.synchronize()
            torch.testing.assert_close(
                output.cpu(),
                expected,
                rtol=2e-3 if dtype == torch.float16 else 2e-5,
                atol=2e-5 if dtype == torch.float16 else 2e-7,
            )
            assert storage[0].item() == storage[-1].item() == -123
            # A strided input must execute the original indexed fallback.
            strided = torch.empty(2 * n, device="mlu", dtype=dtype)[::2]
            strided.copy_(a)
            output.fill_(float("nan"))
            handle(strided, b, output)
            torch.mlu.synchronize()
            torch.testing.assert_close(
                output.cpu(),
                expected,
                rtol=2e-3 if dtype == torch.float16 else 2e-5,
                atol=2e-5 if dtype == torch.float16 else 2e-7,
            )

            if n == 129:
                built = DEFAULT_COMPILER.materialize(
                    compilation, mode="aot", output_dir=tmp_path / str(dtype)
                )
                output.fill_(float("nan"))
                load_built_artifact(built._built_artifact)(a, b, output)
                torch.mlu.synchronize()
                torch.testing.assert_close(
                    output.cpu(),
                    expected,
                    rtol=2e-3 if dtype == torch.float16 else 2e-5,
                    atol=2e-5 if dtype == torch.float16 else 2e-7,
                )
