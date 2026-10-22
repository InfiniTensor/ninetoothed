"""Half storage crosses the host C ABI without changing device value semantics."""

import re

import ninetoothed as nt
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest


def _scalars(x, beta, alpha, out):
    return x.tile((8,)), beta, alpha, out.tile((8,))


def _scaled(x, beta, alpha, out):
    out = x * beta + alpha  # noqa: F841


def _compile():
    return DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_scalars,
            application=_scaled,
            tensors=(
                nt.Tensor(shape=(17,), dtype="float32"),
                nt.Tensor(shape=(), dtype="float16"),
                nt.Tensor(shape=(), dtype="float16"),
                nt.Tensor(shape=(17,), dtype="float32"),
            ),
            backend="ascendc",
            kernel_name="half_host_abi",
            max_num_configs=1,
        )
    )


def test_real_half_launch_uses_storage_and_decodes_before_all_host_work():
    compilation = _compile()
    source = compilation.artifact.primary_source
    kernel, host = source.split('extern "C" int launch_half_host_abi(', 1)
    signature, host_body = host.split(") {", 1)
    half_bindings = tuple(
        (
            binding.name
            for binding in compilation.launch_abi.kernel_args
            if binding.kind in {"scalar", "constexpr"}
        )
    )
    assert half_bindings == ("beta", "alpha")

    for name in half_bindings:
        storage = f"nt_host_half_bits_{name}"
        assert f"half {name}" in kernel
        assert f"uint16_t {storage}" in signature
        assert f"half {name}" not in signature
        decode = f"::memcpy(&{name}, &{storage}, sizeof({name}));"
        assert f"half {name};" in host_body and decode in host_body
        assert host_body.index(decode) < host_body.index("const int64_t")
        assert storage not in host_body.split(decode, 1)[1]
        assert re.search(f"\\b{name}\\b", host_body.split(">>>", 1)[1])

    assert compilation.artifact.metadata["ascendc_host_scalar_abi"] == {
        "version": "uint16-half-v1",
        "half_storage_bindings": half_bindings,
    }
    assert "#include <string.h>" in source
