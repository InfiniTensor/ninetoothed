"""Aligned core vector validation shared by the two native vector targets."""

import ninetoothed as nt
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest, load_built_artifact


def _arrangement(x, y, out):
    return tuple(t.tile((128,)) for t in (x, y, out))


def _application(x, y, out):
    shared = x + y
    square = shared * shared
    out = (square - shared) * 0.5 + y  # noqa: F841


def vector_request(backend, n=1031, dtype="float32", candidate="vector-1024"):
    return CompileRequest(
        arrangement=_arrangement,
        application=_application,
        tensors=tuple(nt.Tensor(shape=(n,), dtype=dtype) for _ in range(3)),
        backend=backend,
        backend_options={
            "arch": {"bangc": "compute_50", "ascendc": "dav-c220"}[backend]
        },
        max_num_configs=1,
        pass_options={f"ssa.{backend}.optimize_schedule": {"candidate": candidate}}
        if candidate
        else None,
    )


def check_vector_numerics(tmp_path, *, backend, device):
    import torch

    native = getattr(torch, device)

    for dtype in (torch.float32, torch.float16):
        for n in (128, 129, 1031):
            torch.manual_seed(n)
            host = [torch.randn(n, dtype=dtype) / 8 for _ in range(2)]
            first = (host[0].double() + host[1].double()).to(dtype)
            square = (first.double() * first.double()).to(dtype)
            difference = (square.double() - first.double()).to(dtype)
            scale = (difference.double() * 0.5).float()
            expected = (scale.double() + host[1].double()).float().to(dtype)
            compilation = DEFAULT_COMPILER.compile(
                vector_request(backend, n, str(dtype))
            )
            handle = DEFAULT_COMPILER.materialize(compilation)
            a, b = [value.to(device) for value in host]
            prefix = 32 // a.element_size()
            # CPU initialization avoids a vendor torch_npu full/fill_ crash.
            owner = torch.full((prefix + n + 16,), -123.0, dtype=dtype).to(device)
            output = owner[prefix : prefix + n]
            assert output.data_ptr() % 32 == 0
            cpu_nan = torch.full((n,), float("nan"), dtype=dtype)
            stream = native.Stream()
            stream.wait_stream(native.current_stream())

            def execute(executable, x):
                with native.stream(stream):
                    output.copy_(cpu_nan)
                    executable(x, b, output)

                stream.synchronize()
                torch.testing.assert_close(
                    output.cpu(),
                    expected,
                    rtol=2e-3 if dtype == torch.float16 else 2e-5,
                    atol=2e-5 if dtype == torch.float16 else 2e-7,
                )
                assert (owner[:prefix].cpu() == -123).all()
                assert (owner[prefix + n :].cpu() == -123).all()

            execute(handle, a)
            host_strided = torch.empty(2 * n, dtype=dtype)
            host_strided[::2] = host[0]
            strided = host_strided.to(device)[::2]
            stream.wait_stream(native.current_stream())
            execute(handle, strided)

            if n == 129:
                built = DEFAULT_COMPILER.materialize(
                    compilation, mode="aot", output_dir=tmp_path / str(dtype)
                )
                execute(load_built_artifact(built._built_artifact), a)
