"""Mechanism-level checks for the neutral frontend/compiler boundary."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed.compiler import Compiler, CompileRequest
from ninetoothed.ir import ssa
from ninetoothed.ir.module import verify_module


def _arrangement(x, y, out):
    return tuple(tensor.tile((32,)) for tensor in (x, y, out))


def _application(x, y, out):
    out = (x + y).to(out.dtype)  # noqa: F841


def _request(**kwargs):
    return CompileRequest(
        arrangement=_arrangement,
        application=_application,
        tensors=(nt.Tensor(1), nt.Tensor(1), nt.Tensor(1)),
        **kwargs,
    )


def test_frozen_module_compiles_without_reentering_python(monkeypatch):
    compiler = Compiler()
    module = compiler.frontend(_request())

    def forbidden(*args, **kwargs):
        raise AssertionError("Compilation reentered the Python frontend.")

    monkeypatch.setattr("ninetoothed.frontend.module.freeze_application", forbidden)
    dtypes = {"x": "float16", "y": "float32", "out": "float32"}
    first = compiler.compile(
        CompileRequest(frontend=module, backend="triton", tensor_dtypes=dtypes)
    )
    second = compiler.compile(
        CompileRequest(frontend=module, backend="cuda", tensor_dtypes=dtypes)
    )
    assert first.kernel.ssa is not None and second.kernel.ssa is not None
    assert first.request.application is None and not first.request.tensors
    add = next(
        op for op in first.kernel.ssa.blocks[0].operations if op.opcode == "arith.add"
    )
    assert add.results[0].type.dtype == "float32"
    assert module.tensors[0].dtype is None


def _imperative_matmul(a, b, out):
    ntl.matmul(a, b, out)


def _matrix_arrangement(a, b, out):
    return tuple(tensor.tile((16, 16)) for tensor in (a, b, out))


def test_module_checks_linalg_shape_even_with_destination_dtype_reference():
    module = Compiler().frontend(
        CompileRequest(
            arrangement=_matrix_arrangement,
            application=_imperative_matmul,
            tensors=tuple(nt.Tensor(2, dtype="float32") for _ in range(3)),
        )
    )
    block = module.program.blocks[0]
    op = next(op for op in block.operations if op.opcode == "linalg.matmul")
    assert op.attrs["dtype_ref"] == "out"
    value = op.results[0]
    wrong = replace(
        op, results=(replace(value, type=replace(value.type, shape=("16", "8"))),)
    )
    program = replace(
        module.program,
        blocks=(
            replace(
                block,
                operations=tuple(
                    wrong if item is op else item for item in block.operations
                ),
            ),
        ),
    )

    with pytest.raises(ssa.VerificationError, match="Type mismatch"):
        verify_module(replace(module, program=program))


def test_lazy_dtype_specializations_reuse_the_frozen_frontend(monkeypatch, tmp_path):
    from ninetoothed.compiler import runtime

    compiler = Compiler()
    frozen = compiler.frontend(_request())
    fixed_size = next(
        name
        for name in frozen.symbols
        if name in frozen.tensors[0].attrs["source_shape"]
    )
    compiled = compiler.compile(
        CompileRequest(
            frontend=frozen,
            backend="cuda",
            compute_arch="sm_89",
            tensor_dtypes={"x": "float16"},
            specialization_values={fixed_size: 65},
        )
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Lazy specialization reentered the Python frontend.")

    monkeypatch.setattr("ninetoothed.frontend.module.freeze_application", forbidden)
    monkeypatch.setattr(
        runtime, "write_source", lambda *args, **kwargs: tmp_path / "lazy.cu"
    )
    monkeypatch.setattr(
        runtime,
        "_public_values",
        lambda abi, args, kwargs, **rest: dict(zip(abi.public_args, args)),
    )
    instances = []

    class FakeHandle:
        def __init__(self, compilation):
            self._compilation = compilation

            for name in (
                "artifact",
                "kernel",
                "source",
                "library",
                "ssa",
                "pass_trace",
                "launch_plan",
                "built_artifact",
            ):
                setattr(self, f"_{name}", None)

        def __call__(self, *args, **kwargs):
            return args[-1]

    def materialize(compilation, **kwargs):
        instances.append(compilation)

        return FakeHandle(compilation)

    monkeypatch.setattr(runtime, "materialize", materialize)
    handle = runtime._materialize_lazy(compiled)

    values = tuple(
        SimpleNamespace(
            shape=(65,),
            stride=lambda dim=None: (1,) if dim is None else 1,
            dtype="float16" if index == 0 else "float32",
        )
        for index in range(3)
    )
    assert handle(*values) is values[-1]
    assert handle(*values) is values[-1]
    assert len(instances) == 1
    specialized = instances[0]
    assert [item.dtype for item in specialized.kernel.tensors] == [
        "float16",
        "float32",
        "float32",
    ]
    assert specialized.request.specialization_values[fixed_size] == 65
    assert specialized.request.frontend is compiled.request.frontend


def test_shared_physical_symbols_are_checked_after_specialization_and_reload():
    from ninetoothed.backends.materializers.cpp import wrapper_source
    from ninetoothed.compiler.driver import _abi_dict
    from ninetoothed.compiler.runtime import _launch_abi_from_dict, _public_values
    from ninetoothed.symbol import Symbol

    n = Symbol("n")
    request = replace(
        _request(),
        tensors=tuple(
            nt.Tensor(shape=(extent,), dtype="float32") for extent in (n, 2 * n, n)
        ),
    )
    module = Compiler().frontend(request)
    compiled = Compiler().compile(CompileRequest(frontend=module))

    with pytest.raises(ValueError, match="composite physical layout guards"):
        wrapper_source(compiled, "enter", "leave")

    abi = _launch_abi_from_dict(_abi_dict(compiled.launch_abi))
    values = tuple(
        SimpleNamespace(shape=(size,), stride=lambda dim: 1) for size in (64, 128, 64)
    )
    assert _public_values(abi, values, {}, specs=()) == dict(
        zip(abi.public_args, values)
    )

    with pytest.raises(TypeError, match="Physical layout"):
        _public_values(
            abi,
            (
                values[0],
                SimpleNamespace(shape=(127,), stride=lambda dim: 1),
                values[2],
            ),
            {},
            specs=(),
        )

    fixed = Compiler().compile(
        CompileRequest(frontend=module, specialization_values={"n": 64})
    )
    fixed_abi = _launch_abi_from_dict(_abi_dict(fixed.launch_abi))
    assert "shape[0] != 128" in wrapper_source(fixed, "enter", "leave")

    with pytest.raises(TypeError, match="requires 64"):
        _public_values(
            fixed_abi,
            (SimpleNamespace(shape=(32,), stride=lambda dim: 1), values[1], values[2]),
            {},
            specs=(),
        )

    composite = Compiler().compile(
        replace(request, tensors=tuple(nt.Tensor(shape=(2 * n,)) for _ in range(3)))
    )
    composite_abi = _launch_abi_from_dict(_abi_dict(composite.launch_abi))
    repeated = (values[1],) * 3
    assert _public_values(composite_abi, repeated, {"n": 64}, specs=())["n"] == 64

    with pytest.raises(TypeError, match="Physical layout"):
        _public_values(composite_abi, repeated, {"n": 63}, specs=())
