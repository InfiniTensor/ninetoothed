"""Mechanism-level checks for the neutral frontend/compiler boundary."""

import ast
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


class _FakeHandle:
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

    def materialize(compilation, **kwargs):
        instances.append(compilation)

        return _FakeHandle(compilation)

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


def _pair_arrangement(x, out):
    return x.tile((32,)), out.tile((32,))


def _dtype_branch(x, out):
    value = x + 2

    if x.dtype == ntl.float16:
        value = x.to(ntl.float32) + 1

    out = value  # noqa: F841


def _dtype_conditional_store(x, out):
    out = x

    if x.dtype == ntl.float16:
        out[0] = x[0]

    if x.dtype != ntl.float16:
        pass


def test_dtype_binding_preserves_type_selected_control_flow():
    from ninetoothed.compiler.specialization import specialize_module
    from ninetoothed.errors import LoweringError

    compiler = Compiler()
    request = CompileRequest(
        arrangement=_pair_arrangement,
        application=_dtype_branch,
        tensors=(nt.Tensor(1), nt.Tensor(1, dtype="float32")),
    )
    frozen = compiler.frontend(request)

    for dtype, delta in (("float16", 1), ("bfloat16", 2)):
        direct = compiler.frontend(replace(request, tensor_dtypes={"x": dtype}))
        specialized = specialize_module(frozen, dtypes={"x": dtype})

        for module in (direct, specialized):
            operations = module.program.blocks[0].operations
            producers = {value.name: op for op in operations for value in op.results}
            store = next(op for op in reversed(operations) if op.opcode == "mem.store")
            result = producers[store.operands[0]]
            assert not any(op.opcode == "scf.if" for op in operations)
            assert producers[result.operands[1]].attrs["value"] == delta
            assert result.results[0].type.dtype == ("float32" if delta == 1 else dtype)

        with pytest.raises(LoweringError, match="Cannot rebind dtype"):
            specialize_module(direct, dtypes={"x": "float32"})

        effects = compiler.frontend(
            replace(request, application=_dtype_conditional_store)
        )
        effects = specialize_module(effects, dtypes={"x": dtype})
        stores = [
            op
            for op in effects.program.blocks[0].operations
            if op.opcode == "mem.store"
        ]
        assert len(stores) == (2 if dtype == "float16" else 1)


def _masked_values(x, out):
    a = ntl.where(x > 0, 1.0, 0.0)
    b = 0.0 if x > 0 else x
    c = 1.0 if x > 0 else 0.0
    out = a + b + c  # noqa: F841


def _augmented_values(x, out):
    value = x
    value += 1.0
    out = value
    out[0] += 1.0


def _scalar_boolean(x, out):
    n = x.source.shape[0]
    m = n * 2

    if n and m:
        out = x

    if n or m:
        out = x

    if not n:
        out = x  # noqa: F841


@pytest.mark.parametrize(
    "application", (_masked_values, _augmented_values, _scalar_boolean)
)
def test_frontend_operation_types_match_shared_inference(application):
    compiler = Compiler()
    module = compiler.frontend(
        CompileRequest(
            arrangement=_pair_arrangement,
            application=application,
            tensors=(nt.Tensor(1, dtype="float16"), nt.Tensor(1, dtype="float16")),
        )
    )

    operations = module.program.blocks[0].operations
    types = {value.name: value.type for value in module.program.inputs}

    for op in operations:
        if op.opcode in {"arith.and", "arith.or"}:
            assert all(types[name].dtype == "bool" for name in op.operands)
            numeric = next(item for item in operations if item.opcode == "shape.dim")
            invalid = replace(op, operands=(numeric.results[0].name, op.operands[1]))
            block = replace(
                module.program.blocks[0],
                operations=tuple(
                    invalid if item is op else item for item in operations
                ),
            )

            with pytest.raises(ssa.VerificationError, match="Logical operands"):
                verify_module(
                    replace(module, program=replace(module.program, blocks=(block,)))
                )

        types.update((value.name, value.type) for value in op.results)

    for backend in ("triton", "cuda", "tilelang"):
        compiled = compiler.compile(CompileRequest(frontend=module, backend=backend))

        if backend != "cuda":
            ast.parse(compiled.artifact.primary_source)


def _pointer_assignment(x, out):
    ptr = x.data_ptr()
    ptr = ptr + 2
    ptr = ptr - 1
    out = ntl.load(ptr)  # noqa: F841


def _pointer_augmented(x, out):
    ptr = x.data_ptr()
    ptr += 2
    ptr -= 1
    out = ntl.load(ptr)  # noqa: F841


def test_pointer_augassign_matches_assignment_before_and_after_binding():
    from ninetoothed.compiler.specialization import specialize_module

    compiler = Compiler()
    programs = []

    for application in (_pointer_assignment, _pointer_augmented):
        request = CompileRequest(
            arrangement=_pair_arrangement,
            application=application,
            tensors=(nt.Tensor(1, name="x"), nt.Tensor(1, name="out")),
            kernel_name="pointer_arithmetic",
        )
        unknown = compiler.frontend(request)
        known = compiler.frontend(
            replace(request, tensor_dtypes={"x": "float32", "out": "float32"})
        )
        bound = specialize_module(unknown, dtypes={"x": "float32", "out": "float32"})
        stages = []

        for module in (unknown, known, bound):
            operations = module.program.blocks[0].operations
            assert not any(
                op.opcode in {"dtype.of", "tensor.cast"} for op in operations
            )
            assert all(
                op.results[0].type.kind == "pointer"
                for op in operations
                if op.opcode in {"arith.add", "arith.sub"}
            )
            compiled = compiler.compile(
                CompileRequest(frontend=module, backend="triton")
            )
            ast.parse(compiled.artifact.primary_source)
            stages.append(compiled.artifact.primary_source)

        programs.append(stages)

    assert programs[0] == programs[1]


def test_runtime_meta_defaults_finish_specialization(monkeypatch, tmp_path):
    from ninetoothed.backends import materializers
    from ninetoothed.compiler import runtime

    bm = nt.Symbol("BLOCK_SIZE_M", meta=True)
    bn = nt.Symbol("BLOCK_SIZE_N", meta=True)

    def arrangement(x, y, out):
        return tuple(t.tile((bm, bn)) for t in (x, y, out))

    compiled = Compiler().compile(
        replace(
            _request(backend="tilelang"),
            arrangement=arrangement,
            tensors=tuple(nt.Tensor(2, dtype="float32") for _ in range(3)),
        )
    )
    instances = []

    def materialize(compilation, **kwargs):
        instances.append(compilation)

        return _FakeHandle(compilation)

    monkeypatch.setattr(
        materializers,
        "materializer_for",
        lambda target: SimpleNamespace(jit_materialize=materialize),
    )
    monkeypatch.setattr(
        runtime, "write_source", lambda *args, **kwargs: tmp_path / "lazy.py"
    )
    handle = runtime.materialize(compiled)
    values = tuple(
        SimpleNamespace(
            shape=(64, 96), stride=lambda dim: (96, 1)[dim], dtype="float32"
        )
        for _ in range(3)
    )
    assert handle(*values) is values[-1]
    assert handle(*values) is values[-1]
    assert len(instances) == 1
    assert not runtime._requires_runtime_specialization(instances[0])

    block = nt.Symbol("BLOCK_SIZE", constexpr=True)

    def legacy_arrangement(x, y, out):
        return tuple(t.tile((block,)) for t in (x, y, out))

    legacy = Compiler().compile(replace(_request(), arrangement=legacy_arrangement))
    binding = next(
        item for item in legacy.launch_abi.kernel_args if item.name == str(block)
    )
    assert runtime._binding_value(binding, {}) == 256
    assert str(block) not in legacy.request.frontend.meta_domains
