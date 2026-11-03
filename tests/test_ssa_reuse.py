"""Core contracts and the first real consumer of shared SSA reuse passes."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest

import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest
from ninetoothed.compiler.cse import eliminate_common_subexpressions
from ninetoothed.compiler.materialization import plan_materialization, retain_bound_plan
from ninetoothed.compiler.passes import default_spec, lower_for_target
from ninetoothed.compiler.unroll import unroll_static_elementwise_loops
from ninetoothed.frontend.python import from_source
from ninetoothed.ir import TensorSpec, ssa


def _program(source, names=("x", "out")):
    return from_source(
        source,
        tuple(
            TensorSpec(name=name, ndim=2, shape=(8, 8), dtype="float32")
            for name in names
        ),
        kind="reuse_case",
    )


def _arrangement(*tensors):
    return tuple(tensor.tile((-1, -1)) for tensor in tensors)


def _application(a, b, c, out):
    out = ((a @ b) + (a @ b)) @ c  # noqa: F841


def _row_arrangement(x, out):
    return x.tile((1, -1)), out.tile((1,))


def _repeated_reduction(x, out):
    out = ntl.sum(x * x, axis=1) + ntl.sum(x * x, axis=1)  # noqa: F841


def _broadcast_row_arrangement(x, out):
    return x.tile((1, -1)), out.tile((1, -1))


def _normalize_row(x, out):
    out = x / ntl.sqrt(ntl.sum(x * x, axis=1)[:, None] / x.shape[-1] + 1e-5)  # noqa: F841


def _compilation(size=32, **options):
    return DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_arrangement,
            application=_application,
            tensors=tuple(
                nt.Tensor(shape=(size, size), dtype=nt.float32) for _ in range(4)
            ),
            backend="cuda",
            max_num_configs=1,
            **options,
        )
    )


def test_cse_preserves_types_effects_and_signed_zero():
    program = _program("def app(x, out):\n    out = (x + 1.0) * (x + 1.0)\n")
    reduced, aliases = eliminate_common_subexpressions(program)
    ssa.verify_program(reduced)
    assert aliases
    assert (
        len([op for op in reduced.blocks[0].operations if op.opcode == "arith.add"])
        == 1
    )
    assert eliminate_common_subexpressions(reduced) == (reduced, ())
    second = next(old for old, new in aliases if old != new)
    output = program.outputs[0]
    typed = replace(
        program,
        outputs=(
            replace(output, type=replace(output.type, attrs={"dtype_ref": second})),
        ),
    )
    typed, bindings = eliminate_common_subexpressions(typed)
    assert typed.outputs[0].type.attrs["dtype_ref"] == dict(bindings)[second]
    program = _program("def app(x, out):\n    out = x + 1.0\n    out = x + 1.0\n")
    reduced, _ = eliminate_common_subexpressions(program)
    assert (
        len([op for op in reduced.blocks[0].operations if op.opcode == "arith.add"])
        == 2
    )
    program = _program("def app(x, out):\n    out = (x + 0.0) * (x + -0.0)\n")
    reduced, _ = eliminate_common_subexpressions(program)
    values = [
        op.attrs["value"]
        for op in reduced.blocks[0].operations
        if op.opcode == "arith.constant"
    ]
    assert any(str(x) == "-0.0" for x in values)

    for backend in ("bangc", "ascendc"):
        compilation = DEFAULT_COMPILER.compile(
            CompileRequest(
                arrangement=_row_arrangement,
                application=_repeated_reduction,
                tensors=(
                    nt.Tensor(shape=(32, 129), dtype=nt.float32),
                    nt.Tensor(shape=(32,), dtype=nt.float32),
                ),
                backend=backend,
                max_num_configs=1,
            )
        )
        assert "nt_tile_buffer_" in compilation.artifact.primary_source

        if backend == "ascendc":
            bf16 = DEFAULT_COMPILER.compile(
                CompileRequest(
                    arrangement=_broadcast_row_arrangement,
                    application=_normalize_row,
                    tensors=tuple(
                        nt.Tensor(shape=(7, 129), dtype=nt.bfloat16) for _ in range(2)
                    ),
                    backend=backend,
                    max_num_configs=1,
                )
            )
            assert "nt_tile_buffer_" not in bf16.artifact.primary_source

        compilation = DEFAULT_COMPILER.compile(
            replace(
                compilation.request,
                frontend=None,
                arrangement=_broadcast_row_arrangement,
                application=_normalize_row,
                tensors=tuple(
                    nt.Tensor(shape=(64, 513), dtype=nt.float32) for _ in range(2)
                ),
            )
        )
        assert "nt_tile_buffer_" in compilation.artifact.primary_source


def test_static_unroll_preserves_order_and_bounds():
    for iterations in (0, 4, 9):
        program = _program(
            f"def app(x, out):\n    value = x\n    for i in range({iterations}):\n        value = value * 0.25 + x\n    out = value\n"
        )
        loop = next(op for op in program.blocks[0].operations if op.opcode == "scf.for")
        output = program.outputs[0]
        program = replace(
            program,
            outputs=(
                replace(
                    output,
                    type=replace(
                        output.type, attrs={"dtype_ref": loop.results[0].name}
                    ),
                ),
            ),
        )
        result, expanded = unroll_static_elementwise_loops(program)
        ssa.verify_program(result)
        codes = [op.opcode for op in result.blocks[0].operations]
        assert ("scf.for" in codes) == (iterations > 8)

        if iterations <= 8:
            assert result.outputs[0].type.attrs["dtype_ref"] != loop.results[0].name
            assert expanded and expanded[0][1] == iterations
            assert [code for code in codes if code in {"arith.mul", "arith.add"}] == [
                "arith.mul",
                "arith.add",
            ] * iterations


def test_materialization_snapshots_stage_abi_and_aot_contract():
    from ninetoothed.backends.materializers.staged import (
        aot_monolithic_compilation,
        stage_compilations,
    )

    program = _program(
        "def app(a, b, c, out):\n    value = a @ b\n    out = (value + value) @ c\n",
        ("a", "b", "c", "out"),
    )
    plan = plan_materialization(program)
    assert plan is not None and len(plan.stages) == 2

    for stage in plan.stages:
        ssa.verify_program(stage.program)

    stale = replace(
        program,
        plans={"materialization": plan, "unrelated": "kept"},
        blocks=(
            replace(program.blocks[0], operations=program.blocks[0].operations[:-1]),
        ),
    )
    assert retain_bound_plan(stale).plans == {"unrelated": "kept"}

    with pytest.raises(ValueError, match="no longer matches"):
        retain_bound_plan(
            replace(stale, metadata={"materialization": {"explicitly_enabled": True}})
        )

    zero = _program(
        "def app(a, b, c, out):\n    value = a @ b\n    out = ((value + value) * 0.0) @ c\n",
        ("a", "b", "c", "out"),
    )
    zero_plan = plan_materialization(zero)
    assert zero_plan is not None
    operations = tuple(
        replace(op, attrs={**op.attrs, "value": -0.0})
        if op.opcode == "arith.constant" and op.attrs.get("value") == 0.0
        else op
        for op in zero.blocks[0].operations
    )
    zero = replace(
        zero,
        plans={"materialization": zero_plan},
        blocks=(replace(zero.blocks[0], operations=operations),),
        metadata={"materialization": {"explicitly_enabled": True}},
    )

    with pytest.raises(ValueError, match="no longer matches"):
        retain_bound_plan(zero)

    compilation = _compilation()
    assert not _compilation(size=8).artifact.execution_plans.get("materialization")
    plan, stages = stage_compilations(compilation)
    assert len(stages) == 2
    assert all(
        not stage.artifact.execution_plans.get("materialization") for stage in stages
    )
    assert plan.scratch[0].name in stages[0].launch_abi.outputs
    assert not aot_monolithic_compilation(compilation).artifact.execution_plans.get(
        "materialization"
    )
    explicit = _compilation(
        pass_options={"ssa.plan_materialization": {"enabled": True}}
    )

    with pytest.raises(ValueError, match="JIT only"):
        aot_monolithic_compilation(explicit)

    disabled = lower_for_target(
        program,
        backend="cuda",
        pass_options={
            "ssa.cse": {"enabled": False},
            "ssa.plan_materialization": {"enabled": False},
        },
    )
    assert not disabled.metadata["cse"]["enabled"]
    assert not disabled.plans.get("materialization")
    custom = tuple(name for name in default_spec("cuda").passes if name != "ssa.cse")

    with pytest.raises(ValueError, match="missing from the pipeline"):
        lower_for_target(
            program,
            backend="cuda",
            pass_pipeline=custom,
            pass_options={"ssa.cse": {"enabled": True}},
        )

    with pytest.raises(ValueError, match="boolean"):
        lower_for_target(
            program, backend="cuda", pass_options={"ssa.cse": {"enabled": 1}}
        )


def test_stage_execution_retains_owners_on_failure_and_falls_back_on_alias(
    monkeypatch, tmp_path
):
    import torch

    from ninetoothed.backends.materializers import staged
    from ninetoothed.compiler import runtime

    compilation = _compilation()
    plan, stages = staged.stage_compilations(compilation)
    allocated = []
    recorded = []
    calls = []
    state = {"fail": True, "record_fail": False}
    launch_error = RuntimeError("Second stage failed")
    stream = SimpleNamespace(synchronize=lambda: calls.append("synchronize"))

    class Tensor:
        device = SimpleNamespace(type="cuda", index=0)

        def __init__(self, identity):
            self.identity = identity

        def untyped_storage(self):
            return SimpleNamespace(data_ptr=lambda: self.identity, nbytes=lambda: 1)

        def record_stream(self, current):
            assert current is stream
            recorded.append(self.identity)

            if state["record_fail"]:
                raise RuntimeError("Record failed.")

    def empty(*args, **kwargs):
        owner = Tensor(100 + len(allocated))
        allocated.append(owner)

        return owner

    def materialize(stage, **kwargs):
        index = next((i for i, value in enumerate(stages) if value is stage), None)

        def call(*args):
            calls.append(index)

            if index == 1 and state["fail"]:
                raise launch_error
            return "fallback" if index is None else None

        call._artifact = stage.artifact

        return call

    monkeypatch.setattr(staged, "stage_compilations", lambda _: (plan, stages))
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _: stream)
    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(runtime, "_runtime_wrapper", lambda function, *a, **k: function)
    monkeypatch.setattr(runtime, "_verified_runtime_launch", lambda function: function)
    monkeypatch.setattr(
        runtime,
        "Handle",
        lambda comp, handles, launch, source: SimpleNamespace(
            _artifact=comp.artifact, _launch=launch
        ),
    )
    monkeypatch.setattr(
        staged, "write_source", lambda *a, **k: tmp_path / "reference.cu"
    )
    monkeypatch.setattr(staged, "compilation_cache_key", lambda _: "test")
    handle = staged.materialize_staged(compilation, stage_materializer=materialize)
    arguments = [Tensor(i) for i in range(4)]

    with pytest.raises(RuntimeError, match="Second stage failed") as failure:
        handle._launch(*arguments)

    assert failure.value is launch_error
    assert calls == [0, 1]
    assert set(recorded) == {0, 1, 2, 3, 100}
    state["record_fail"] = True

    with pytest.raises(RuntimeError) as failure:
        handle._launch(*arguments)

    assert failure.value is launch_error and calls[-1] == "synchronize"
    assert len({owner.identity for owner in allocated}) == 2
    arguments[-1] = arguments[0]
    assert handle._launch(*arguments) == "fallback"
    assert len(handle.execution_artifacts_for(*arguments)) == 1


def test_cuda_reuse_pipeline_numerics_stream_and_aot(tmp_path):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")

    for dtype in (torch.float32, torch.float16):
        host = [torch.randn(32, 32, dtype=dtype) / 32 for _ in range(3)]
        first = (host[0].float() @ host[1].float()).to(dtype)
        expected = ((first + first).float() @ host[2].float()).to(dtype)
        arguments = [value.cuda() for value in host]
        output = torch.empty_like(arguments[0])
        compilation = DEFAULT_COMPILER.compile(
            CompileRequest(
                arrangement=_arrangement,
                application=_application,
                tensors=tuple(
                    nt.Tensor(shape=(32, 32), dtype=str(dtype)) for _ in range(4)
                ),
                backend="cuda",
                max_num_configs=1,
            )
        )
        handle = DEFAULT_COMPILER.materialize(compilation)
        assert len(handle.stage_artifacts) == 2
        storage = torch.empty(32 * 32 + 1, device="cuda", dtype=dtype)
        aliased_input = torch.from_dlpack(storage[:-1]).view(32, 32)
        aliased_output = torch.from_dlpack(storage[1:]).view(32, 32)
        assert (
            aliased_input.untyped_storage().data_ptr()
            != aliased_output.untyped_storage().data_ptr()
        )
        assert (
            len(
                handle.execution_artifacts_for(
                    aliased_input, arguments[1], arguments[2], aliased_output
                )
            )
            == 1
        )

        with torch.cuda.stream(torch.cuda.Stream()):
            handle(*arguments, output)

        torch.cuda.synchronize()
        torch.testing.assert_close(
            output.cpu(),
            expected,
            rtol=2e-3 if dtype == torch.float16 else 2e-5,
            atol=2e-5 if dtype == torch.float16 else 2e-7,
        )
        disabled = DEFAULT_COMPILER.compile(
            replace(
                compilation.request,
                pass_options={
                    "ssa.cse": {"enabled": False},
                    "ssa.unroll_static_elementwise": {"enabled": False},
                    "ssa.plan_materialization": {"enabled": False},
                },
            )
        )
        baseline = DEFAULT_COMPILER.materialize(disabled)
        baseline(*arguments, output)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output.cpu(),
            expected,
            rtol=2e-3 if dtype == torch.float16 else 2e-5,
            atol=2e-5 if dtype == torch.float16 else 2e-7,
        )
        built = DEFAULT_COMPILER.materialize(
            compilation, mode="aot", output_dir=tmp_path / str(dtype)
        )
        from ninetoothed.compiler import load_built_artifact

        load_built_artifact(built._built_artifact)(*arguments, output)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            output.cpu(),
            expected,
            rtol=2e-3 if dtype == torch.float16 else 2e-5,
            atol=2e-5 if dtype == torch.float16 else 2e-7,
        )
