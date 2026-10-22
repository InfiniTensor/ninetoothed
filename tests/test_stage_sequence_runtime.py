"""Guarded native partitions retain per-call owners and the original fallback."""

from contextlib import nullcontext

import torch

from ninetoothed.backends.materializers import staged
from ninetoothed.compiler import runtime
from ninetoothed.compiler.dot_materialization import plan_ordered_dot_materialization
from tests.test_cuda_wmma_grouping import _compile


class _Argument:
    shape = (64, 64)
    device = "cuda:0"

    def __init__(self, pointer=1024, width=4):
        self.pointer = pointer
        self.width = width
        self.strides = (64, 1)
        self.recorded_streams = []

    def stride(self):
        return self.strides

    def data_ptr(self):
        return self.pointer

    def element_size(self):
        return self.width

    def record_stream(self, stream):
        self.recorded_streams.append(stream)


def test_stage_owners_are_fresh_and_guard_failure_uses_whole_original_call(
    monkeypatch, tmp_path
):
    compilation = _compile()
    plan = plan_ordered_dot_materialization(
        compilation.kernel.ssa, compilation.kernel.tensors
    )
    _, stages = staged.stage_compilations_from_plan(
        compilation, plan, scratch_specs=plan.scratch_specs
    )
    calls = []
    allocations = []
    streams = [object(), object()]
    active = [0]

    class Handle:
        def __init__(self, compilation, handles, launch, source):
            self._launch = launch
            self._artifact = compilation.artifact

        def __call__(self, *args, **kwargs):
            return self._launch(*args, **kwargs)

    def allocate(shape, *, dtype, device):
        owner = _Argument()
        allocations.append((owner, shape, dtype, device))

        return owner

    def materialize(stage, *, output_dir):
        def invoke(*arguments):
            calls.append((stage.kernel.kernel_name, arguments))

        return invoke

    def fallback(*args, **kwargs):
        calls.append(("original", args, kwargs))

        return "original-result"

    fallback._artifact = compilation.artifact

    monkeypatch.setattr(runtime, "Handle", Handle)
    monkeypatch.setattr(
        runtime, "_runtime_wrapper", lambda function, *args, **kwargs: function
    )
    monkeypatch.setattr(runtime, "_verified_runtime_launch", lambda function: function)
    monkeypatch.setattr(
        staged, "write_source", lambda *args, **kwargs: tmp_path / "public.cu"
    )
    monkeypatch.setattr(torch, "empty", allocate)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: streams[active[0]])
    handle = staged.materialize_stage_sequence(
        compilation,
        plan,
        stages,
        stage_materializer=materialize,
        device_type="cuda",
        argument_guard=plan.runtime_eligible,
        fallback_handle=fallback,
    )
    x = _Argument(0x10000, 2)
    y = _Argument(0x40000, 2)
    out = _Argument(0x80000, 4)
    handle(x, y, out)
    first = allocations[0][0]
    assert allocations[0][1:] == ((128, 64), torch.float32, "cuda:0")
    assert calls[0][1] == (x, y, first)
    assert calls[1][1] == (first, out)
    assert first.recorded_streams == [streams[0]]
    active[0] = 1
    handle(x=x, y=y, out=out)
    second = allocations[1][0]
    assert second is not first and second.recorded_streams == [streams[1]]
    assert calls[2][1] == (x, y, second) and calls[3][1] == (second, out)
    x.strides = (128, 2)
    assert handle(x=x, y=y, out=out) == "original-result"
    assert len(allocations) == 2 and len(calls) == 5
    assert calls[-1] == ("original", (), {"x": x, "y": y, "out": out})
    assert handle.stage_artifacts == tuple(stage.artifact for stage in stages)
    assert handle._artifact.metadata["whole_call_guard"]
    assert (
        handle._artifact.metadata["primary_source_role"] == "whole-call-guard-fallback"
    )
    assert handle.execution_artifacts_for(x, y, out) == (compilation.artifact,)
    x.strides = (64, 1)
    assert handle.execution_artifacts_for(x, y, out) == handle.stage_artifacts
    out.pointer = x.pointer
    assert handle(x, y, out) == "original-result"
    assert len(allocations) == 2 and len(calls) == 6
    assert calls[-1] == ("original", (x, y, out), {})
    assert handle.execution_artifacts_for(x, y, out) == (compilation.artifact,)
