from ninetoothed.backends.core import Target
from ninetoothed.compiler.passes import (
    BACKEND_SPECIFIC,
    HARDWARE_DEPENDENT,
    HARDWARE_INDEPENDENT,
    PipelineSpec,
    default_spec,
    lower_for_target,
    registered,
)
from ninetoothed.frontend.python import from_source
from ninetoothed.ir import TensorSpec


def _program(source: str, tensors: tuple[TensorSpec, ...], kind: str):
    program = from_source(source, tensors, kind=kind)
    assert program is not None

    return program


def _opcodes(operations):
    for operation in operations:
        yield operation.opcode

        for region in operation.regions:
            yield from _opcodes(region.operations)


def _operations(operations):
    for operation in operations:
        yield operation

        for region in operation.regions:
            yield from _operations(region.operations)


class TestPipeline:
    def test_pipeline_attaches_target_schedule_without_coarse_nodes(self):
        program = _program(
            "\ndef add(x, y, out):\n    out = x + y\n",
            (
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="y"),
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
            ),
            "add",
        )
        lowered = lower_for_target(
            program,
            backend=Target.CUDA,
            compiler_options={"num_warps": 4, "num_stages": 3},
        )
        assert tuple(lowered.metadata["pass_trace"]) == (
            "ssa.canonicalize",
            "ssa.analyze_effects",
            "ssa.select_schedule",
            "ssa.cuda.optimize_schedule",
            "ssa.decompose_linalg",
            "ssa.cuda.lower_memory_scopes",
            "ssa.cuda.lower_intrinsics",
        )
        assert lowered.metadata["target_backend"] == "cuda"
        assert lowered.metadata["schedule"]["granularity"] == "elementwise-grid"
        assert tuple(lowered.metadata["optimization"]["passes"]) == (
            "coalesced-linear-indexing",
        )
        assert (
            lowered.metadata["optimization"]["lowering"]
            == "ssa-operation-linear-emission"
        )
        forbidden = "tem" + "plate"
        assert forbidden not in str(lowered.metadata["optimization"]).lower()
        assert lowered.metadata["memory_scope"]["register"] == "thread-local"
        assert not lowered.metadata["coarse_operator_nodes"]
        opcodes = tuple(_opcodes(lowered.blocks[0].operations))
        assert "arith.add" in opcodes
        assert "mem.store" in opcodes

    def test_schedule_sees_linalg_before_decomposition(self):
        program = _program(
            "\ndef matmul(a, b, out):\n    out = a @ b\n",
            (
                TensorSpec(ndim=2, shape=("m", "k"), dtype="float32", name="a"),
                TensorSpec(ndim=2, shape=("k", "n"), dtype="float32", name="b"),
                TensorSpec(ndim=2, shape=("m", "n"), dtype="float32", name="out"),
            ),
            "matmul",
        )
        lowered = lower_for_target(program, backend=Target.CUDA)
        opcodes = tuple(_opcodes(lowered.blocks[0].operations))
        assert lowered.metadata["analysis"]["has_dot"]
        assert lowered.metadata["schedule"]["granularity"] == "blocked-linalg"
        assert lowered.metadata["linalg_decomposed"]
        assert "linalg.matmul" not in opcodes
        assert "scf.for" in opcodes
        assert "tensor.extract" in opcodes
        assert "arith.mul" in opcodes
        assert "arith.add" in opcodes

    def test_backend_specific_intrinsics_are_annotations_not_semantic_ops(self):
        program = _program(
            "\ndef copy(x, out):\n    out = x\n",
            (
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
            ),
            "copy",
        )

        for backend, expected_program_id in (
            (Target.TRITON, "tl.program_id"),
            (Target.TILELANG, "T.Kernel + T.get_thread_binding"),
            (Target.TVM, "T.thread_binding"),
        ):
            lowered = lower_for_target(program, backend=backend)
            assert (
                lowered.metadata["backend_intrinsics"]["program_id"]
                == expected_program_id
            )

            for operation in _operations(lowered.blocks[0].operations):
                assert "backend_intrinsic" in operation.attrs
                assert "optimization" in operation.attrs

    def test_pass_registry_classifies_hardware_independent_and_target_passes(self):
        independent = {
            descriptor.name for descriptor in registered(category=HARDWARE_INDEPENDENT)
        }
        dependent = {
            descriptor.name for descriptor in registered(category=HARDWARE_DEPENDENT)
        }
        triton_specific = {
            descriptor.name
            for descriptor in registered(
                category=BACKEND_SPECIFIC, backend=Target.TRITON
            )
        }
        assert "ssa.canonicalize" in independent
        assert "ssa.decompose_linalg" in independent
        assert "ssa.analyze_effects" in independent
        assert "ssa.select_schedule" in independent
        assert not dependent
        assert "ssa.triton.optimize_schedule" in triton_specific
        assert "ssa.triton.lower_memory_scopes" in triton_specific
        assert "ssa.triton.lower_intrinsics" in triton_specific
        assert "ssa.cuda.optimize_schedule" not in triton_specific
        assert "ssa.cuda.lower_intrinsics" not in triton_specific

    def test_each_backend_registers_required_contract_passes(self):
        for backend in Target:
            backend_passes = {
                descriptor.name
                for descriptor in registered(category=BACKEND_SPECIFIC, backend=backend)
            }
            required = {
                f"ssa.{backend.value}.optimize_schedule",
                f"ssa.{backend.value}.lower_memory_scopes",
                f"ssa.{backend.value}.lower_intrinsics",
            }
            assert required <= backend_passes
            assert required <= set(default_spec(backend).passes)

    def test_custom_pipeline_can_disable_backend_optimization_pass(self):
        program = _program(
            "\ndef copy(x, out):\n    out = x\n",
            (
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
            ),
            "copy",
        )
        lowered = lower_for_target(
            program,
            backend=Target.TRITON,
            pass_pipeline=PipelineSpec(
                passes=(
                    "ssa.canonicalize",
                    "ssa.decompose_linalg",
                    "ssa.analyze_effects",
                    "ssa.select_schedule",
                    "ssa.triton.lower_memory_scopes",
                    "ssa.triton.lower_intrinsics",
                ),
                mode="custom",
                reason="test pipeline without backend optimization",
            ),
        )
        assert "ssa.triton.optimize_schedule" not in lowered.metadata["pass_trace"]
        assert "optimization" not in lowered.metadata
        assert lowered.metadata["pipeline_selection"]["mode"] == "custom"
        assert lowered.metadata["pipeline_selection"]["categories"][
            HARDWARE_INDEPENDENT
        ] == (
            "ssa.canonicalize",
            "ssa.decompose_linalg",
            "ssa.analyze_effects",
            "ssa.select_schedule",
        )

    def test_autotune_pipeline_records_candidates_and_selected_passes(self):
        program = _program(
            "\ndef copy(x, out):\n    out = x\n",
            (
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
            ),
            "copy",
        )
        lowered = lower_for_target(program, backend=Target.TRITON, autotune=True)
        selection = lowered.metadata["pipeline_selection"]
        assert selection["mode"] == "autotune"
        assert "ssa.triton.optimize_schedule" in selection["selected_passes"]
        assert selection["candidate_pipelines"]
        assert "policy-autotune" in selection["reason"]
