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
        assert "memory_scope" not in lowered.metadata
        assert "backend_intrinsics" not in lowered.metadata
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
        assert triton_specific == {"ssa.triton.optimize_schedule"}

    def test_each_backend_registers_required_contract_passes(self):
        for backend in Target:
            backend_passes = {
                descriptor.name
                for descriptor in registered(category=BACKEND_SPECIFIC, backend=backend)
            }
            assert backend_passes == {f"ssa.{backend.value}.optimize_schedule"}
            assert (
                f"ssa.{backend.value}.optimize_schedule" in default_spec(backend).passes
            )

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

    def test_default_pipeline_records_selected_passes(self):
        program = _program(
            "\ndef copy(x, out):\n    out = x\n",
            (
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
            ),
            "copy",
        )
        lowered = lower_for_target(program, backend=Target.TRITON)
        selection = lowered.metadata["pipeline_selection"]
        assert selection["mode"] == "default"
        assert "ssa.triton.optimize_schedule" in selection["selected_passes"]
        assert selection["reason"] == "default backend pipeline"

    def test_blocked_linalg_exposes_backend_schedule_candidates(self):
        program = _program(
            "\ndef matmul(a, b, out):\n    out = a @ b\n",
            (
                TensorSpec(ndim=2, shape=("m", "k"), dtype="float16", name="a"),
                TensorSpec(ndim=2, shape=("k", "n"), dtype="float16", name="b"),
                TensorSpec(ndim=2, shape=("m", "n"), dtype="float16", name="out"),
            ),
            "matmul",
        )

        for backend in Target:
            lowered = lower_for_target(program, backend=backend)
            candidates = lowered.metadata["schedule_candidates"]
            assert len(candidates) >= 3
            assert (
                lowered.metadata["selected_schedule_candidate"] == candidates[0]["name"]
            )
            assert (
                lowered.metadata["schedule"]["tile"]
                == candidates[0]["schedule"]["tile"]
            )

    def test_schedule_candidate_can_be_selected_by_pass_option(self):
        program = _program(
            "\ndef matmul(a, b, out):\n    out = a @ b\n",
            (
                TensorSpec(ndim=2, shape=("m", "k"), dtype="float16", name="a"),
                TensorSpec(ndim=2, shape=("k", "n"), dtype="float16", name="b"),
                TensorSpec(ndim=2, shape=("m", "n"), dtype="float16", name="out"),
            ),
            "matmul",
        )
        lowered = lower_for_target(
            program,
            backend=Target.TRITON,
            pass_options={"ssa.triton.optimize_schedule": {"candidate": "wide"}},
        )
        assert lowered.metadata["selected_schedule_candidate"] == "wide"
        assert lowered.metadata["schedule"]["tile"] == {
            "block_m": 64,
            "block_n": 64,
            "block_k": 32,
        }
        assert lowered.metadata["schedule"]["num_warps"] == 8
