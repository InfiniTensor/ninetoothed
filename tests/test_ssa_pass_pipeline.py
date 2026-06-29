from ninetoothed.backends.base import BackendName
from ninetoothed.ir import (
    ElementwiseAssignOpIR,
    ExprIR,
    MatmulOpIR,
    ProgramIR,
    TensorTypeIR,
    program_to_ssa,
)
from ninetoothed.ssa_passes import (
    BACKEND_SPECIFIC,
    HARDWARE_DEPENDENT,
    HARDWARE_INDEPENDENT,
    SSAPipelineSpec,
    lower_ssa_for_backend,
    registered_ssa_passes,
)


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


class TestSSAPassPipeline:
    def test_pipeline_attaches_target_schedule_without_coarse_nodes(self):
        program = ProgramIR(
            kind="elementwise",
            operations=(
                ElementwiseAssignOpIR(
                    output="out",
                    expression=ExprIR(
                        kind="binary",
                        value="add",
                        args=(
                            ExprIR(kind="var", value="x"),
                            ExprIR(kind="var", value="y"),
                        ),
                    ),
                    extent="n",
                ),
            ),
        )
        tensors = (
            TensorTypeIR("x", 1, "float32", ("n",)),
            TensorTypeIR("y", 1, "float32", ("n",)),
            TensorTypeIR("out", 1, "float32", ("n",)),
        )
        generic_ssa = program_to_ssa(program, tensors)
        lowered = lower_ssa_for_backend(
            generic_ssa,
            backend=BackendName.CUDA,
            compiler_options={"num_warps": 4, "num_stages": 3},
        )
        assert tuple(lowered.metadata["pass_trace"]) == (
            "ssa.canonicalize",
            "ssa.analyze_effects",
            "ssa.select_schedule",
            "ssa.cuda.optimize_schedule",
            "ssa.decompose_linalg",
            "ssa.lower_memory_scopes",
            "ssa.lower_backend_intrinsics",
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
        assert "AttentionOpIR" not in opcodes
        assert "FlashAttentionOpIR" not in opcodes

    def test_schedule_sees_linalg_before_decomposition(self):
        program = ProgramIR(
            kind="matmul",
            operations=(
                MatmulOpIR(lhs="a", rhs="b", output="out", m="m", n="n", k="k"),
            ),
        )
        tensors = (
            TensorTypeIR("a", 2, "float32", ("m", "k")),
            TensorTypeIR("b", 2, "float32", ("k", "n")),
            TensorTypeIR("out", 2, "float32", ("m", "n")),
        )
        lowered = lower_ssa_for_backend(
            program_to_ssa(program, tensors), backend=BackendName.CUDA
        )
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
        program = ProgramIR(
            kind="elementwise",
            operations=(
                ElementwiseAssignOpIR(
                    output="out", expression=ExprIR(kind="var", value="x"), extent="n"
                ),
            ),
        )
        tensors = (
            TensorTypeIR("x", 1, "float32", ("n",)),
            TensorTypeIR("out", 1, "float32", ("n",)),
        )
        for backend, expected_program_id in (
            (BackendName.TRITON, "tl.program_id"),
            (BackendName.TILELANG, "T.Kernel + T.get_thread_binding"),
            (BackendName.TVM, "T.thread_binding"),
        ):
            lowered = lower_ssa_for_backend(
                program_to_ssa(program, tensors), backend=backend
            )
            assert (
                lowered.metadata["backend_intrinsics"]["program_id"]
                == expected_program_id
            )
            for operation in _operations(lowered.blocks[0].operations):
                assert "backend_intrinsic" in operation.attrs
                assert "optimization" in operation.attrs
                assert "AttentionOpIR" not in operation.opcode

    def test_pass_registry_classifies_hardware_independent_and_target_passes(self):
        independent = {
            descriptor.name
            for descriptor in registered_ssa_passes(category=HARDWARE_INDEPENDENT)
        }
        dependent = {
            descriptor.name
            for descriptor in registered_ssa_passes(category=HARDWARE_DEPENDENT)
        }
        triton_specific = {
            descriptor.name
            for descriptor in registered_ssa_passes(
                category=BACKEND_SPECIFIC, backend=BackendName.TRITON
            )
        }
        assert "ssa.canonicalize" in independent
        assert "ssa.decompose_linalg" in independent
        assert "ssa.analyze_effects" in independent
        assert "ssa.select_schedule" in dependent
        assert "ssa.lower_backend_intrinsics" in dependent
        assert "ssa.triton.optimize_schedule" in triton_specific
        assert "ssa.cuda.optimize_schedule" not in triton_specific

    def test_custom_pipeline_can_disable_backend_optimization_pass(self):
        program = ProgramIR(
            kind="elementwise",
            operations=(
                ElementwiseAssignOpIR(
                    output="out", expression=ExprIR(kind="var", value="x"), extent="n"
                ),
            ),
        )
        tensors = (
            TensorTypeIR("x", 1, "float32", ("n",)),
            TensorTypeIR("out", 1, "float32", ("n",)),
        )
        lowered = lower_ssa_for_backend(
            program_to_ssa(program, tensors),
            backend=BackendName.TRITON,
            pass_pipeline=SSAPipelineSpec(
                passes=(
                    "ssa.canonicalize",
                    "ssa.decompose_linalg",
                    "ssa.analyze_effects",
                    "ssa.select_schedule",
                    "ssa.lower_memory_scopes",
                    "ssa.lower_backend_intrinsics",
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
        ] == ("ssa.canonicalize", "ssa.decompose_linalg", "ssa.analyze_effects")

    def test_autotune_pipeline_records_candidates_and_selected_passes(self):
        program = ProgramIR(
            kind="elementwise",
            operations=(
                ElementwiseAssignOpIR(
                    output="out", expression=ExprIR(kind="var", value="x"), extent="n"
                ),
            ),
        )
        tensors = (
            TensorTypeIR("x", 1, "float32", ("n",)),
            TensorTypeIR("out", 1, "float32", ("n",)),
        )
        lowered = lower_ssa_for_backend(
            program_to_ssa(program, tensors), backend=BackendName.TRITON, autotune=True
        )
        selection = lowered.metadata["pipeline_selection"]
        assert selection["mode"] == "autotune"
        assert "ssa.triton.optimize_schedule" in selection["selected_passes"]
        assert selection["candidate_pipelines"]
        assert "policy-autotune" in selection["reason"]
