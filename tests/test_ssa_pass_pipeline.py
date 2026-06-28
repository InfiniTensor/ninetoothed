import unittest

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


class SSAPassPipelineTest(unittest.TestCase):
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

        self.assertEqual(
            tuple(lowered.metadata["pass_trace"]),
            (
                "ssa.canonicalize",
                "ssa.analyze_effects",
                "ssa.select_schedule",
                "ssa.cuda.optimize_schedule",
                "ssa.decompose_linalg",
                "ssa.lower_memory_scopes",
                "ssa.lower_backend_intrinsics",
            ),
        )
        self.assertEqual(lowered.metadata["target_backend"], "cuda")
        self.assertEqual(
            lowered.metadata["schedule"]["granularity"], "elementwise-grid"
        )
        self.assertEqual(
            tuple(lowered.metadata["optimization"]["passes"]),
            ("coalesced-linear-indexing",),
        )
        self.assertEqual(
            lowered.metadata["optimization"]["lowering"],
            "ssa-operation-linear-emission",
        )
        forbidden = "tem" + "plate"
        self.assertNotIn(forbidden, str(lowered.metadata["optimization"]).lower())
        self.assertEqual(lowered.metadata["memory_scope"]["register"], "thread-local")
        self.assertFalse(lowered.metadata["coarse_operator_nodes"])

        opcodes = tuple(_opcodes(lowered.blocks[0].operations))
        self.assertIn("arith.add", opcodes)
        self.assertIn("mem.store", opcodes)
        self.assertNotIn("AttentionOpIR", opcodes)
        self.assertNotIn("FlashAttentionOpIR", opcodes)

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

        self.assertTrue(lowered.metadata["analysis"]["has_dot"])
        self.assertEqual(lowered.metadata["schedule"]["granularity"], "blocked-linalg")
        self.assertTrue(lowered.metadata["linalg_decomposed"])
        self.assertNotIn("linalg.matmul", opcodes)
        self.assertIn("scf.for", opcodes)
        self.assertIn("tensor.extract", opcodes)
        self.assertIn("arith.mul", opcodes)
        self.assertIn("arith.add", opcodes)

    def test_backend_specific_intrinsics_are_annotations_not_semantic_ops(self):
        program = ProgramIR(
            kind="elementwise",
            operations=(
                ElementwiseAssignOpIR(
                    output="out",
                    expression=ExprIR(kind="var", value="x"),
                    extent="n",
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
            with self.subTest(backend=backend.value):
                lowered = lower_ssa_for_backend(
                    program_to_ssa(program, tensors), backend=backend
                )
                self.assertEqual(
                    lowered.metadata["backend_intrinsics"]["program_id"],
                    expected_program_id,
                )
                for operation in _operations(lowered.blocks[0].operations):
                    self.assertIn("backend_intrinsic", operation.attrs)
                    self.assertIn("optimization", operation.attrs)
                    self.assertNotIn("AttentionOpIR", operation.opcode)

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

        self.assertIn("ssa.canonicalize", independent)
        self.assertIn("ssa.decompose_linalg", independent)
        self.assertIn("ssa.analyze_effects", independent)
        self.assertIn("ssa.select_schedule", dependent)
        self.assertIn("ssa.lower_backend_intrinsics", dependent)
        self.assertIn("ssa.triton.optimize_schedule", triton_specific)
        self.assertNotIn("ssa.cuda.optimize_schedule", triton_specific)

    def test_custom_pipeline_can_disable_backend_optimization_pass(self):
        program = ProgramIR(
            kind="elementwise",
            operations=(
                ElementwiseAssignOpIR(
                    output="out",
                    expression=ExprIR(kind="var", value="x"),
                    extent="n",
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

        self.assertNotIn("ssa.triton.optimize_schedule", lowered.metadata["pass_trace"])
        self.assertNotIn("optimization", lowered.metadata)
        self.assertEqual(lowered.metadata["pipeline_selection"]["mode"], "custom")
        self.assertEqual(
            lowered.metadata["pipeline_selection"]["categories"][HARDWARE_INDEPENDENT],
            ("ssa.canonicalize", "ssa.decompose_linalg", "ssa.analyze_effects"),
        )

    def test_autotune_pipeline_records_candidates_and_selected_passes(self):
        program = ProgramIR(
            kind="elementwise",
            operations=(
                ElementwiseAssignOpIR(
                    output="out",
                    expression=ExprIR(kind="var", value="x"),
                    extent="n",
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
            autotune=True,
        )

        selection = lowered.metadata["pipeline_selection"]
        self.assertEqual(selection["mode"], "autotune")
        self.assertIn("ssa.triton.optimize_schedule", selection["selected_passes"])
        self.assertTrue(selection["candidate_pipelines"])
        self.assertIn("policy-autotune", selection["reason"])


if __name__ == "__main__":
    unittest.main()
