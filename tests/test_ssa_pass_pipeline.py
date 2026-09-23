from dataclasses import replace

import pytest

from ninetoothed.backends.core import Target
from ninetoothed.compiler.passes import (
    BACKEND_SPECIFIC,
    HARDWARE_INDEPENDENT,
    LANGUAGE_SPECIFIC,
    PLATFORM_SPECIFIC,
    Pass,
    Pipeline,
    PipelineSpec,
    create_default_registry,
    default_spec,
    lower_for_target,
    registered,
)
from ninetoothed.frontend.python import from_source
from ninetoothed.ir import TensorSpec
from ninetoothed.targets import PlatformProfile, TargetContext


def _program(source: str, tensors: tuple[TensorSpec, ...], kind: str):
    program = from_source(source, tensors, kind=kind)
    assert program is not None

    return program


def _copy_program():
    return _program(
        "\ndef copy(x, out):\n    out = x\n",
        (
            TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
            TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
        ),
        "copy",
    )


def _with_required_capabilities(program, *capabilities):
    return replace(
        program,
        metadata=dict(program.metadata)
        | {"required_capabilities": tuple(capabilities)},
    )


def _target_context(*, supported=(), unsupported=()):
    return TargetContext(
        backend=Target.TRITON,
        platform=PlatformProfile(
            name="test-platform",
            backend_modes={"triton": frozenset({"jit"})},
            supported_capabilities=frozenset(supported),
            unsupported_capabilities=frozenset(unsupported),
        ),
    )


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
            "ssa.validate_target_capabilities",
        )
        assert lowered.metadata["target_backend"] == "cuda"
        assert lowered.metadata["target_platform"] == "generic"
        assert lowered.metadata["schedule"]["granularity"] == "elementwise-grid"
        assert not lowered.metadata["optimization"]
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

    def test_pass_registry_classifies_and_registers_backend_contracts(self):
        independent = {
            descriptor.name for descriptor in registered(category=HARDWARE_INDEPENDENT)
        }
        assert "ssa.canonicalize" in independent
        assert "ssa.decompose_linalg" in independent
        assert "ssa.analyze_effects" in independent
        assert "ssa.select_schedule" in independent

        for backend in Target:
            backend_passes = {
                descriptor.name
                for descriptor in registered(category=BACKEND_SPECIFIC, backend=backend)
            }
            assert backend_passes == {f"ssa.{backend.value}.optimize_schedule"}
            assert (
                f"ssa.{backend.value}.optimize_schedule" in default_spec(backend).passes
            )

        assert {
            descriptor.name for descriptor in registered(category="backend_specific")
        } == {
            "ssa.triton.optimize_schedule",
            "ssa.cuda.optimize_schedule",
            "ssa.tilelang.optimize_schedule",
        }

        platform_passes = {
            descriptor.name for descriptor in registered(category=PLATFORM_SPECIFIC)
        }
        assert platform_passes == {"ssa.validate_target_capabilities"}

    def test_legacy_backend_specific_custom_pass_is_normalized(self):
        class LegacyBackendPass(Pass):
            name = "test.legacy_backend_pass"
            category = BACKEND_SPECIFIC

            def run(self, program, context):
                del context

                return type(program)(
                    kind=program.kind,
                    inputs=program.inputs,
                    outputs=program.outputs,
                    blocks=program.blocks,
                    metadata=dict(program.metadata) | {"legacy_pass_ran": True},
                )

        registry = create_default_registry()
        registry.register(LegacyBackendPass)

        legacy_names = registry.names(category=BACKEND_SPECIFIC)
        language_names = registry.names(category=LANGUAGE_SPECIFIC)

        assert legacy_names == language_names
        assert "test.legacy_backend_pass" in legacy_names

        program = _copy_program()

        lowered = lower_for_target(
            program,
            backend=Target.TRITON,
            pass_pipeline=PipelineSpec(
                passes=("test.legacy_backend_pass",),
                mode="custom",
            ),
            pass_registry=registry,
        )
        categories = lowered.metadata["pipeline_selection"]["categories"]

        assert lowered.metadata["legacy_pass_ran"]
        assert lowered.metadata["pass_trace"] == (
            "test.legacy_backend_pass",
            "ssa.validate_target_capabilities",
        )
        assert categories[LANGUAGE_SPECIFIC] == ("test.legacy_backend_pass",)
        assert categories[BACKEND_SPECIFIC] == categories[LANGUAGE_SPECIFIC]

    def test_custom_pipeline_can_disable_backend_optimization_pass(self):
        program = _copy_program()

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
        program = _copy_program()

        lowered = lower_for_target(program, backend=Target.TRITON)
        selection = lowered.metadata["pipeline_selection"]
        assert selection["mode"] == "default"
        assert "ssa.triton.optimize_schedule" in selection["selected_passes"]
        assert selection["reason"] == "default backend pipeline"
        assert (
            selection["categories"][BACKEND_SPECIFIC]
            == selection["categories"][LANGUAGE_SPECIFIC]
        )

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

        expected_counts = {
            Target.TRITON: 3,
            Target.CUDA: 1,
            Target.TILELANG: 3,
        }

        for backend, expected_count in expected_counts.items():
            lowered = lower_for_target(program, backend=backend)
            candidates = lowered.metadata["schedule_candidates"]
            assert len(candidates) == expected_count
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

    def test_platform_capability_validation_uses_profile_contract(self):
        program = _copy_program()

        program = _with_required_capabilities(program, "dtype.fp8")

        with pytest.raises(ValueError, match="does not support `dtype.fp8`"):
            lower_for_target(
                program,
                backend=Target.TRITON,
                target_context=_target_context(unsupported=("dtype.fp8",)),
            )

        lowered = lower_for_target(
            program,
            backend=Target.TRITON,
            target_context=_target_context(supported=("dtype.fp8",)),
        )
        assert lowered.metadata["target_capabilities"]["supported"] == ("dtype.fp8",)

    def test_pow_operation_requires_declared_platform_capability(self):
        program = _program(
            "\ndef pow(x, exponent, out):\n    out = x ** exponent\n",
            (
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="x"),
                TensorSpec(
                    ndim=1,
                    shape=("n",),
                    dtype="float32",
                    name="exponent",
                ),
                TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out"),
            ),
            "pow",
        )

        with pytest.raises(ValueError, match="does not support `math.pow`"):
            lower_for_target(
                program,
                backend=Target.TRITON,
                target_context=_target_context(unsupported=("math.pow",)),
            )

        lowered = lower_for_target(program, backend=Target.TRITON)
        assert lowered.metadata["target_capabilities"]["unresolved"] == ("math.pow",)

    def test_final_capability_validation_cannot_be_bypassed_by_manual_pipeline(self):
        class AddFp8Requirement(Pass):
            name = "test.add_fp8_requirement"

            def run(self, program, context):
                del context

                return _with_required_capabilities(program, "dtype.fp8")

        program = _copy_program()

        with pytest.raises(ValueError, match="does not support `dtype.fp8`"):
            lower_for_target(
                program,
                backend=Target.TRITON,
                target_context=_target_context(unsupported=("dtype.fp8",)),
                pass_pipeline=Pipeline((AddFp8Requirement(),)),
            )
