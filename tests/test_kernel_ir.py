import inspect

import pytest

from ninetoothed.backends import Artifact, Capability, Target, emit
from ninetoothed.compiler.passes import Context, Descriptor, PipelineSpec
from ninetoothed.frontend.python import from_source
from ninetoothed.ir import Kernel, TensorSpec, ir_to_dict, ssa


class FakeTensor:
    name = "input"
    ndim = 2
    dtype = "float16"
    shape = ("m", "n")
    constexpr = False
    jagged_dim = None
    source = None


class TestKernel:
    def test_tensor_spec_can_be_extracted_from_tensor_like_object(self):
        tensor = FakeTensor()
        tensor.source = tensor
        tensor_spec = TensorSpec.from_tensor(tensor)
        assert tensor_spec.name == "input"
        assert tensor_spec.ndim == 2
        assert tensor_spec.dtype == "float16"
        assert tensor_spec.shape == ("m", "n")

    def test_public_ir_dataclass_constructors_are_keyword_only(self):
        classes = (
            TensorSpec,
            Kernel,
            ssa.Type,
            ssa.Value,
            ssa.Operation,
            ssa.Block,
            ssa.Program,
            Capability,
            Artifact,
            PipelineSpec,
            Context,
            Descriptor,
        )

        for cls in classes:
            parameters = inspect.signature(cls).parameters.values()
            kinds = {parameter.kind for parameter in parameters}
            assert kinds <= {inspect.Parameter.KEYWORD_ONLY}

    def test_ir_dataclasses_reject_positional_construction(self):
        with pytest.raises(TypeError):
            TensorSpec(1, ("n",), "float32", name="x")  # type: ignore[misc]

        with pytest.raises(TypeError):
            Kernel("k", "source")  # type: ignore[misc]

        with pytest.raises(TypeError):
            ssa.Type("tensor")  # type: ignore[misc]

        with pytest.raises(TypeError):
            ssa.Operation("arith.constant")  # type: ignore[misc]

    def test_tensor_spec_name_controls_binding_not_source_name(self):
        tensor_spec = TensorSpec(
            ndim=1,
            shape=("n",),
            dtype="float32",
            name="x",
            attrs={"source_name": "storage_x"},
        )
        program = from_source(
            "\ndef copy(x):\n    return x\n",
            tensor_irs=(tensor_spec,),
            kind="copy",
        )

        assert program is not None
        assert program.inputs[0].name == "x"
        assert program.inputs[0].type.attrs["source_name"] == "storage_x"

    def test_kernel_metadata_is_immutably_extended(self):
        kernel = Kernel(kernel_name="k", source="source", metadata={"a": 1})
        updated = kernel.with_metadata(b=2)
        assert kernel.metadata == {"a": 1}
        assert updated.metadata == {"a": 1, "b": 2}

    def test_kernel_metadata_extension_preserves_ssa(self):
        program = ssa.Program(
            kind="elementwise",
            inputs=(
                ssa.Value(name="x", type=ssa.Type(kind="tensor", dtype="float32")),
            ),
        )
        kernel = Kernel(kernel_name="k", source="source", ssa=program)
        updated = kernel.with_metadata(a=1)
        assert updated.ssa == program

    def test_ssa_text_render_is_readable(self):
        value = ssa.Value(name="%0", type=ssa.Type(kind="scalar", dtype="float32"))
        program = ssa.Program(
            kind="add",
            blocks=(
                ssa.Block(
                    operations=(
                        ssa.Operation(
                            opcode="arith.constant",
                            results=(value,),
                            attrs={"value": 1.0},
                        ),
                    )
                ),
            ),
        )
        text = ssa.render(program)
        assert text.startswith("ssa @add {")
        assert "%0 = arith.constant" in text
        assert '{"kind"' not in text

    def test_ir_to_dict_is_json_serializable(self):
        program = ssa.Program(
            kind="fill",
            blocks=(
                ssa.Block(
                    operations=(
                        ssa.Operation(
                            opcode="arith.constant",
                            results=(
                                ssa.Value(
                                    name="%0",
                                    type=ssa.Type(kind="scalar", dtype="float32"),
                                ),
                            ),
                            attrs={"value": 1.0},
                        ),
                    )
                ),
            ),
        )
        payload = ir_to_dict(program)
        assert payload["kind"] == "fill"
        assert payload["blocks"][0]["operations"][0]["opcode"] == "arith.constant"

    def test_kernel_name_controls_generated_artifact_names_and_entrypoint(self):
        value = ssa.Value(name="out", type=ssa.Type(kind="tensor", shape=("n",)))
        program = ssa.Program(
            kind="generated_kernel",
            inputs=(value,),
            outputs=(value,),
            blocks=(ssa.Block(operations=()),),
        )
        tensor_spec = TensorSpec(ndim=1, shape=("n",), dtype="float32", name="out")
        kernel = Kernel(
            kernel_name="generated_kernel",
            source="source",
            tensors=(tensor_spec,),
            ssa=program,
        )
        artifact = emit(kernel, Target.CUDA)
        assert "generated_kernel.cu" in artifact.sources
        assert artifact.entrypoint == "launch_generated_kernel"
        assert "generated_kernel_kernel" in artifact.primary_source
