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

    def test_kernel_metadata_is_immutably_extended(self):
        kernel = Kernel(kernel_name="k", source="source", metadata={"a": 1})
        updated = kernel.with_metadata(b=2)
        assert kernel.metadata == {"a": 1}
        assert updated.metadata == {"a": 1, "b": 2}

    def test_kernel_metadata_extension_preserves_ssa(self):
        program = ssa.Program(
            kind="elementwise",
            inputs=(ssa.Value("x", ssa.Type("tensor", dtype="float32")),),
        )
        kernel = Kernel(kernel_name="k", source="source", ssa=program)
        updated = kernel.with_metadata(a=1)
        assert updated.ssa == program

    def test_ssa_text_render_is_readable(self):
        value = ssa.Value("%0", ssa.Type("scalar", dtype="float32"))
        program = ssa.Program(
            kind="add",
            blocks=(
                ssa.Block(
                    operations=(
                        ssa.Operation(
                            "arith.constant",
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
                            "arith.constant",
                            results=(
                                ssa.Value("%0", ssa.Type("scalar", dtype="float32")),
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
