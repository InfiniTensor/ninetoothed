from ninetoothed.ir import (
    AxisReductionAssignOpIR,
    CopyOpIR,
    ElementwiseAssignOpIR,
    ElementwiseBinaryOpIR,
    ExprIR,
    FillOpIR,
    FlashAttentionOpIR,
    KernelIR,
    MatmulOpIR,
    ProgramIR,
    ReductionOpIR,
    SSAProgramIR,
    SSATypeIR,
    SSAValueIR,
    TensorTypeIR,
    TransposeOpIR,
    ir_to_dict,
    program_to_ssa,
)


class FakeTensor:
    name = "input"
    ndim = 2
    dtype = "float16"
    shape = ("m", "n")
    constexpr = False
    jagged_dim = None
    source = None


class TestKernelIR:
    def test_tensor_type_can_be_extracted_from_tensor_like_object(self):
        tensor = FakeTensor()
        tensor.source = tensor
        tensor_ir = TensorTypeIR.from_tensor(tensor)
        assert tensor_ir.name == "input"
        assert tensor_ir.ndim == 2
        assert tensor_ir.dtype == "float16"
        assert tensor_ir.shape == ("m", "n")

    def test_kernel_ir_metadata_is_immutably_extended(self):
        kernel = KernelIR(kernel_name="k", source="source", metadata={"a": 1})
        updated = kernel.with_metadata(b=2)
        assert kernel.metadata == {"a": 1}
        assert updated.metadata == {"a": 1, "b": 2}

    def test_kernel_ir_metadata_extension_preserves_program(self):
        program = ProgramIR(
            kind="elementwise",
            operations=(
                ElementwiseBinaryOpIR(
                    operator="add", lhs="x", rhs="y", output="out", extent="n"
                ),
            ),
        )
        kernel = KernelIR(kernel_name="k", source="source", program=program)
        updated = kernel.with_metadata(a=1)
        assert updated.program == program

    def test_kernel_ir_metadata_extension_preserves_ssa(self):
        ssa = SSAProgramIR(
            kind="elementwise",
            inputs=(SSAValueIR("x", SSATypeIR("tensor", dtype="float32")),),
        )
        kernel = KernelIR(kernel_name="k", source="source", ssa=ssa)
        updated = kernel.with_metadata(a=1)
        assert updated.ssa == ssa

    def test_program_ir_can_carry_expression_assignment(self):
        expression = ExprIR(
            kind="call",
            value="exp",
            args=(
                ExprIR(
                    kind="unary", value="neg", args=(ExprIR(kind="var", value="x"),)
                ),
            ),
        )
        program = ProgramIR(
            kind="elementwise",
            operations=(ElementwiseAssignOpIR(output="out", expression=expression),),
        )
        assert program.operations[0].output == "out"
        assert program.operations[0].expression.value == "exp"

    def test_program_ir_can_carry_structured_non_pointwise_ops(self):
        operations = (
            FillOpIR(output="out", value=1.0),
            CopyOpIR(input="x", output="out"),
            ReductionOpIR(operator="sum", input="x", output="out"),
            MatmulOpIR(lhs="a", rhs="b", output="out"),
            TransposeOpIR(input="x", output="out"),
        )
        for operation in operations:
            program = ProgramIR(kind="structured", operations=(operation,))
            assert program.operations[0] == operation

    def test_program_ir_converts_elementwise_expression_to_ssa(self):
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
                            ExprIR(
                                kind="call",
                                value="exp",
                                args=(ExprIR(kind="var", value="y"),),
                            ),
                        ),
                    ),
                ),
            ),
        )
        ssa = program_to_ssa(
            program,
            (
                TensorTypeIR("x", 1, dtype="float32"),
                TensorTypeIR("y", 1, dtype="float32"),
                TensorTypeIR("out", 1, dtype="float32"),
            ),
        )
        opcodes = [operation.opcode for operation in ssa.blocks[0].operations]
        assert opcodes == ["math.exp", "arith.add", "mem.store"]
        assert ssa.blocks[0].operations[0].results[0].name == "%0"
        assert ssa.blocks[0].operations[-1].operands == ("%1", "out")

    def test_program_ir_converts_axis_reduction_to_ssa_reduce(self):
        program = ProgramIR(
            kind="axis_reduction",
            operations=(
                AxisReductionAssignOpIR(
                    output="out",
                    expression=ExprIR(
                        kind="axis_reduce",
                        value={"operator": "sum", "axis": 1},
                        args=(ExprIR(kind="var", value="x"),),
                    ),
                ),
            ),
        )
        ssa = program_to_ssa(
            program,
            (
                TensorTypeIR("x", 2, dtype="float32"),
                TensorTypeIR("out", 1, dtype="float32"),
            ),
        )
        assert ssa.blocks[0].operations[0].opcode == "reduce.sum"
        assert ssa.blocks[0].operations[0].attrs == {"axis": 1}
        assert ssa.outputs[0].name == "out"

    def test_program_to_ssa_reuses_common_pure_expressions(self):
        reduce_x = ExprIR(
            kind="axis_reduce",
            value={"operator": "sum", "axis": 1},
            args=(ExprIR(kind="var", value="x"),),
        )
        program = ProgramIR(
            kind="axis_reduction",
            operations=(
                AxisReductionAssignOpIR(
                    output="out",
                    expression=ExprIR(
                        kind="binary", value="add", args=(reduce_x, reduce_x)
                    ),
                ),
            ),
        )
        ssa = program_to_ssa(
            program,
            (
                TensorTypeIR("x", 2, dtype="float32"),
                TensorTypeIR("out", 1, dtype="float32"),
            ),
        )
        opcodes = [operation.opcode for operation in ssa.blocks[0].operations]
        assert opcodes.count("reduce.sum") == 1
        assert ssa.blocks[0].operations[1].operands == ("%0", "%0")

    def test_program_ir_converts_matmul_and_flash_attention_to_ssa(self):
        matmul = program_to_ssa(
            ProgramIR(
                kind="matmul", operations=(MatmulOpIR(lhs="a", rhs="b", output="out"),)
            ),
            (
                TensorTypeIR("a", 2, dtype="float32"),
                TensorTypeIR("b", 2, dtype="float32"),
                TensorTypeIR("out", 2, dtype="float32"),
            ),
        )
        flash = program_to_ssa(
            ProgramIR(
                kind="flash_attention",
                operations=(
                    FlashAttentionOpIR(
                        query="q", key="k", value="v", output="out", scale=0.125
                    ),
                ),
            ),
            (
                TensorTypeIR("q", 2, dtype="float16"),
                TensorTypeIR("k", 2, dtype="float16"),
                TensorTypeIR("v", 2, dtype="float16"),
                TensorTypeIR("out", 2, dtype="float16"),
            ),
        )
        assert matmul.blocks[0].operations[0].opcode == "linalg.matmul"
        assert flash.blocks[0].operations[0].opcode == "linalg.flash_attention"
        assert flash.blocks[0].operations[0].attrs["scale"] == 0.125

    def test_ssa_ir_is_json_serializable(self):
        program = ProgramIR(
            kind="fill", operations=(FillOpIR(output="out", value=1.0),)
        )
        ssa = program_to_ssa(program, (TensorTypeIR("out", 1, dtype="float32"),))
        payload = ir_to_dict(ssa)
        assert payload["kind"] == "fill"
        assert payload["blocks"][0]["operations"][0]["opcode"] == "arith.constant"
