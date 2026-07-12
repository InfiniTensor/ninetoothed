import pickle

import pytest

from ninetoothed.ir import Kernel, TensorSpec, ssa


def test_ssa_mappings_and_nested_sequences_are_immutable():
    operation = ssa.Operation(
        opcode="arith.constant",
        attrs={"nested": {"values": [1, 2]}},
    )
    program = ssa.Program(
        kind="immutable",
        blocks=(ssa.Block(operations=(operation,)),),
        metadata={"trace": ["frontend"]},
    )

    with pytest.raises(TypeError):
        program.metadata["new"] = True

    with pytest.raises(TypeError):
        operation.attrs["nested"]["new"] = True

    assert operation.attrs["nested"]["values"] == (1, 2)
    assert program.metadata["trace"] == ("frontend",)


def test_kernel_mappings_are_immutable_and_pickle_safe():
    kernel = Kernel(
        kernel_name="copy",
        source="def copy(x): pass",
        tensors=(TensorSpec(name="x", ndim=1, attrs={"shape": [4]}),),
        compiler_options={"passes": ["canonicalize"]},
        metadata={"nested": {"value": 1}},
    )
    restored = pickle.loads(pickle.dumps(kernel))
    assert restored == kernel
    assert restored.compiler_options["passes"] == ("canonicalize",)

    with pytest.raises(TypeError):
        restored.metadata["nested"]["value"] = 2
