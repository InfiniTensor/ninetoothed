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
    from ninetoothed.compiler.passes import lower_for_target, program_facts
    from ninetoothed.frontend.python import from_source

    tensors = tuple(
        TensorSpec(name=name, ndim=1, shape=(4,), dtype="float32")
        for name in ("x", "out")
    )
    kernel = Kernel(
        kernel_name="copy",
        source="def copy(x): pass",
        tensors=tensors,
        ssa=lower_for_target(
            from_source("def copy(x, out):\n    out = x\n", tensors, kind="copy"),
            backend="triton",
            tensors=tensors,
        ),
        compiler_options={"passes": ["canonicalize"]},
        metadata={"nested": {"value": 1}},
    )
    restored = pickle.loads(pickle.dumps(kernel))
    assert restored == kernel
    assert restored.compiler_options["passes"] == ("canonicalize",)
    assert program_facts(restored.ssa, "analysis")["store_count"] == 1

    with pytest.raises(TypeError):
        restored.metadata["nested"]["value"] = 2
