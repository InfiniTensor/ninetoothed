# ruff: noqa: F841
import ninetoothed.language as ntl
from ninetoothed.ir import TensorTypeIR
from ninetoothed.ssa import application_to_ssa, render_ssa_program


def reference_attention(q, k, v, is_causal, o):
    q_loaded = (q * 1.44269504089).to(q.dtype)
    acc = ntl.zeros((q.shape[-2], q.shape[-1]), dtype=ntl.float32)
    l_i = ntl.full((q.shape[-2],), 1, dtype=ntl.float32)
    m_i = ntl.full((q.shape[-2],), float("-inf"), dtype=ntl.float32)

    for i in range(k.shape[0]):
        qk = ntl.dot(q_loaded, ntl.trans(k[i]))
        qk = ntl.where(k[i].offsets(-2) < k.source.shape[-2], qk, float("-inf"))

        if is_causal:
            mask = q.offsets(-2)[:, None] >= k[i].offsets(-2)[None, :]
            qk = ntl.where(mask, qk, float("-inf"))

        m_ij = ntl.maximum(m_i, ntl.max(qk, 1))
        p = ntl.exp2(qk - m_ij[:, None])
        l_ij = ntl.sum(p, 1)
        alpha = ntl.exp2(m_i - m_ij)
        acc = acc * alpha[:, None] + ntl.dot(p.to(v[i].dtype), v[i])
        m_i = m_ij
        l_i = l_i * alpha + l_ij

    acc /= l_i[:, None]
    o = acc


def _attention_tensors():
    return (
        TensorTypeIR("q", 4, dtype="float16", shape=("B", "H", "M", "D")),
        TensorTypeIR("k", 4, dtype="float16", shape=("B", "H", "N", "D")),
        TensorTypeIR("v", 4, dtype="float16", shape=("B", "H", "N", "D")),
        TensorTypeIR("is_causal", 0, dtype="bool", constexpr=True),
        TensorTypeIR("o", 4, dtype="float16", shape=("B", "H", "M", "D")),
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


class TestApplicationSSALowering:
    def test_reference_attention_lowers_to_fine_grained_region_ssa(self):
        ssa = application_to_ssa(
            reference_attention, _attention_tensors(), kind="reference_attention"
        )
        assert ssa is not None
        opcodes = tuple(_opcodes(ssa.blocks[0].operations))
        assert "scf.for" in opcodes
        assert "scf.if" in opcodes
        assert "linalg.dot" in opcodes
        assert "linalg.transpose" in opcodes
        assert "select.where" in opcodes
        assert "reduce.max" in opcodes
        assert "reduce.sum" in opcodes
        assert "math.exp2" in opcodes
        assert "index.offset" in opcodes
        assert "mem.store" in opcodes
        assert "linalg.flash_attention" not in opcodes
        assert not ssa.metadata["coarse_operator_nodes"]

    def test_reference_attention_loop_uses_block_args_for_carried_state(self):
        ssa = application_to_ssa(reference_attention, _attention_tensors())
        loops = [
            operation
            for operation in _operations(ssa.blocks[0].operations)
            if operation.opcode == "scf.for"
        ]
        assert len(loops) == 1
        loop = loops[0]
        carried_names = tuple((item["name"] for item in loop.attrs["iter_args"]))
        assert carried_names == ("acc", "m_i", "l_i")
        assert len(loop.results) == 3
        assert len(loop.regions[0].args) >= 4
        assert loop.regions[0].operations[-1].opcode == "scf.yield"

    def test_namespace_calls_are_not_treated_as_tensor_methods(self):
        ssa = application_to_ssa(reference_attention, _attention_tensors())
        reductions = [
            operation
            for operation in _operations(ssa.blocks[0].operations)
            if operation.opcode in {"reduce.max", "reduce.sum"}
        ]
        assert reductions

        for operation in reductions:
            assert "ntl" not in operation.operands

    def test_textual_render_is_ssa_not_json_or_coarse_attention_node(self):
        ssa = application_to_ssa(reference_attention, _attention_tensors())
        text = render_ssa_program(ssa)
        assert "ssa @reference_attention" in text
        assert "scf.for" in text
        assert "linalg.dot" in text
        assert "shape.dim" in text
        assert '{"kind"' not in text
        assert "AttentionOpIR" not in text
        assert "FlashAttentionOpIR" not in text
