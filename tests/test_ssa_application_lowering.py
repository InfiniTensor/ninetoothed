# ruff: noqa: F841
import unittest

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


class ApplicationSSALoweringTest(unittest.TestCase):
    def test_reference_attention_lowers_to_fine_grained_region_ssa(self):
        ssa = application_to_ssa(
            reference_attention,
            _attention_tensors(),
            kind="reference_attention",
        )
        self.assertIsNotNone(ssa)

        opcodes = tuple(_opcodes(ssa.blocks[0].operations))
        self.assertIn("scf.for", opcodes)
        self.assertIn("scf.if", opcodes)
        self.assertIn("linalg.dot", opcodes)
        self.assertIn("linalg.transpose", opcodes)
        self.assertIn("select.where", opcodes)
        self.assertIn("reduce.max", opcodes)
        self.assertIn("reduce.sum", opcodes)
        self.assertIn("math.exp2", opcodes)
        self.assertIn("index.offset", opcodes)
        self.assertIn("mem.store", opcodes)
        self.assertNotIn("linalg.flash_attention", opcodes)
        self.assertFalse(ssa.metadata["coarse_operator_nodes"])

    def test_reference_attention_loop_uses_block_args_for_carried_state(self):
        ssa = application_to_ssa(reference_attention, _attention_tensors())
        loops = [
            operation
            for operation in _operations(ssa.blocks[0].operations)
            if operation.opcode == "scf.for"
        ]
        self.assertEqual(len(loops), 1)

        loop = loops[0]
        carried_names = tuple(item["name"] for item in loop.attrs["iter_args"])
        self.assertEqual(carried_names, ("acc", "m_i", "l_i"))
        self.assertEqual(len(loop.results), 3)
        self.assertGreaterEqual(len(loop.regions[0].args), 4)
        self.assertEqual(loop.regions[0].operations[-1].opcode, "scf.yield")

    def test_namespace_calls_are_not_treated_as_tensor_methods(self):
        ssa = application_to_ssa(reference_attention, _attention_tensors())
        reductions = [
            operation
            for operation in _operations(ssa.blocks[0].operations)
            if operation.opcode in {"reduce.max", "reduce.sum"}
        ]

        self.assertTrue(reductions)
        for operation in reductions:
            self.assertNotIn("ntl", operation.operands)

    def test_textual_render_is_ssa_not_json_or_coarse_attention_node(self):
        ssa = application_to_ssa(reference_attention, _attention_tensors())
        text = render_ssa_program(ssa)

        self.assertIn("ssa @reference_attention", text)
        self.assertIn("scf.for", text)
        self.assertIn("linalg.dot", text)
        self.assertIn("shape.dim", text)
        self.assertNotIn('{"kind"', text)
        self.assertNotIn("AttentionOpIR", text)
        self.assertNotIn("FlashAttentionOpIR", text)


if __name__ == "__main__":
    unittest.main()
