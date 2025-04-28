import torch
import torch.nn.functional as F

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor
from tests.skippers import skip_if_cuda_not_available


def arrangement(q, k, v, o):
    BLOCK_SIZE_M = ninetoothed.block_size(lower_bound=64, upper_bound=128)
    BLOCK_SIZE_N = ninetoothed.block_size(lower_bound=32, upper_bound=64)

    def arrange_q_or_o(input):
        arranged = input.tile((1, 1, BLOCK_SIZE_M, -1))
        arranged.dtype = arranged.dtype.squeeze((0, 1))

        return arranged

    def arrange_k_or_v(input):
        arranged = (
            input.tile((1, 1, BLOCK_SIZE_N, -1))
            .tile((1, 1, -1, -1))
            .expand((-1, -1, q_arranged.shape[-2], -1))
        )
        arranged.dtype = arranged.dtype.squeeze((0, 1, 3))
        arranged.dtype.dtype = arranged.dtype.dtype.squeeze((0, 1))

        return arranged

    q_arranged = arrange_q_or_o(q)

    return q_arranged, arrange_k_or_v(k), arrange_k_or_v(v), arrange_q_or_o(o)


def application(q, k, v, o):
    q_loaded = (q * 1.44269504089).to(q.dtype)

    acc = ntl.zeros((q.shape[-2], q.shape[-1]), dtype=ntl.float32)
    l_i = ntl.full((q.shape[-2],), 1, dtype=ntl.float32)
    m_i = ntl.full((q.shape[-2],), float("-inf"), dtype=ntl.float32)

    for i in range(k.shape[0]):
        qk = ntl.dot(q_loaded, ntl.trans(k[i]))

        m_ij = ntl.maximum(m_i, ntl.max(qk, 1))
        p = ntl.exp2(qk - m_ij[:, None])
        l_ij = ntl.sum(p, 1)

        alpha = ntl.exp2(m_i - m_ij)
        acc = acc * alpha[:, None] + ntl.dot(p.to(v[i].dtype), v[i])
        m_i = m_ij
        l_i = l_i * alpha + l_ij

    acc /= l_i[:, None]
    o = acc  # noqa: F841


def attention(q, k, v):
    o = torch.empty_like(q, dtype=v.dtype)

    attention_kernel = ninetoothed.make(
        arrangement,
        application,
        (
            Tensor(
                4,
                shape_options=(
                    None,
                    None,
                    {"constexpr": True},
                    {"constexpr": True, "upper_bound": 128},
                ),
            )
            for _ in range(4)
        ),
    )

    attention_kernel(q, k, v, o)

    return o


@skip_if_cuda_not_available
class TestCUDA:
    @classmethod
    def setup_class(cls):
        torch.manual_seed(0)

        shape = (2, 4, 1024, 64)

        cls.q = torch.randn(shape, device="cuda")
        cls.k = torch.randn(shape, device="cuda")
        cls.v = torch.randn(shape, device="cuda")

    def test_fp32(self):
        q = type(self).q.to(torch.float32)
        k = type(self).k.to(torch.float32)
        v = type(self).v.to(torch.float32)

        assert torch.allclose(
            attention(q, k, v),
            F.scaled_dot_product_attention(q, k, v, scale=1),
            atol=0.01,
            rtol=0.01,
        )

    def test_fp16(self):
        q = type(self).q.to(torch.float16)
        k = type(self).k.to(torch.float16)
        v = type(self).v.to(torch.float16)

        assert torch.allclose(
            attention(q, k, v),
            F.scaled_dot_product_attention(q, k, v, scale=1),
            atol=0.01,
        )
