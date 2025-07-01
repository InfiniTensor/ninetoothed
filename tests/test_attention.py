import pytest
import torch
import torch.nn.functional as F

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor
from tests.skippers import skip_if_cuda_not_available

BLOCK_SIZE_M = ninetoothed.block_size(lower_bound=64, upper_bound=128)
BLOCK_SIZE_N = ninetoothed.block_size(lower_bound=32, upper_bound=64)


def arrangement(
    q, k, v, is_causal, o, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N
):
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

    return (
        q_arranged,
        arrange_k_or_v(k),
        arrange_k_or_v(v),
        is_causal,
        arrange_q_or_o(o),
    )


def application(q, k, v, is_causal, o):
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
    o = acc  # noqa: F841


def attention(q, k, v, is_causal=False):
    o = torch.empty_like(q, dtype=v.dtype)

    q_, k_, v_, o_ = (
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
    )
    is_causal_ = Tensor(0, constexpr=True)

    tensors = (q_, k_, v_, is_causal_, o_)

    attention_kernel = ninetoothed.make(arrangement, application, tensors)

    attention_kernel(q, k, v, is_causal, o)

    return o


@skip_if_cuda_not_available
class TestCUDA:
    shapes = ((2, 4, 1024, 64), (2, 4, 1, 64))

    dtypes = (torch.float32, torch.float16)

    is_causal_values = (False, True)

    @classmethod
    def setup_class(cls):
        torch.manual_seed(0)

        cls.args = {
            shape: tuple(torch.randn(shape, device="cuda") for _ in range(3))
            for shape in cls.shapes
        }

    @pytest.mark.parametrize("is_causal", is_causal_values)
    @pytest.mark.parametrize("dtype", dtypes)
    @pytest.mark.parametrize("shape", shapes)
    def test(self, shape, dtype, is_causal):
        q, k, v = (arg.to(dtype) for arg in type(self).args[shape])

        if dtype == torch.float32:
            atol = 0.025
            rtol = 0.025
        elif dtype == torch.float16:
            atol = 0.01
            rtol = 0.01

        assert torch.allclose(
            attention(q, k, v, is_causal=is_causal),
            F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=1),
            atol=atol,
            rtol=rtol,
        )
