import pytest
import torch
import torch.nn.functional as F

from tests.conftest import assert_allclose


class TestAdd:
    def test_correctness(self, dtype, device):
        from examples.add import add

        a = torch.randn(98432, dtype=dtype, device=device)
        b = torch.randn(98432, dtype=dtype, device=device)

        expected = torch.add(a, b)
        actual = add(a, b)

        assert_allclose(actual, expected, atol=0, rtol=0)


class TestSoftmax:
    def test_correctness(self, dtype, device):
        from examples.softmax import softmax

        x = torch.randn(1823, 781, dtype=dtype, device=device)

        expected = torch.softmax(x, dim=-1)
        actual = softmax(x)

        assert_allclose(actual, expected, atol=0.001, rtol=0)


class TestMatMul:
    def test_correctness(self, dtype, device):
        from examples.matmul import mm

        a = torch.randn(512, 512, dtype=dtype, device=device)
        b = torch.randn(512, 512, dtype=dtype, device=device)

        expected = torch.mm(a, b)
        actual = mm(a, b)

        assert_allclose(actual, expected, atol=0.01, rtol=0)


class TestFusedRMSNorm:
    def test_correctness(self, dtype, device):
        from examples.fused_rms_norm import fused_rms_norm

        x = torch.randn(1151, 8192, dtype=dtype, device=device)
        w = torch.randn(8192, dtype=dtype, device=device)
        eps = 1e-5

        expected = F.rms_norm(x, x.shape[-1:], w, eps)
        actual = fused_rms_norm(x, w, eps)

        assert_allclose(actual, expected, atol=0.001, rtol=0.005)


class TestSiLU:
    def test_correctness(self, dtype, device):
        from examples.silu import silu

        x = torch.randn(8, 256, 512, dtype=dtype, device=device)

        expected = F.silu(x)
        actual = silu(x)

        assert_allclose(actual, expected, atol=0.001, rtol=0.001)


class TestBMM:
    def test_correctness(self, dtype, device):
        from examples.bmm import bmm

        a = torch.randn(4, 512, 1024, dtype=dtype, device=device)
        b = torch.randn(4, 1024, 2028, dtype=dtype, device=device)

        expected = torch.bmm(a, b)
        actual = bmm(a, b)

        assert_allclose(actual, expected, atol=0.01, rtol=0)


class TestAddMM:
    def test_correctness(self, dtype, device):
        from examples.addmm import addmm

        c = torch.randn(512, 512, dtype=dtype, device=device)
        a = torch.randn(512, 512, dtype=dtype, device=device)
        b = torch.randn(512, 512, dtype=dtype, device=device)

        expected = torch.addmm(c, a, b)
        actual = addmm(c, a, b)

        assert_allclose(actual, expected, atol=0.01, rtol=0.01)


class TestScaledDotProductAttention:
    def test_correctness(self, dtype, device):
        from examples.scaled_dot_product_attention import scaled_dot_product_attention

        q = torch.randn(2, 8, 1024, 64, dtype=dtype, device=device)
        k = torch.randn(2, 8, 1024, 64, dtype=dtype, device=device)
        v = torch.randn(2, 8, 1024, 64, dtype=dtype, device=device)

        expected = F.scaled_dot_product_attention(q, k, v)
        actual = scaled_dot_product_attention(q, k, v)

        assert_allclose(actual, expected, atol=0.01, rtol=0)


class TestSwiGLU:
    def test_correctness(self, dtype, device):
        from examples.swiglu import swiglu

        a = torch.randn(4096, dtype=dtype, device=device)
        b = torch.randn(4096, dtype=dtype, device=device)

        expected = a * (b * torch.sigmoid(b.float()).to(dtype))
        actual = swiglu(a, b)

        assert_allclose(actual, expected, atol=0.001, rtol=0.001)


class TestConv2D:
    def test_correctness(self, dtype, device):
        from examples.conv2d import conv2d

        input = torch.randn(1, 3, 32, 32, dtype=dtype, device=device)
        filter = torch.randn(16, 3, 3, 3, dtype=dtype, device=device)

        expected = F.conv2d(input, filter)
        actual = conv2d(input, filter)

        assert_allclose(actual, expected, atol=0.01, rtol=0.01)


class TestRotaryPositionEmbedding:
    def test_correctness(self, dtype, device):
        from examples.rotary_position_embedding import rotary_position_embedding

        batch, seq_len, num_heads, head_dim = 2, 128, 8, 64
        input = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)

        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim // 2, dtype=torch.float32, device=device) / (head_dim // 2)))
        angles = positions[:, None] * freqs[None, :]
        sin_table = torch.sin(angles).to(dtype)
        cos_table = torch.cos(angles).to(dtype)

        actual = rotary_position_embedding(input, sin_table, cos_table)

        assert actual.shape == input.shape
        assert not torch.allclose(actual, input, atol=0.01)


class TestMaxPool2D:
    def test_correctness(self, dtype, device):
        from examples.max_pool2d import max_pool2d

        input = torch.randn(1, 3, 32, 32, dtype=dtype, device=device)
        window_shape = (2, 2)

        expected = F.max_pool2d(input, window_shape, stride=window_shape)
        actual = max_pool2d(input, window_shape)

        assert_allclose(actual, expected, atol=0, rtol=0)
