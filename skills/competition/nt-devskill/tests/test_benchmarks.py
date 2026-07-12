import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.benchmark, pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is not available"
)]

WARMUP = 10
TRIALS = 50


def _bench(fn, args, warmup=WARMUP, trials=TRIALS):
    for _ in range(warmup):
        fn(*args)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(trials):
        fn(*args)
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / trials  # ms


class TestAddBenchmark:
    def test_latency(self, dtype, device):
        from examples.add import add

        a = torch.randn(98432, dtype=dtype, device=device)
        b = torch.randn(98432, dtype=dtype, device=device)

        latency = _bench(add, (a, b))
        print(f"\n  add: {latency:.4f} ms")

        ref_latency = _bench(torch.add, (a, b))
        print(f"  torch.add: {ref_latency:.4f} ms")
        print(f"  ratio: {latency / ref_latency:.2f}x")


class TestMatMulBenchmark:
    def test_latency(self, dtype, device):
        from examples.matmul import mm

        a = torch.randn(1024, 1024, dtype=dtype, device=device)
        b = torch.randn(1024, 1024, dtype=dtype, device=device)

        latency = _bench(mm, (a, b))
        print(f"\n  matmul: {latency:.4f} ms")

        ref_latency = _bench(torch.mm, (a, b))
        print(f"  torch.mm: {ref_latency:.4f} ms")
        print(f"  ratio: {latency / ref_latency:.2f}x")


class TestSoftmaxBenchmark:
    def test_latency(self, dtype, device):
        from examples.softmax import softmax

        x = torch.randn(4096, 4096, dtype=dtype, device=device)

        latency = _bench(softmax, (x,))
        print(f"\n  softmax: {latency:.4f} ms")

        ref_latency = _bench(lambda t: torch.softmax(t, dim=-1), (x,))
        print(f"  torch.softmax: {ref_latency:.4f} ms")
        print(f"  ratio: {latency / ref_latency:.2f}x")


class TestFusedRMSNormBenchmark:
    def test_latency(self, dtype, device):
        from examples.fused_rms_norm import fused_rms_norm

        x = torch.randn(4096, 4096, dtype=dtype, device=device)
        w = torch.randn(4096, dtype=dtype, device=device)
        eps = 1e-5

        latency = _bench(fused_rms_norm, (x, w, eps))
        print(f"\n  fused_rms_norm: {latency:.4f} ms")

        ref_latency = _bench(
            lambda x, w, eps: F.rms_norm(x, x.shape[-1:], w, eps),
            (x, w, eps),
        )
        print(f"  torch rms_norm: {ref_latency:.4f} ms")
        print(f"  ratio: {latency / ref_latency:.2f}x")


class TestSiLUBenchmark:
    def test_latency(self, dtype, device):
        from examples.silu import silu

        x = torch.randn(4096, 4096, dtype=dtype, device=device)

        latency = _bench(silu, (x,))
        print(f"\n  silu: {latency:.4f} ms")

        ref_latency = _bench(F.silu, (x,))
        print(f"  torch.silu: {ref_latency:.4f} ms")
        print(f"  ratio: {latency / ref_latency:.2f}x")


class TestBMMBenchmark:
    def test_latency(self, dtype, device):
        from examples.bmm import bmm

        a = torch.randn(4, 1024, 1024, dtype=dtype, device=device)
        b = torch.randn(4, 1024, 1024, dtype=dtype, device=device)

        latency = _bench(bmm, (a, b))
        print(f"\n  bmm: {latency:.4f} ms")

        ref_latency = _bench(torch.bmm, (a, b))
        print(f"  torch.bmm: {ref_latency:.4f} ms")
        print(f"  ratio: {latency / ref_latency:.2f}x")


class TestAddMMBenchmark:
    def test_latency(self, dtype, device):
        from examples.addmm import addmm

        c = torch.randn(1024, 1024, dtype=dtype, device=device)
        a = torch.randn(1024, 1024, dtype=dtype, device=device)
        b = torch.randn(1024, 1024, dtype=dtype, device=device)

        latency = _bench(addmm, (c, a, b))
        print(f"\n  addmm: {latency:.4f} ms")

        ref_latency = _bench(torch.addmm, (c, a, b))
        print(f"  torch.addmm: {ref_latency:.4f} ms")
        print(f"  ratio: {latency / ref_latency:.2f}x")


class TestSDPABenchmark:
    def test_latency(self, dtype, device):
        from examples.scaled_dot_product_attention import scaled_dot_product_attention

        q = torch.randn(2, 8, 1024, 64, dtype=dtype, device=device)
        k = torch.randn(2, 8, 1024, 64, dtype=dtype, device=device)
        v = torch.randn(2, 8, 1024, 64, dtype=dtype, device=device)

        latency = _bench(scaled_dot_product_attention, (q, k, v))
        print(f"\n  sdpa: {latency:.4f} ms")

        ref_latency = _bench(
            lambda q, k, v: F.scaled_dot_product_attention(q, k, v),
            (q, k, v),
        )
        print(f"  torch.sdpa: {ref_latency:.4f} ms")
        print(f"  ratio: {latency / ref_latency:.2f}x")


class TestSwiGLUBenchmark:
    def test_latency(self, dtype, device):
        from examples.swiglu import swiglu

        a = torch.randn(4096, 4096, dtype=dtype, device=device)
        b = torch.randn(4096, 4096, dtype=dtype, device=device)

        latency = _bench(swiglu, (a, b))
        print(f"\n  swiglu: {latency:.4f} ms")

        ref_latency = _bench(
            lambda a, b: a * (b * torch.sigmoid(b.float()).to(dtype)),
            (a, b),
        )
        print(f"  torch.swiglu: {ref_latency:.4f} ms")
        print(f"  ratio: {latency / ref_latency:.2f}x")


class TestConv2DBenchmark:
    def test_latency(self, dtype, device):
        from examples.conv2d import conv2d

        input = torch.randn(4, 64, 32, 32, dtype=dtype, device=device)
        filter = torch.randn(128, 64, 3, 3, dtype=dtype, device=device)

        latency = _bench(conv2d, (input, filter))
        print(f"\n  conv2d: {latency:.4f} ms")

        ref_latency = _bench(lambda i, f: F.conv2d(i, f), (input, filter))
        print(f"  torch.conv2d: {ref_latency:.4f} ms")
        print(f"  ratio: {latency / ref_latency:.2f}x")


class TestMaxPool2DBenchmark:
    def test_latency(self, dtype, device):
        from examples.max_pool2d import max_pool2d

        input = torch.randn(4, 64, 32, 32, dtype=dtype, device=device)
        window_shape = (2, 2)

        latency = _bench(max_pool2d, (input, window_shape))
        print(f"\n  max_pool2d: {latency:.4f} ms")

        ref_latency = _bench(
            lambda x: F.max_pool2d(x, window_shape, stride=window_shape),
            (input,),
        )
        print(f"  torch.max_pool2d: {ref_latency:.4f} ms")
        print(f"  ratio: {latency / ref_latency:.2f}x")


class TestRotaryPositionEmbeddingBenchmark:
    def test_latency(self, dtype, device):
        from examples.rotary_position_embedding import rotary_position_embedding

        batch, seq_len, num_heads, head_dim = 2, 128, 8, 64
        input = torch.randn(batch, seq_len, num_heads, head_dim, dtype=dtype, device=device)
        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim // 2, dtype=torch.float32, device=device) / (head_dim // 2)))
        angles = positions[:, None] * freqs[None, :]
        sin_table = torch.sin(angles).to(dtype)
        cos_table = torch.cos(angles).to(dtype)

        latency = _bench(rotary_position_embedding, (input, sin_table, cos_table))
        print(f"\n  rotary_position_embedding: {latency:.4f} ms")
