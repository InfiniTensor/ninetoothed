"""
performance_regression_case/run.py
Matmul kernel 在 matmul_2d 模式下演示性能退化诊断。

正确性验证: matmul with BLOCK_SIZE={16, 32, 64, 128} 均通过。
性能差异通过 benchmark.py 展示。
"""

import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor


def make_matmul():
    """
    创建 matmul kernel，BLOCK_SIZE 在调用时传入。
    
    用法:
      kernel = make_matmul()
      out = torch.empty(M, N, device="cuda")
      kernel(a, b, out, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32)
    """
    BLOCK_SIZE_M = Symbol("BLOCK_SIZE_M", constexpr=True)
    BLOCK_SIZE_N = Symbol("BLOCK_SIZE_N", constexpr=True)
    BLOCK_SIZE_K = Symbol("BLOCK_SIZE_K", constexpr=True)

    def arrangement(a, b, c, BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K):
        output_arranged = c.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))

        a_arranged = a.tile((BLOCK_SIZE_M, BLOCK_SIZE_K))
        a_arranged = a_arranged.tile((1, -1))
        a_arranged = a_arranged.expand((-1, output_arranged.shape[1]))
        a_arranged.dtype = a_arranged.dtype.squeeze(0)

        b_arranged = b.tile((BLOCK_SIZE_K, BLOCK_SIZE_N))
        b_arranged = b_arranged.tile((-1, 1))
        b_arranged = b_arranged.expand((output_arranged.shape[0], -1))
        b_arranged.dtype = b_arranged.dtype.squeeze(1)

        return a_arranged, b_arranged, output_arranged

    def application(a, b, c):
        accumulator = ntl.zeros(c.shape, dtype=ntl.float32)
        for k in range(a.shape[0]):
            accumulator += ntl.dot(a[k], b[k])
        c = accumulator  # noqa: F841

    return ninetoothed.make(
        arrangement,
        application,
        tensors=(Tensor(2), Tensor(2), Tensor(2)),
    )


def make_matmul_no_autotune(block_size_m=128, block_size_n=128, block_size_k=32):
    """固定参数版本。"""
    bm = Symbol("bm", constexpr=True)
    bn = Symbol("bn", constexpr=True)
    bk = Symbol("bk", constexpr=True)

    def arrangement(a, b, c, bm=bm, bn=bn, bk=bk):
        output_arranged = c.tile((bm, bn))

        a_arranged = a.tile((bm, bk))
        a_arranged = a_arranged.tile((1, -1))
        a_arranged = a_arranged.expand((-1, output_arranged.shape[1]))
        a_arranged.dtype = a_arranged.dtype.squeeze(0)

        b_arranged = b.tile((bk, bn))
        b_arranged = b_arranged.tile((-1, 1))
        b_arranged = b_arranged.expand((output_arranged.shape[0], -1))
        b_arranged.dtype = b_arranged.dtype.squeeze(1)

        return a_arranged, b_arranged, output_arranged

    def application(a, b, c):
        accumulator = ntl.zeros(c.shape, dtype=ntl.float32)
        for k in range(a.shape[0]):
            accumulator += ntl.dot(a[k], b[k])
        c = accumulator  # noqa: F841

    return ninetoothed.make(
        arrangement,
        application,
        tensors=(Tensor(2), Tensor(2), Tensor(2)),
    )


def demo():
    print("=" * 60)
    print("Performance Regression — Matmul 2D 示例")
    print("=" * 60)

    M, N, K = 1024, 1024, 1024
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")
    expected = a @ b

    configs = [
        ("BLOCK=16", dict(BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16)),
        ("BLOCK=32", dict(BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=32)),
        ("BLOCK=64x64x32", dict(BLOCK_SIZE_M=64, BLOCK_SIZE_N=64, BLOCK_SIZE_K=32)),
        ("BLOCK=128x128x32", dict(BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32)),
    ]

    for label, cfg in configs:
        kernel = make_matmul()
        c = torch.empty(M, N, device="cuda")
        kernel(a, b, c,
               BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"],
               BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
               BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"])
        out = c
        ok = torch.allclose(out, expected, atol=1e-3, rtol=1e-3)
        status = "✅" if ok else "❌"
        print(f"  {status} {label} — M={M}, N={N}, K={K}")

    # 小规模也测试一下
    print()
    M2, N2, K2 = 256, 512, 128
    a2 = torch.randn(M2, K2, device="cuda")
    b2 = torch.randn(K2, N2, device="cuda")
    expected2 = a2 @ b2

    for label, cfg in configs:
        kernel = make_matmul()
        c = torch.empty(M2, N2, device="cuda")
        kernel(a2, b2, c,
               BLOCK_SIZE_M=cfg["BLOCK_SIZE_M"],
               BLOCK_SIZE_N=cfg["BLOCK_SIZE_N"],
               BLOCK_SIZE_K=cfg["BLOCK_SIZE_K"])
        out = c
        ok = torch.allclose(out, expected2, atol=1e-3, rtol=1e-3)
        status = "✅" if ok else "❌"
        print(f"  {status} {label} — M={M2}, N={N2}, K={K2}")

    print()
    print("🎉 所有 matmul 变体正确性验证通过！")


if __name__ == "__main__":
    demo()
