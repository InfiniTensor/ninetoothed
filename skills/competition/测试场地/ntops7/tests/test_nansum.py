"""Comprehensive correctness test for nansum operator."""
import torch
from ntops.torch.nansum import nansum as nt_nansum


def check(name, got, expected, atol=1e-2, rtol=1e-3):
    ok = torch.allclose(got, expected, atol=atol, rtol=rtol)
    if not ok:
        diff = (got - expected).abs()
        print(
            f"  FAIL {name}: max_diff={diff.max().item():.6f}, "
            f"mean_diff={diff.mean().item():.6f}"
        )
        print(f"    got[:5]={got.flatten()[:5].tolist()}")
        print(f"    exp[:5]={expected.flatten()[:5].tolist()}")
    else:
        print(f"  PASS {name}")
    return ok


def test_all():
    device = "cuda"
    torch.manual_seed(42)
    passed = 0
    total = 0

    shapes = [(128, 256), (64, 8192), (1024, 1024), (4, 32, 128)]
    dims = [None, 0, 1]
    dtypes = [torch.float32, torch.float16]

    for shape in shapes:
        for dim in dims:
            if dim is not None and dim >= len(shape):
                continue
            for dtype in dtypes:
                name = f"shape={shape} dim={dim} dtype={dtype}"
                total += 1
                x = torch.randn(shape, dtype=dtype, device=device)
                mask = torch.rand(shape, device=device) < 0.1
                x[mask] = float("nan")
                got = nt_nansum(x, dim=dim)
                exp = torch.nansum(x, dim=dim)
                atol = 0.05 if dtype == torch.float16 else 1e-2
                rtol = 1e-2 if dtype == torch.float16 else 1e-3
                if check(name, got, exp, atol=atol, rtol=rtol):
                    passed += 1

    # All-NaN row test
    for dtype in dtypes:
        name = f"all_nan_row dtype={dtype}"
        total += 1
        x = torch.full((8, 128), float("nan"), dtype=dtype, device=device)
        x[:, :10] = 1.0
        x[3, :] = float("nan")
        got = nt_nansum(x, dim=1)
        exp = torch.nansum(x, dim=1)
        atol = 0.05 if dtype == torch.float16 else 1e-2
        if check(name, got, exp, atol=atol):
            passed += 1

    # All NaN entire tensor
    for dtype in dtypes:
        name = f"all_nan_tensor dtype={dtype}"
        total += 1
        x = torch.full((32, 64), float("nan"), dtype=dtype, device=device)
        got = nt_nansum(x)
        exp = torch.nansum(x)
        if check(name, got, exp, atol=0):
            passed += 1

    # Non-contiguous: transposed
    for dtype in dtypes:
        name = f"noncontig_transpose dtype={dtype}"
        total += 1
        x = torch.randn(256, 128, dtype=dtype, device=device)
        mask = torch.rand(256, 128, device=device) < 0.1
        x[mask] = float("nan")
        xt = x.t()
        got = nt_nansum(xt, dim=0)
        exp = torch.nansum(xt, dim=0)
        atol = 0.05 if dtype == torch.float16 else 1e-2
        if check(name, got, exp, atol=atol):
            passed += 1

    # Non-contiguous: sliced
    for dtype in dtypes:
        name = f"noncontig_slice dtype={dtype}"
        total += 1
        x = torch.randn(128, 256, dtype=dtype, device=device)
        mask = torch.rand(128, 256, device=device) < 0.1
        x[mask] = float("nan")
        xs = x[:, ::2]
        got = nt_nansum(xs, dim=1)
        exp = torch.nansum(xs, dim=1)
        atol = 0.05 if dtype == torch.float16 else 1e-2
        if check(name, got, exp, atol=atol):
            passed += 1

    # keepdim=True
    for dtype in dtypes:
        name = f"keepdim dim=1 dtype={dtype}"
        total += 1
        x = torch.randn(64, 128, dtype=dtype, device=device)
        mask = torch.rand(64, 128, device=device) < 0.1
        x[mask] = float("nan")
        got = nt_nansum(x, dim=1, keepdim=True)
        exp = torch.nansum(x, dim=1, keepdim=True)
        assert got.shape == exp.shape, f"Shape mismatch: {got.shape} vs {exp.shape}"
        atol = 0.05 if dtype == torch.float16 else 1e-2
        if check(name, got, exp, atol=atol):
            passed += 1

    # keepdim=True with dim=None
    for dtype in dtypes:
        name = f"keepdim dim=None dtype={dtype}"
        total += 1
        x = torch.randn(32, 64, dtype=dtype, device=device)
        got = nt_nansum(x, keepdim=True)
        exp = torch.nansum(x, keepdim=True)
        assert got.shape == exp.shape, f"Shape mismatch: {got.shape} vs {exp.shape}"
        if check(name, got, exp, atol=0.05 if dtype == torch.float16 else 1e-2):
            passed += 1

    # Large reduction dim (8192+)
    for dtype in dtypes:
        name = f"large_reduction dim=1 size=8192 dtype={dtype}"
        total += 1
        x = torch.randn(64, 8192, dtype=dtype, device=device)
        mask = torch.rand(64, 8192, device=device) < 0.1
        x[mask] = float("nan")
        got = nt_nansum(x, dim=1)
        exp = torch.nansum(x, dim=1)
        atol = 0.5 if dtype == torch.float16 else 0.1
        if check(name, got, exp, atol=atol):
            passed += 1

    # No NaN at all (ensure normal path works)
    for dtype in dtypes:
        name = f"no_nan dtype={dtype}"
        total += 1
        x = torch.randn(64, 128, dtype=dtype, device=device)
        got = nt_nansum(x, dim=1)
        exp = torch.nansum(x, dim=1)
        atol = 0.05 if dtype == torch.float16 else 1e-2
        if check(name, got, exp, atol=atol):
            passed += 1

    # Negative dim
    for dtype in dtypes:
        name = f"negative_dim=-1 dtype={dtype}"
        total += 1
        x = torch.randn(64, 128, dtype=dtype, device=device)
        mask = torch.rand(64, 128, device=device) < 0.1
        x[mask] = float("nan")
        got = nt_nansum(x, dim=-1)
        exp = torch.nansum(x, dim=-1)
        atol = 0.05 if dtype == torch.float16 else 1e-2
        if check(name, got, exp, atol=atol):
            passed += 1

    # 1D tensor
    for dtype in dtypes:
        name = f"1d_tensor dtype={dtype}"
        total += 1
        x = torch.randn(1024, dtype=dtype, device=device)
        x[::10] = float("nan")
        got = nt_nansum(x)
        exp = torch.nansum(x)
        atol = 0.05 if dtype == torch.float16 else 1e-2
        if check(name, got, exp, atol=atol):
            passed += 1

    # 3D tensor dim=1
    for dtype in dtypes:
        name = f"3d_dim1 dtype={dtype}"
        total += 1
        x = torch.randn(4, 32, 64, dtype=dtype, device=device)
        mask = torch.rand(4, 32, 64, device=device) < 0.1
        x[mask] = float("nan")
        got = nt_nansum(x, dim=1)
        exp = torch.nansum(x, dim=1)
        atol = 0.05 if dtype == torch.float16 else 1e-2
        if check(name, got, exp, atol=atol):
            passed += 1

    print(f"\n=== Results: {passed}/{total} passed ===")
    return passed == total


if __name__ == "__main__":
    success = test_all()
    exit(0 if success else 1)
