"""
Comprehensive correctness test for scatter_add on MetaX C500 GPU.

Covers:
  - 1D / 2D / 3D / 4D shapes
  - dim in {0, 1, -1, last-1}
  - dtypes float32 and float16
  - Random index with >=30% duplicate rate (forces atomic accumulation)
  - All-zero index (max race -- every src element targets position 0)
  - Non-contiguous self and src (.t() / [::2] slicing)
  - Boundary: index == self.shape[dim] - 1 (upper bound)
  - src.shape[d] < self.shape[d] on non-scatter dims
  - Out-of-bounds index must raise
"""

import sys
import time

import torch

sys.path.insert(0, "/data/ntops13/src")
import ntops  # noqa: E402


def _make_case(shape, dim, dtype, device, dup_ratio=0.4, full_zero=False,
               src_smaller=None):
    """Build self / index / src with a controlled duplicate ratio."""
    self_t = torch.randn(shape, dtype=dtype, device=device)
    src_shape = list(shape)
    if src_smaller is not None:
        for d, s in src_smaller.items():
            src_shape[d] = s
    src_t = torch.randn(src_shape, dtype=dtype, device=device)

    dim_len = self_t.shape[dim]

    if full_zero:
        index = torch.zeros(src_shape, dtype=torch.long, device=device)
    else:
        # Build an index tensor with a configurable duplicate rate.
        # First fill with monotonic mod, then overwrite `dup_ratio` fraction
        # with random positions (some of which will collide).
        numel = 1
        for s in src_shape:
            numel *= s
        flat = torch.arange(numel, device=device) % max(1, dim_len // 2)
        if dup_ratio > 0:
            n_dup = int(numel * dup_ratio)
            dup_positions = torch.randint(0, dim_len, (n_dup,), device=device)
            flat[:n_dup] = dup_positions
        index = flat.view(src_shape)

    return self_t, src_t, index


def _check_close(a, b, dtype, msg):
    if dtype is torch.float16:
        atol, rtol = 1e-1, 1e-2
    else:
        atol, rtol = 1e-3, 1e-3
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        diff = (a - b).abs()
        raise AssertionError(
            f"{msg}: max|diff|={diff.max().item():.4e} "
            f"mean|diff|={diff.mean().item():.4e}"
        )


def run_correctness():
    device = "cuda"
    results = []
    total = 0
    passed = 0

    # shape x dim x dtype
    cases = []
    shapes = [
        (128,),
        (64, 32),
        (16, 24, 12),
        (8, 6, 4, 3),
    ]
    for shape in shapes:
        ndim = len(shape)
        dims = [0]
        if ndim >= 2:
            dims.append(1)
        dims.append(-1)
        if ndim >= 3:
            dims.append(ndim - 2)
        for dim in dims:
            for dtype in (torch.float32, torch.float16):
                cases.append((shape, dim, dtype, "rand_dup30", 0.4, False, None))
                cases.append((shape, dim, dtype, "all_zero", 0.0, True, None))

    # src smaller than self on non-scatter dim
    cases.append(((16, 24, 12), 1, torch.float32,
                  "src_smaller", 0.4, False, {0: 10, 2: 8}))
    cases.append(((64, 32), 0, torch.float16,
                  "src_smaller", 0.4, False, {1: 20}))

    # boundary: index == self.shape[dim] - 1 everywhere
    for shape, dim in [((32, 16), 0), ((16, 24, 12), 2), ((8, 6, 4, 3), 1)]:
        for dtype in (torch.float32, torch.float16):
            self_t = torch.randn(shape, dtype=dtype, device=device)
            src_t = torch.randn(shape, dtype=dtype, device=device)
            dim_len = self_t.shape[dim]
            index = torch.full(shape, dim_len - 1,
                               dtype=torch.long, device=device)
            ref = torch.scatter_add(self_t, dim, index, src_t)
            got = ntops.torch.scatter_add(self_t, dim, index, src_t)
            total += 1
            try:
                _check_close(ref, got, dtype,
                             f"boundary_max_idx shape={shape} dim={dim} "
                             f"dtype={dtype}")
                passed += 1
                results.append(f"PASS  boundary shape={shape} dim={dim} "
                               f"dtype={dtype}")
            except AssertionError as e:
                results.append(f"FAIL  boundary shape={shape} dim={dim} "
                               f"dtype={dtype} -- {e}")

    # Main parameterized loop
    for shape, dim, dtype, tag, dup, full_zero, src_small in cases:
        total += 1
        self_t, src_t, index = _make_case(
            shape, dim, dtype, device,
            dup_ratio=dup, full_zero=full_zero, src_smaller=src_small,
        )
        ref = torch.scatter_add(self_t.clone(), dim, index, src_t)
        got = ntops.torch.scatter_add(self_t, dim, index, src_t)
        label = (f"shape={shape} dim={dim} dtype={dtype} "
                 f"tag={tag} dup={dup} full_zero={full_zero} "
                 f"src_small={src_small}")
        try:
            _check_close(ref, got, dtype, label)
            passed += 1
            results.append(f"PASS  {label}")
        except AssertionError as e:
            results.append(f"FAIL  {label} -- {e}")

    # Non-contiguous self and src
    total += 1
    base = torch.randn(32, 48, dtype=torch.float32, device=device)
    self_nc = base.t()  # (48, 32) non-contiguous
    src_base = torch.randn(16, 48, dtype=torch.float32, device=device)
    src_nc = src_base.t()  # (48, 16) non-contiguous
    index_nc = torch.randint(0, 32, (48, 16), dtype=torch.long, device=device)
    ref = torch.scatter_add(self_nc.clone(), 1, index_nc, src_nc)
    got = ntops.torch.scatter_add(self_nc, 1, index_nc, src_nc)
    try:
        _check_close(ref, got, torch.float32, "noncontig self+src dim=1")
        passed += 1
        results.append("PASS  noncontig self+src dim=1 float32")
    except AssertionError as e:
        results.append(f"FAIL  noncontig self+src dim=1 -- {e}")

    total += 1
    base = torch.randn(8, 16, 6, dtype=torch.float16, device=device)
    self_nc = base[:, ::2, :]  # (8, 8, 6) non-contiguous
    src_nc = torch.randn(8, 8, 6, dtype=torch.float16, device=device).transpose(0, 2)  # (6, 8, 8) non-contig
    # Adjust self to match src shape: we want self.shape[d] >= src.shape[d]
    # Use self (8, 16, 8) with src (6, 8, 8) on dim=0
    base = torch.randn(8, 16, 8, dtype=torch.float16, device=device)
    self_nc = base[:, ::2, :]  # (8, 8, 8) non-contiguous
    src_nc = torch.randn(6, 8, 8, dtype=torch.float16, device=device)
    index_nc = torch.randint(0, 8, (6, 8, 8), dtype=torch.long, device=device)
    ref = torch.scatter_add(self_nc.clone(), 0, index_nc, src_nc)
    got = ntops.torch.scatter_add(self_nc, 0, index_nc, src_nc)
    try:
        _check_close(ref, got, torch.float16, "noncontig self dim=0 fp16 3D")
        passed += 1
        results.append("PASS  noncontig self dim=0 fp16 3D")
    except AssertionError as e:
        results.append(f"FAIL  noncontig self dim=0 fp16 3D -- {e}")

    # Out-of-bounds index must raise
    total += 1
    self_t = torch.randn(8, 16, dtype=torch.float32, device=device)
    src_t = torch.randn(8, 16, dtype=torch.float32, device=device)
    index_oob = torch.full((8, 16), 99, dtype=torch.long, device=device)
    try:
        ntops.torch.scatter_add(self_t, 1, index_oob, src_t)
        results.append("FAIL  out-of-bounds did not raise")
    except (IndexError, ValueError, RuntimeError) as e:
        passed += 1
        results.append(f"PASS  out-of-bounds raised {type(e).__name__}: {e}")

    total += 1
    index_neg = torch.full((8, 16), -3, dtype=torch.long, device=device)
    try:
        ntops.torch.scatter_add(self_t, 1, index_neg, src_t)
        results.append("FAIL  negative index did not raise")
    except (IndexError, ValueError, RuntimeError) as e:
        passed += 1
        results.append(f"PASS  negative index raised {type(e).__name__}: {e}")

    # Wrong dtype / shape checks
    total += 1
    try:
        ntops.torch.scatter_add(
            self_t, 1,
            torch.randint(0, 16, (8, 16), dtype=torch.int32, device=device),
            src_t,
        )
        results.append("FAIL  wrong index dtype did not raise")
    except TypeError as e:
        passed += 1
        results.append(f"PASS  wrong index dtype raised TypeError: {e}")

    print("=" * 70)
    print(f"Correctness: {passed}/{total}")
    print("=" * 70)
    for r in results:
        print(r)
    if passed != total:
        sys.exit(1)


if __name__ == "__main__":
    t0 = time.perf_counter()
    run_correctness()
    print(f"\nTotal: {time.perf_counter() - t0:.1f}s")
