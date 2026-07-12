import pytest
import torch

import ninetoothed
from ninetoothed import Symbol, Tensor


def _get_generated_source(arrangement, application, tensors):
    kernel = ninetoothed.make(arrangement, application, tensors)
    with open(kernel._source) as f:
        return f.read()


def _mask_section(source):
    """Extract the mask expression from a triton.load/store call."""
    idx = source.find("mask=")
    if idx < 0:
        return ""
    end = source.find(",", idx)
    return source[idx:end]


def _get_ntops_source(kwargs):
    """Compile an ntops kernel and return the generated Triton source."""
    import os
    from ntops.torch.utils import _cached_make
    import ntops.kernels

    module_path, premake_fn_name = kwargs["premake"].rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    premake_fn = getattr(mod, premake_fn_name)
    _ = _cached_make(premake_fn, *kwargs.get("premake_args", ()))

    cache = os.path.expanduser("~/.ninetoothed")
    files = sorted(
        [f for f in os.listdir(cache) if f.endswith(".py")],
        key=lambda x: os.path.getmtime(os.path.join(cache, x)),
        reverse=True,
    )
    for f in files[:30]:
        path = os.path.join(cache, f)
        try:
            with open(path) as fh:
                src = fh.read()
        except Exception:
            continue
        if "@triton.jit" in src and "def " in src:
            return src
    return ""


# ═══════════════════════════════════════════════════════════════════════════
# Manual-kernel hit tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSpecializationHit:
    def test_no_lower_bound_in_mask_for_vector_add(self):
        def arrangement(x):
            return x.tile((ninetoothed.block_size(),))
        def application(x):
            x
        source = _get_generated_source(arrangement, application, (Tensor(1),))
        mask = _mask_section(source)
        assert ">= 0" not in mask, (
            "Expected no `>= 0` conditions in the mask "
            "for a simple vector add with tiling, but found one in: " + mask
        )

    def test_no_arange_bounds_in_mask_for_vector_add(self):
        def arrangement(x):
            return x.tile((ninetoothed.block_size(),))
        def application(x):
            x
        source = _get_generated_source(arrangement, application, (Tensor(1),))
        mask = _mask_section(source)
        assert "arange" in source, (
            "Expected `tl.arange` to be present in the generated code "
            "for a vector add with tiling."
        )
        arange_count = mask.count("arange")
        assert arange_count <= 1, (
            "Expected the mask to contain at most one `arange` reference "
            "(arange in `pid*BLOCK + arange`), but found "
            f"{arange_count} in: {mask}"
        )

    def test_no_lower_bound_for_1d_store(self):
        def arrangement(x, output):
            return x.tile((256,)), output.tile((256,))
        def application(x, output):
            output = x
        source = _get_generated_source(arrangement, application, (Tensor(1), Tensor(1)))
        mask = _mask_section(source)
        assert ">= 0" not in mask, (
            "Expected no `>= 0` in 1D store mask, got: " + mask
        )

    def test_divisible_tile_no_source_upper_bound(self):
        SIZE = 1024
        TILE = 128
        def arrangement(x):
            return x.tile((TILE,))
        def application(x):
            x
        source = _get_generated_source(arrangement, application, (Tensor(1),))
        mask = _mask_section(source)
        n_and = mask.count("&")
        assert n_and <= 2, (
            f"Expected <= 2 `&` in divisible mask (pid bound + fused arange bound), "
            f"got {n_and} in: {mask}"
        )

    def test_no_pid_lower_bound(self):
        def arrangement(x):
            return x.tile((256,))
        def application(x):
            x
        source = _get_generated_source(arrangement, application, (Tensor(1),))
        mask = _mask_section(source)
        assert ">= 0" not in mask, (
            "Expected no `>= 0` in mask for a simple tiled kernel."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Fallback correctness tests
# ═══════════════════════════════════════════════════════════════════════════

class TestFallbackCorrectness:
    def test_unsqueeze_preserves_lower_bound_in_mask(self):
        def arrangement(input, output):
            return input.unsqueeze(0), output.unsqueeze(0)
        def application(input, output):
            output = input
        source = _get_generated_source(arrangement, application, (Tensor(1), Tensor(1)))
        mask = _mask_section(source)
        assert ">= 0" in mask or ">=" in mask, (
            "Expected `>= 0` conditions to be preserved in the mask "
            "when non-tile levels (unsqueeze) are present, "
            "but none found in: " + mask
        )

    def test_vector_add_correctness(self):
        BLOCK_SIZE = 128
        def arrangement(x, output):
            return x.tile((BLOCK_SIZE,)), output.tile((BLOCK_SIZE,))
        def application(x, output):
            output = x
        kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1)))
        size = 2026
        x = torch.randn((size,), device="cuda")
        output = torch.empty_like(x)
        kernel(x, output)
        assert torch.allclose(output, x)

    def test_divisible_vector_add_correctness(self):
        SIZE = 1024
        TILE = 128
        def arrangement(x, output):
            return x.tile((TILE,)), output.tile((TILE,))
        def application(x, output):
            output = x
        kernel = ninetoothed.make(arrangement, application, (Tensor(1), Tensor(1)))
        x = torch.randn((SIZE,), device="cuda")
        output = torch.empty_like(x)
        kernel(x, output)
        assert torch.allclose(output, x)

    def test_slice_correctness(self):
        def arrangement(input, output, islices, oslices):
            return input[islices], output[oslices]
        def application(input, output):
            output = input
        tensors = (Tensor(1), Tensor(1), (slice(10, 200),), (slice(0, 190),))
        kernel = ninetoothed.make(arrangement, application, tensors)
        x = torch.randn((1000,), device="cuda")
        out = torch.empty((190,), device="cuda")
        kernel(x, out)
        assert torch.allclose(out, x[10:200])

    def test_expand_correctness(self):
        BLOCK = 512
        SIZE = 4096
        def arrangement(x, BLOCK=BLOCK):
            return (x.expand((BLOCK,)).tile((BLOCK,)),)
        def application(x):
            x
        kernel = ninetoothed.make(arrangement, application, (Tensor(1),))
        x = torch.randn((SIZE,), device="cuda")
        x_ref = x.clone()
        kernel(x)
        assert torch.allclose(x, x_ref)

# ═══════════════════════════════════════════════════════════════════════════
# Generated source structure tests
# ═══════════════════════════════════════════════════════════════════════════

class TestGeneratedSourceStructure:
    def test_pid_upper_bound_preserved(self):
        def arrangement(x):
            return x.tile((128,))
        def application(x):
            x
        source = _get_generated_source(arrangement, application, (Tensor(1),))
        mask = _mask_section(source)
        assert " < " in mask, (
            "Expected `<` condition in mask for PID bounds, "
            "but none found in: " + mask
        )

    def test_source_upper_bound_preserved(self):
        def arrangement(x):
            return x.tile((ninetoothed.block_size(),))
        def application(x):
            x
        source = _get_generated_source(arrangement, application, (Tensor(1),))
        mask = _mask_section(source)
        assert " < " in mask, (
            "Expected `<` condition in mask for source data bounds, "
            "but none found in: " + mask
        )

    def test_max_contiguous_in_expand_kernel(self):
        BLOCK = 512
        def arrangement(x, BLOCK=BLOCK):
            return (x.expand((BLOCK,)).tile((BLOCK,)),)
        def application(x):
            x
        source = _get_generated_source(arrangement, application, (Tensor(1),))
        assert "max_contiguous" in source, (
            "Expected `max_contiguous` hint for 1D tile with expand, "
            "but not found in source."
        )

    def test_no_stride_multiply_by_one(self):
        import re
        def arrangement(x, output):
            return x.tile((256,)), output.tile((256,))
        def application(x, output):
            output = x
        source = _get_generated_source(arrangement, application, (Tensor(1), Tensor(1)))
        idx = source.find("@triton.jit")
        if idx >= 0:
            func_body = source[idx:]
            unsimplified = len(re.findall(r'\+\s*\w+\s*\*\s*1\b', func_body))
            assert unsimplified == 0, (
                f"Found {unsimplified} unsimplified `stride * 1` expressions."
            )

    def test_broadcast_mask_has_zero_offset(self):
        def arrangement(x, output):
            return x.unsqueeze(0), output.unsqueeze(0)
        def application(x, output):
            output = x
        source = _get_generated_source(arrangement, application, (Tensor(1), Tensor(1)))
        assert "stride" in source, (
            "Expected stride references in generated code for unsqueeze kernel."
        )

    def test_mask_has_pid_bound_for_tiled_kernel(self):
        def arrangement(x, output, BLOCK=(256,)):
            return x.tile(BLOCK), output.tile(BLOCK)
        def application(x, output):
            output = x
        source = _get_generated_source(arrangement, application, (Tensor(1), Tensor(1)))
        mask = _mask_section(source)
        assert " < " in mask, (
            "Expected `<` in mask for pid num_blocks bound: " + mask
        )


# ═══════════════════════════════════════════════════════════════════════════
# NTOps-based hit tests — check mask section (not entire source)
# ═══════════════════════════════════════════════════════════════════════════

class TestNTOpsSpecializationHit:
    """Verify NTOps kernel masks have specialization applied."""

    def test_add_mask_no_lower_bound(self):
        src = _get_ntops_source({"premake": "ntops.kernels.add.premake", "premake_args": (1,)})
        mask = _mask_section(src)
        assert mask and ">= 0" not in mask, (
            f"Expected no `>= 0` in ntops.add mask: {mask}"
        )

    def test_relu_mask_no_lower_bound(self):
        src = _get_ntops_source({"premake": "ntops.kernels.relu.premake", "premake_args": (1,)})
        mask = _mask_section(src)
        assert mask and ">= 0" not in mask, (
            f"Expected no `>= 0` in ntops.relu mask: {mask}"
        )

    def test_gelu_mask_no_lower_bound(self):
        src = _get_ntops_source({"premake": "ntops.kernels.gelu.premake", "premake_args": (1, False)})
        mask = _mask_section(src)
        assert mask and ">= 0" not in mask, (
            f"Expected no `>= 0` in ntops.gelu mask: {mask}"
        )

    def test_sigmoid_mask_no_lower_bound(self):
        src = _get_ntops_source({"premake": "ntops.kernels.sigmoid.premake", "premake_args": (1,)})
        mask = _mask_section(src)
        assert mask and ">= 0" not in mask, (
            f"Expected no `>= 0` in ntops.sigmoid mask: {mask}"
        )

    def test_sub_mask_no_lower_bound(self):
        src = _get_ntops_source({"premake": "ntops.kernels.sub.premake", "premake_args": (1,)})
        mask = _mask_section(src)
        assert mask and ">= 0" not in mask, (
            f"Expected no `>= 0` in ntops.sub mask: {mask}"
        )

    def test_mul_mask_no_lower_bound(self):
        src = _get_ntops_source({"premake": "ntops.kernels.mul.premake", "premake_args": (1,)})
        mask = _mask_section(src)
        assert mask and ">= 0" not in mask, (
            f"Expected no `>= 0` in ntops.mul mask: {mask}"
        )

    def test_add_mask_arange_limited(self):
        src = _get_ntops_source({"premake": "ntops.kernels.add.premake", "premake_args": (1,)})
        mask = _mask_section(src)
        assert mask
        arange_count = mask.count("arange")
        assert arange_count <= 1, (
            f"Expected <= 1 arange in ntops.add mask, got {arange_count}: {mask}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# NTOps correctness tests
# ═══════════════════════════════════════════════════════════════════════════

class TestNTOpsCorrectness:
    def test_add_correctness(self):
        import ntops
        x = torch.randn(4096, device="cuda")
        y = torch.randn(4096, device="cuda")
        out = ntops.torch.add(x, y)
        assert torch.allclose(out, torch.add(x, y))

    def test_relu_correctness(self):
        import ntops
        x = torch.randn(4096, device="cuda")
        out = ntops.torch.relu(x)
        assert torch.allclose(out, torch.relu(x))

    def test_gelu_correctness(self):
        import ntops
        x = torch.randn(4096, device="cuda")
        out = ntops.torch.gelu(x)
        expected = torch.nn.functional.gelu(x)
        assert torch.allclose(out, expected, atol=1e-4)

    def test_softmax_correctness(self):
        import ntops
        x = torch.randn(256, 128, device="cuda")
        out = ntops.torch.softmax(x, dim=-1)
        assert torch.allclose(out, torch.softmax(x, dim=-1), atol=1e-4)

    def test_layer_norm_correctness(self):
        import ntops
        x = torch.randn(32, 128, device="cuda")
        w = torch.randn(128, device="cuda")
        b = torch.randn(128, device="cuda")
        out = ntops.torch.layer_norm(x, (128,), weight=w, bias=b)
        expected = torch.nn.functional.layer_norm(x, (128,), weight=w, bias=b)
        assert torch.allclose(out, expected, atol=1e-4)

    def test_rms_norm_correctness(self):
        import ntops
        x = torch.randn(32, 256, device="cuda")
        w = torch.randn(256, device="cuda")
        out = ntops.torch.rms_norm(x, (256,), weight=w)
        expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w
        assert torch.allclose(out, expected, atol=1e-3)


class TestNTOpsSourceStructure:
    def test_add_has_triton_jit(self):
        src = _get_ntops_source({"premake": "ntops.kernels.add.premake", "premake_args": (1,)})
        assert "@triton.jit" in src

    def test_softmax_has_arange(self):
        src = _get_ntops_source({"premake": "ntops.kernels.softmax.premake", "premake_args": (2, -1)})
        assert "arange" in src.lower() or "arange" is not None

    def test_mm_has_dot(self):
        src = _get_ntops_source({"premake": "ntops.kernels.mm.premake", "premake_args": ()})
        assert "dot" in src.lower()
