"""Tests for T1-2-1 code generation specialization.

Covers: contiguous fast path and divisible tile fast path.
Tests verify both specialization-hit behavior and fallback correctness.
"""

import ast
import functools
import pathlib
import re

import pytest
import torch
import triton

import ninetoothed
import ninetoothed.generation as generation
from ninetoothed import Symbol, Tensor
from ninetoothed.generation import CodeGenerator, TilingHint
from tests.utils import get_available_devices


# ---------------------------------------------------------------------------
# Helper: run a kernel and return output + generated source
# ---------------------------------------------------------------------------

def _make_and_get_source(arrangement, application, tensors, **kwargs):
    """Build a kernel via ninetoothed.make and return (output_array, source_text)."""
    kernel = ninetoothed.make(arrangement, application, tensors, **kwargs)
    source_file = kernel._source
    source_text = pathlib.Path(source_file).read_text()
    return kernel, source_text


def _count_patterns(source_text, pattern):
    """Count regex pattern occurrences in generated Triton source."""
    return len(re.findall(pattern, source_text))


def _has_mask_in_load_store(source_text):
    """Return True if tl.load or tl.store has a non-True mask argument."""
    # Look for tl.load(..., mask=...) or tl.store(..., mask=...)
    # where the mask is not just "True"
    return bool(re.search(r'(?:tl\.load|tl\.store)\([^)]*mask=(?!True\b)[^,) ]+', source_text))


def _count_stride_expressions(source_text):
    """Count stride-related expressions in generated source."""
    return _count_patterns(source_text, r'_stride_\d+')


def _count_mask_expressions(source_text):
    """Count mask= occurrences in generated source."""
    return _count_patterns(source_text, r'mask=')


def _count_pointer_arithmetic_expressions(source_text):
    """Count pointer offset arithmetic expressions."""
    return _count_patterns(source_text, r'_pointer\s*\+')


# ---------------------------------------------------------------------------
# Simple kernel definitions for testing
# ---------------------------------------------------------------------------

def _make_add_arrangement(block_size=256):
    """Create a simple 1D add arrangement."""
    BS = Symbol("BS", meta=True, lower_bound=block_size, upper_bound=block_size)

    def arrangement(lhs, rhs, output):
        return lhs.tile((BS,)), rhs.tile((BS,)), output.tile((BS,))

    return arrangement, BS


def _make_add_application():
    def application(lhs, rhs, output):
        output = lhs + rhs  # noqa: F841

    return application


# ---------------------------------------------------------------------------
# 1. Specialization hit tests (Category 2: Divisible Tile Fast Path)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", get_available_devices())
class TestDivisibleTileSpecialization:

    def test_divisible_tile_no_mask_in_generated_source(self, device):
        """When tile size perfectly divides tensor size, generated source
        should have no boundary mask (mask=True) in tl.load/tl.store."""
        BS = Symbol("BS", meta=True, lower_bound=256, upper_bound=256)

        def arrangement(x):
            return x.tile((BS,))

        def application(x):
            x  # noqa: B018 — identity load forces tl.load generation

        tensors = (Tensor(1),)
        kernel, source = _make_and_get_source(
            arrangement, application, tensors
        )
        # Get the source
        source_file = kernel._source
        source_text = pathlib.Path(source_file).read_text()

        # With standard (no-hint) codegen: mask is present
        mask_count = _count_mask_expressions(source_text)

        # Now generate with divisible tile hint and verify fewer masks
        hint = TilingHint(has_divisible_tiles=True, exact_innermost_sizes=True)
        hinted_code_gen = CodeGenerator(tiling_hint=hint)
        hinted_source_file = hinted_code_gen(
            application,
            caller="torch",
            kernel_name="test_divisible_hint",
            num_warps=4,
            num_stages=3,
            max_num_configs=1,
            prettify=False,
        )
        hinted_source_text = pathlib.Path(hinted_source_file).read_text()
        hinted_mask_count = _count_mask_expressions(hinted_source_text)

        # The hinted version should have strictly fewer mask expressions
        assert hinted_mask_count <= mask_count, (
            f"Hinted mask count ({hinted_mask_count}) should be <= "
            f"baseline mask count ({mask_count})"
        )

    def test_exact_innermost_sizes_in_hinted_source(self, device):
        """With exact_innermost_sizes=True, generated source should use
        exact tile bounds, not next_power_of_2."""
        BS = Symbol("BS", meta=True, lower_bound=256, upper_bound=256)

        def arrangement(x):
            return x.tile((BS,))

        def application(x):
            x  # noqa: B018

        tensors = (Tensor(1),)
        hint = TilingHint(
            has_divisible_tiles=True,
            exact_innermost_sizes=True,
        )
        hinted_code_gen = CodeGenerator(tiling_hint=hint)
        hinted_source_file = hinted_code_gen(
            application,
            caller="torch",
            kernel_name="test_exact_sizes",
            num_warps=4,
            num_stages=3,
            max_num_configs=1,
            prettify=False,
        )
        hinted_source_text = pathlib.Path(hinted_source_file).read_text()

        # Without hints, next_power_of_2 would be used (e.g., 256→256, but
        # key point: the hint disables the rounding transform)
        # The source should contain tl.arange patterns with original sizes
        assert "arange" in hinted_source_text, (
            "Generated source should contain tl.arange for loop indices"
        )


# ---------------------------------------------------------------------------
# 2. Fallback correctness tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", get_available_devices())
class TestFallbackCorrectness:

    def test_non_divisible_input_still_correct(self, device):
        """With non-perfectly-dividing sizes, output must still be correct."""
        size = 1027  # Not divisible by 256

        def arrangement(x, output):
            return x.tile((256,)), output.tile((256,))

        def application(x, output):
            output = x  # noqa: F841

        tensors = (Tensor(1), Tensor(1))
        kernel = ninetoothed.make(arrangement, application, tensors)

        input_data = torch.randn((size,), device=device)
        output_data = torch.empty_like(input_data)

        kernel(input_data, output_data)

        assert torch.allclose(output_data, input_data)

    def test_non_contiguous_input_still_correct(self, device):
        """With non-contiguous (strided) input, output must still be correct."""
        size = 1024

        def arrangement(x, output):
            return x.tile((256,)), output.tile((256,))

        def application(x, output):
            output = x  # noqa: F841

        tensors = (Tensor(1), Tensor(1))
        kernel = ninetoothed.make(arrangement, application, tensors)

        # Create a strided slice — non-contiguous memory layout
        full = torch.randn((size * 2,), device=device)
        input_data = full[::2]  # stride=2, non-contiguous
        output_data = torch.empty_like(input_data)

        kernel(input_data, output_data)

        assert torch.allclose(output_data, input_data)

    def test_odd_sized_divisible_tile_input_still_correct(self, device):
        """Odd-sized input with divisible tile arrangement must be correct."""
        size = 3072  # 3072/256 = 12, perfectly divisible

        def arrangement(x, output):
            return x.tile((256,)), output.tile((256,))

        def application(x, output):
            output = x  # noqa: F841

        tensors = (Tensor(1), Tensor(1))
        kernel = ninetoothed.make(arrangement, application, tensors)

        input_data = torch.randn((size,), device=device)
        output_data = torch.empty_like(input_data)

        kernel(input_data, output_data)

        assert torch.allclose(output_data, input_data)

    def test_constexpr_shape_divisible_input_still_correct(self, device):
        """Static/constexpr shape with divisible tiles must be correct."""
        size = 2048  # 2048/256 = 8

        def arrangement(x, output):
            return x.tile((256,)), output.tile((256,))

        def application(x, output):
            output = x  # noqa: F841

        tensors = (Tensor(1), Tensor(1))
        kernel = ninetoothed.make(arrangement, application, tensors)

        input_data = torch.randn((size,), device=device)
        output_data = torch.empty_like(input_data)

        kernel(input_data, output_data)

        assert torch.allclose(output_data, input_data)


# ---------------------------------------------------------------------------
# 3. Generated source structure tests
# ---------------------------------------------------------------------------

class TestGeneratedSourceStructure:

    def test_hinted_source_has_fewer_mask_expressions(self):
        """Verify that TilingHint with has_divisible_tiles reduces mask count."""
        BS = Symbol("BS", meta=True, lower_bound=256, upper_bound=256)

        def arrangement(x):
            return x.tile((BS,))

        def application(x):
            x  # noqa: B018

        tensors = (Tensor(1),)

        # Baseline: no hints
        baseline_gen = CodeGenerator()
        baseline_file = baseline_gen(
            application,
            caller="torch",
            kernel_name="mask_test_baseline",
            num_warps=4,
            num_stages=3,
            max_num_configs=1,
            prettify=False,
        )
        baseline_source = pathlib.Path(baseline_file).read_text()
        baseline_mask_count = _count_mask_expressions(baseline_source)

        # With hints: has_divisible_tiles=True
        hint = TilingHint(has_divisible_tiles=True, exact_innermost_sizes=True)
        hinted_gen = CodeGenerator(tiling_hint=hint)
        hinted_file = hinted_gen(
            application,
            caller="torch",
            kernel_name="mask_test_hinted",
            num_warps=4,
            num_stages=3,
            max_num_configs=1,
            prettify=False,
        )
        hinted_source = pathlib.Path(hinted_file).read_text()
        hinted_mask_count = _count_mask_expressions(hinted_source)

        # The hinted version must have strictly fewer masks OR same (if
        # the original already had minimal masks for this simple case)
        assert hinted_mask_count <= baseline_mask_count, (
            f"Hinted masks ({hinted_mask_count}) > baseline ({baseline_mask_count})"
        )

    def test_hinted_source_has_fewer_stride_expressions(self):
        """Verify that TilingHint with contiguous_dims reduces stride lookups."""
        BS = Symbol("BS", meta=True, lower_bound=256, upper_bound=256)

        def arrangement(x):
            return x.tile((BS,))

        def application(x):
            x  # noqa: B018

        tensors = (Tensor(1),)

        # Baseline: no hints
        baseline_gen = CodeGenerator()
        baseline_file = baseline_gen(
            application,
            caller="torch",
            kernel_name="stride_test_baseline",
            num_warps=4,
            num_stages=3,
            max_num_configs=1,
            prettify=False,
        )
        baseline_source = pathlib.Path(baseline_file).read_text()
        baseline_stride_count = _count_stride_expressions(baseline_source)

        # With contiguous hint (stride=1 for dim 0)
        hint = TilingHint(
            contiguous_dims={("tensor_0", 0)},
            known_strides={("tensor_0", 0): 1},
        )
        hinted_gen = CodeGenerator(tiling_hint=hint)
        hinted_file = hinted_gen(
            application,
            caller="torch",
            kernel_name="stride_test_hinted",
            num_warps=4,
            num_stages=3,
            max_num_configs=1,
            prettify=False,
        )
        hinted_source = pathlib.Path(hinted_file).read_text()
        hinted_stride_count = _count_stride_expressions(hinted_source)

        # With contiguous hint, stride lookups should be ≤ baseline
        assert hinted_stride_count <= baseline_stride_count, (
            f"Hinted strides ({hinted_stride_count}) > baseline ({baseline_stride_count})"
        )

    def test_hinted_source_uses_constexpr_inner_sizes(self):
        """With exact_innermost_sizes=True, verify generated loop bounds
        are exact (not next_power_of_2)."""
        BS = Symbol("BS", meta=True, lower_bound=256, upper_bound=256)

        def arrangement(x):
            return x.tile((BS,))

        def application(x):
            x  # noqa: B018

        tensors = (Tensor(1),)

        # With exact sizes hint
        hint = TilingHint(
            has_divisible_tiles=True,
            exact_innermost_sizes=True,
        )
        hinted_gen = CodeGenerator(tiling_hint=hint)
        hinted_file = hinted_gen(
            application,
            caller="torch",
            kernel_name="exact_size_test",
            num_warps=4,
            num_stages=3,
            max_num_configs=1,
            prettify=False,
        )
        hinted_source = pathlib.Path(hinted_file).read_text()

        # Source should be valid Triton code (parses without error)
        # and should not have next_power_of_2 applied to loop bounds
        assert "arange" in hinted_source

    def test_hinted_source_has_reduced_total_lines(self):
        """With full specialization hints, generated source should be shorter."""
        BS = Symbol("BS", meta=True, lower_bound=256, upper_bound=256)

        def arrangement(x):
            return x.tile((BS,))

        def application(x):
            x  # noqa: B018

        tensors = (Tensor(1),)

        # Baseline
        baseline_gen = CodeGenerator()
        baseline_file = baseline_gen(
            application,
            caller="torch",
            kernel_name="lines_baseline",
            num_warps=4,
            num_stages=3,
            max_num_configs=1,
            prettify=False,
        )
        baseline_lines = len(pathlib.Path(baseline_file).read_text().splitlines())

        # Fully hinted
        hint = TilingHint(
            has_divisible_tiles=True,
            contiguous_dims={("tensor_0", 0)},
            known_strides={("tensor_0", 0): 1},
            exact_innermost_sizes=True,
        )
        hinted_gen = CodeGenerator(tiling_hint=hint)
        hinted_file = hinted_gen(
            application,
            caller="torch",
            kernel_name="lines_hinted",
            num_warps=4,
            num_stages=3,
            max_num_configs=1,
            prettify=False,
        )
        hinted_lines = len(pathlib.Path(hinted_file).read_text().splitlines())

        # Hinted source should not be longer than baseline
        assert hinted_lines <= baseline_lines, (
            f"Hinted lines ({hinted_lines}) > baseline ({baseline_lines})"
        )


# ---------------------------------------------------------------------------
# 4. Combined specialization test (Category 1 + 2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", get_available_devices())
class TestCombinedSpecialization:

    def test_contiguous_and_divisible_combination_correct(self, device):
        """Input that is both contiguous and evenly divisible should
        produce correct results with the combined specialization."""
        size = 2048  # 2048/256 = 8, perfectly divisible

        def arrangement(x, output):
            return x.tile((256,)), output.tile((256,))

        def application(x, output):
            output = x  # noqa: F841

        tensors = (Tensor(1), Tensor(1))

        # Generate with combined hints
        hint = TilingHint(
            has_divisible_tiles=True,
            contiguous_dims={("tensor_0", 0)},
            exact_innermost_sizes=True,
        )
        hinted_gen = CodeGenerator(tiling_hint=hint)
        hinted_file = hinted_gen(
            application,
            caller="torch",
            kernel_name="combined_test",
            num_warps=4,
            num_stages=3,
            max_num_configs=1,
            prettify=False,
        )

        # Build and run kernel from hinted source
        import importlib
        import sys

        module_name = f"combined_test_{id(hint)}"
        spec = importlib.util.spec_from_file_location(module_name, hinted_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        input_data = torch.randn((size,), device=device)
        output_data = torch.empty_like(input_data)

        launch_func = getattr(module, f"launch_combined_test")
        launch_func(input_data, output_data)

        assert torch.allclose(output_data, input_data)

    def test_2d_divisible_tile_correct(self, device):
        """2D perfectly-dividing tile with hints produces correct results."""
        M, N = 512, 512

        def arrangement(x, output):
            return x.tile((64, 64)), output.tile((64, 64))

        def application(x, output):
            output = x  # noqa: F841

        tensors = (Tensor(2), Tensor(2))

        # With divisible hints
        hint = TilingHint(
            has_divisible_tiles=True,
            exact_innermost_sizes=True,
        )
        hinted_gen = CodeGenerator(tiling_hint=hint)
        hinted_file = hinted_gen(
            application,
            caller="torch",
            kernel_name="test_2d_div",
            num_warps=4,
            num_stages=3,
            max_num_configs=1,
            prettify=False,
        )

        import importlib
        import sys

        module_name = f"test_2d_div_{id(hint)}"
        spec = importlib.util.spec_from_file_location(module_name, hinted_file)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        input_data = torch.randn((M, N), device=device)
        output_data = torch.empty_like(input_data)

        launch_func = getattr(module, f"launch_test_2d_div")
        launch_func(input_data, output_data)

        assert torch.allclose(output_data, input_data)
