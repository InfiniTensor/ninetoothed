"""Tests for the T1-2-1 specialization enhancements.

Each test exercises one of the three required categories:

* specialization-hit: an input that the dispatcher routes to the
  specialized variant, and we assert the generated source for that
  variant is more compact than the fallback.
* fallback-correctness: an input that the dispatcher routes to the
  generic variant, and we assert numerical results match the
  reference.
* generated-source-structure: directly inspects the generated ``.cpp``
  / ``.py`` files for the kernel under ``CACHE_DIR``, counting mask /
  stride / pointer expressions across variants.
"""

import functools
import pathlib
import re

import pytest
import torch

import ninetoothed
import ninetoothed.generation
from ninetoothed import Tensor
from tests.utils import get_available_devices


_OUTPUT_DIR = ninetoothed.generation.CACHE_DIR


def _make_kernel_name(prefix):
    """Return a fresh kernel name. Test isolation across runs."""

    _make_kernel_name._counter += 1
    return f"{prefix}_spec_{_make_kernel_name._counter}"


_make_kernel_name._counter = 0


def _read_variant_sources(kernel_name):
    """Return a dict mapping variant_suffix → triton kernel python source.

    ``_aot`` writes per-variant Python sources to
    ``{kernel_name}.{variant_suffix}.py`` alongside each compiled .cpp.
    """

    output_dir = pathlib.Path(_OUTPUT_DIR)
    sources = {}

    for py in output_dir.glob(f"{kernel_name}.*.py"):
        suffix = py.name[len(kernel_name) + 1 : -len(".py")]
        sources[suffix] = py.read_text()

    return sources


def _read_dispatcher(kernel_name):
    return (pathlib.Path(_OUTPUT_DIR) / f"{kernel_name}.cpp").read_text()


def _count_mask_arguments(source):
    """Count occurrences of ``mask=`` in tl.load / tl.store calls."""

    return len(re.findall(r"\bmask\s*=", source))


def _count_stride_multiplications(source):
    """Count expressions that multiply by a ``..._stride_N`` name."""

    return len(re.findall(r"\*\s*\w+_stride_\d+", source))


def _count_pointer_arith(source):
    """Count add expressions involving a tensor ``..._pointer`` name."""

    return len(re.findall(r"\w+_pointer\s*\+", source))


# ---------------------------------------------------------------------------
# Specialization hit + fallback correctness
# ---------------------------------------------------------------------------


def _arrange_add(input, other, output):
    def _arrange(tensor):
        return tensor.tile((256,))

    return _arrange(input), _arrange(other), _arrange(output)


def _apply_add(input, other, output):
    output = input + other  # noqa: F841


def _build_add_kernel(name, device):
    tensors = tuple(Tensor(1, dtype=ninetoothed.float32) for _ in range(3))

    return ninetoothed.make(
        _arrange_add,
        _apply_add,
        tensors,
        caller=device,
        kernel_name=name,
        output_dir=_OUTPUT_DIR,
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_specialization_hit_add_divisible(device):
    """A size that is divisible by BLOCK_SIZE (256) routes to the
    specialized variant; the variant source must omit mask=."""

    name = _make_kernel_name("add_hit_div")
    kernel = _build_add_kernel(name, device)

    size = 4096  # exactly divisible by 256

    x = torch.randn((size,), dtype=torch.float32, device=device)
    y = torch.randn((size,), dtype=torch.float32, device=device)
    z = torch.empty_like(x)

    kernel(x, y, z)

    assert torch.allclose(z, x + y)

    sources = _read_variant_sources(name)
    specialized = [
        src
        for suffix, src in sources.items()
        if "divisibility_16_16_16" in suffix
        and "size_i32_stride_i32" in suffix
    ]
    assert specialized, "no fully-divisibility-specialized variant emitted"

    for src in specialized:
        assert _count_mask_arguments(src) == 0, (
            f"specialized add variant still contains mask=:\n{src}"
        )


@pytest.mark.parametrize("device", get_available_devices())
def test_specialization_fallback_add_non_divisible(device):
    """A non-divisible size routes to the fallback (or a partially
    specialized) variant; correctness must hold and fallback source
    must still contain mask=."""

    name = _make_kernel_name("add_fb_non_div")
    kernel = _build_add_kernel(name, device)

    size = 4097  # NOT divisible by 256

    x = torch.randn((size,), dtype=torch.float32, device=device)
    y = torch.randn((size,), dtype=torch.float32, device=device)
    z = torch.empty_like(x)

    kernel(x, y, z)

    assert torch.allclose(z, x + y)

    sources = _read_variant_sources(name)
    fallback = [
        src
        for suffix, src in sources.items()
        if "size_i64_stride_i64" in suffix
    ]
    assert fallback, "no fallback variant emitted"

    for src in fallback:
        assert _count_mask_arguments(src) > 0, (
            "fallback add variant unexpectedly has no mask= — "
            "specialization may have leaked into fallback"
        )


# ---------------------------------------------------------------------------
# Contiguous fast path (stride folding)
# ---------------------------------------------------------------------------


def _build_copy_kernel(name, device):
    def _arrange(input, output):
        return input.tile((64, 64)), output.tile((64, 64))

    def _apply(input, output):
        output = input  # noqa: F841

    tensors = (
        Tensor(2, dtype=ninetoothed.float32),
        Tensor(2, dtype=ninetoothed.float32),
    )

    return ninetoothed.make(
        _arrange,
        _apply,
        tensors,
        caller=device,
        kernel_name=name,
        output_dir=_OUTPUT_DIR,
    )


@pytest.mark.parametrize("device", get_available_devices())
def test_specialization_hit_copy_contiguous(device):
    """Contiguous row-major 2-D copy hits the contiguity-specialized
    variant; that variant must not contain ``* ..._stride_N``
    expressions for the contiguous innermost stride."""

    name = _make_kernel_name("copy_hit_cont")
    kernel = _build_copy_kernel(name, device)

    m, n = 512, 512  # both divisible by 64; row-major default stride

    x = torch.randn((m, n), dtype=torch.float32, device=device)
    y = torch.empty_like(x)

    kernel(x, y)

    assert torch.allclose(y, x)

    sources = _read_variant_sources(name)

    # Maximally specialized variant for 2 row-major 2D tensors:
    #   divisibility on innermost dim only (per-tensor) → "_1_16_1_16"
    #   contiguity on innermost dim only             → "_0_1_0_1"
    specialized = [
        src
        for suffix, src in sources.items()
        if "_size_i32_stride_i32" in suffix
        and re.search(r"divisibility_1_16_1_16_contiguity_0_1_0_1_", suffix)
    ]
    fallback = [
        src
        for suffix, src in sources.items()
        if "size_i64_stride_i64" in suffix
    ]
    assert specialized, (
        f"no fully-contiguous-specialized variant emitted; "
        f"available suffixes: {list(sources)}"
    )
    assert fallback, "no fallback variant emitted"

    for src in specialized:
        assert _count_stride_multiplications(src) < _count_stride_multiplications(
            fallback[0]
        ), (
            "contiguity-specialized copy variant did not reduce stride "
            "multiplications relative to fallback"
        )


@pytest.mark.parametrize("device", get_available_devices())
def test_specialization_fallback_copy_strided(device):
    """A transposed (non-contiguous innermost stride) input falls back
    to the generic variant; correctness must hold."""

    name = _make_kernel_name("copy_fb_strided")
    kernel = _build_copy_kernel(name, device)

    m, n = 512, 512

    # Use a transposed view so strides are NOT (n, 1) anymore.
    base = torch.randn((n, m), dtype=torch.float32, device=device)
    x = base.T.contiguous().T  # forces non-contiguous innermost
    y = torch.empty_like(x)
    if x.stride(-1) == 1:
        # Fallback platform may have realigned strides; create strided manually.
        x = torch.randn((m, n * 2), dtype=torch.float32, device=device)[:, ::2]
        y = torch.empty_like(x).contiguous()

    kernel(x, y)

    assert torch.allclose(y, x)


# ---------------------------------------------------------------------------
# Generated source structure metrics (≥2 cases)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", get_available_devices())
def test_generated_source_mask_reduction_add(device):
    """Source-structure assertion: the specialized add variant
    contains strictly fewer mask expressions than the fallback."""

    name = _make_kernel_name("add_metric")
    _build_add_kernel(name, device)

    sources = _read_variant_sources(name)
    fallback = next(
        src for suffix, src in sources.items() if "size_i64_stride_i64" in suffix
    )
    specialized = [
        src
        for suffix, src in sources.items()
        if "divisibility_16_16_16" in suffix
        and "size_i32_stride_i32" in suffix
    ]
    assert specialized

    fallback_masks = _count_mask_arguments(fallback)

    for src in specialized:
        assert _count_mask_arguments(src) < fallback_masks


@pytest.mark.parametrize("device", get_available_devices())
def test_generated_source_stride_reduction_copy(device):
    """Source-structure assertion: the contiguity-specialized copy
    variant contains strictly fewer ``* stride`` expressions than the
    fallback."""

    name = _make_kernel_name("copy_metric")
    _build_copy_kernel(name, device)

    sources = _read_variant_sources(name)
    fallback = next(
        src for suffix, src in sources.items() if "size_i64_stride_i64" in suffix
    )
    specialized = [
        src
        for suffix, src in sources.items()
        if "_size_i32_stride_i32" in suffix
        and re.search(r"divisibility_1_16_1_16_contiguity_0_1_0_1_", suffix)
    ]
    assert specialized

    fallback_strides = _count_stride_multiplications(fallback)

    for src in specialized:
        assert _count_stride_multiplications(src) < fallback_strides


# ---------------------------------------------------------------------------
# Dispatcher safety
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device", get_available_devices())
def test_dispatcher_uses_strict_divisibility_check(device):
    """The dispatcher must check ``% BLOCK_SIZE`` for tensors whose
    innermost tile size is statically known, not the old ``% 16``."""

    name = _make_kernel_name("add_dispatch")
    _build_add_kernel(name, device)

    dispatcher = _read_dispatcher(name)

    # BLOCK_SIZE was 256, threshold should be 256 (since 256 > 16)
    assert "shape[0] % 256 == 0" in dispatcher, (
        f"dispatcher did not strengthen divisibility check:\n{dispatcher}"
    )
