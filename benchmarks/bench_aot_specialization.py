"""T1-2-1 baseline/submitted benchmark with actual dispatch and source diagnostics."""

import argparse
import ast
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

# Test-only dispatcher trace. It records the selected variant but does not alter guards.
os.environ.setdefault("NINETOOTHED_DISPATCH_TRACE", "1")

SOURCE_FILES = [
    "src/ninetoothed/aot.py",
    "src/ninetoothed/generation.py",
]

TILE_SHAPE_PROFILES = {
    1: {
        # 1D: several block widths, including small and very wide tiles.
        "tiny": (64,),
        "narrow": (128,),
        "default": (256,),
        "wide": (512,),
        "xwide": (1024,),
    },
    2: {
        # 2D: square, row-skewed, column-skewed, and small/large blocks.
        "tiny_square": (8, 8),
        "square": (16, 16),
        "large_square": (32, 32),
        "row_major": (8, 32),
        "row_wide": (4, 64),
        "col_major": (32, 8),
        "col_tall": (64, 4),
    },
    3: {
        # 3D: balanced, small, cube-like, and skewed tiles.
        "small": (2, 4, 8),
        "balanced": (4, 8, 8),
        "cubeish": (8, 8, 4),
        "channel_heavy": (2, 8, 16),
        "depth_heavy": (8, 4, 8),
        "flat_xy": (1, 16, 16),
        "slab_z": (16, 4, 4),
    },
}

# Explicit None/default coverage:
# tile_profile=None should behave like the public default path, not like a
# separate hand-picked hidden profile.
DEFAULT_TILE_PROFILES = {
    1: "default",
    2: "square",
    3: "balanced",
}

# Full sweep profiles used by source diagnostics and the expanded benchmark.
# None is intentionally included for each dimension.
TILE_SWEEP_PROFILES = {
    1: [None, "tiny", "narrow", "default", "wide", "xwide"],
    2: [
        None,
        "tiny_square",
        "square",
        "large_square",
        "row_major",
        "row_wide",
        "col_major",
        "col_tall",
    ],
    3: [
        None,
        "small",
        "balanced",
        "cubeish",
        "channel_heavy",
        "depth_heavy",
        "flat_xy",
        "slab_z",
    ],
}

# Keep fallback runtime coverage smaller than contiguous sweep to avoid making
# every benchmark run too long.  Source diagnostics still cover all profiles.
FALLBACK_TILE_SWEEP_PROFILES = {
    # One None/default-path fallback per dimension is enough for runtime.
    # Full fallback-like shape evidence remains in source diagnostics.
    1: [None, "default"],
    2: [None],
    3: [None],
}

# Compile-light runtime subset:
# Keep None/default-path coverage and only a few simple runtime profiles.
# Full TILE_SWEEP_PROFILES is still used by source diagnostics, so coverage
# evidence remains broad without forcing every runtime run to compile every tile.
RUNTIME_TILE_SWEEP_PROFILES = {
    # Runtime must stay compile-light.  Heavy/odd tile profiles are still
    # covered by TILE_SWEEP_PROFILES in source diagnostics below.
    1: [None, "default", "wide"],
    2: [None, "square"],
    3: [None, "balanced"],
}


def _tile_profile_label(profile):
    return "none" if profile is None else profile


def _tile_shape_for_ndim(ndim, profile=None):
    profiles = TILE_SHAPE_PROFILES.get(ndim)
    if profiles is None:
        raise ValueError(f"Unsupported ndim: {ndim}")

    if profile is None:
        profile = DEFAULT_TILE_PROFILES[ndim]

    if profile not in profiles:
        raise ValueError(
            f"Unsupported tile profile for {ndim}D: {profile}. "
            f"Available profiles: {sorted(profiles)}"
        )

    return profiles[profile]


def _make_add_components(ndim, dtype_nt=None, tile_profile=None):
    import ninetoothed
    from ninetoothed import Tensor

    if dtype_nt is None:
        dtype_nt = ninetoothed.float32

    tile_shape = _tile_shape_for_ndim(ndim, tile_profile)

    def arrangement(input, other, output):
        return (
            input.tile(tile_shape),
            other.tile(tile_shape),
            output.tile(tile_shape),
        )

    def application(input, other, output):
        output = input + other  # noqa: F841

    tensors = tuple(Tensor(ndim, dtype=dtype_nt) for _ in range(3))
    types = arrangement(*tensors)
    params = application.__code__.co_varnames[: application.__code__.co_argcount]
    application.__annotations__ = dict(zip(params, types))

    return arrangement, application, tensors


def _make_add_application(ndim, tile_profile=None):
    _, application, _ = _make_add_components(ndim, tile_profile=tile_profile)
    return application


def _make_scalar_components():
    import ninetoothed
    from ninetoothed import Tensor

    def arrangement(input, scale, output):
        return input.tile((256,)), scale, output.tile((256,))

    def application(input, scale, output):
        output = input * scale  # noqa: F841

    tensors = (
        Tensor(1, dtype=ninetoothed.float32),
        Tensor(0, dtype=ninetoothed.float32),
        Tensor(1, dtype=ninetoothed.float32),
    )
    types = arrangement(*tensors)
    params = application.__code__.co_varnames[: application.__code__.co_argcount]
    application.__annotations__ = dict(zip(params, types))

    return arrangement, application, tensors


def _make_scalar_application():
    _, application, _ = _make_scalar_components()
    return application


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _skip_if_no_cuda():
    if _cuda_available():
        return

    import pytest
    pytest.skip("CUDA is required for NineToothed/Triton code generation tests")


def _has_int64_fallback_cpp(names):
    return any(
        ("size_i64_stride_i64.cpp" in name)
        or ("size_int64_stride_int64.cpp" in name)
        or ("size_Int64_stride_Int64.cpp" in name)
        or ("stride_i64" in name)
        or ("stride_int64" in name)
        for name in names
    )


def _score_hint(speedup):
    if speedup >= 1.10:
        return "full"
    if speedup >= 1.00:
        return "partial"
    if speedup >= 0.95:
        return "30%"
    return "0"


def _make_tiling_hint(variant_name):
    """Build a TilingHint for diagnostic source inspection only.

    This is NOT used to compute a local generated-code score.  It only lets the
    benchmark print mask/stride/pointer expression counts as debug evidence.
    """
    import ninetoothed.generation

    TilingHint = getattr(ninetoothed.generation, "TilingHint", None)
    if TilingHint is None:
        return None

    if variant_name == "flatten_contiguous_divisible":
        try:
            return TilingHint(
                kind="flatten_contiguous_divisible",
                flatten_contiguous=True,
                divisible_tile=True,
            )
        except TypeError:
            return TilingHint(
                has_divisible_tiles=True,
                exact_innermost_sizes=True,
            )

    if variant_name == "flatten_contiguous_masked":
        try:
            return TilingHint(
                kind="flatten_contiguous_masked",
                flatten_contiguous=True,
                divisible_tile=False,
            )
        except TypeError:
            return TilingHint()

    if variant_name in ("fallback", "fallback_non_contiguous", "fallback_scalar_arg"):
        try:
            return TilingHint(kind="fallback")
        except TypeError:
            return TilingHint()

    try:
        return TilingHint(kind="legacy")
    except TypeError:
        return TilingHint()


def _read_generated_triton_source_for_diagnostics(application, kernel_name, variant_name):
    """Read CodeGenerator-emitted Triton Python source for diagnostics only.

    This intentionally does NOT inspect triton.tools.compile generated C/C++
    wrappers.  The official hidden generated-code metric is evaluated by the
    grader.  Local counts printed by this benchmark are only explanatory.
    """
    import ninetoothed.generation

    generator = ninetoothed.generation.CodeGenerator()
    kwargs = dict(
        caller="cuda",
        kernel_name=f"{kernel_name}_metric_src",
        num_warps=4,
        num_stages=3,
        max_num_configs=None,
        prettify=False,
    )

    hint = _make_tiling_hint(variant_name)

    try:
        if hint is not None:
            path = generator(application, tiling_hint=hint, **kwargs)
        else:
            path = generator(application, **kwargs)
    except TypeError:
        path = generator(application, **kwargs)

    return pathlib.Path(path).read_text(errors="ignore")


def _count_regex(text, pattern):
    import re
    return len(re.findall(pattern, text))


def _triton_source_diagnostic_counts(application, kernel_name, variant_name):
    text = _read_generated_triton_source_for_diagnostics(
        application, kernel_name, variant_name
    )

    kernel_param_names = []
    try:
        tree = ast.parse(text)
        expected_name = f"{kernel_name}_metric_src"
        kernel_def = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == expected_name
            ),
            None,
        )
        if kernel_def is not None:
            kernel_param_names = [arg.arg for arg in kernel_def.args.args]
    except SyntaxError:
        kernel_param_names = []

    return {
        "source_bytes": len(text.encode("utf-8")),
        "source_line_count": len(text.splitlines()),
        "kernel_param_count": len(kernel_param_names),
        "stride_param_count": sum(
            1 for name in kernel_param_names if "stride" in name
        ),
        "size_param_count": sum(
            1 for name in kernel_param_names if "size" in name or "shape" in name
        ),
        "mask_expr_count": (
            _count_regex(text, r"\bmask\s*=")
            + _count_regex(text, r"\bmask\b")
            + _count_regex(text, r"\bwhere\b")
        ),
        "stride_expr_count": (
            _count_regex(text, r"\bstride\b")
            + _count_regex(text, r"\bstrides\b")
            + _count_regex(text, r"_stride")
        ),
        "pointer_expr_count": (
            _count_regex(text, r"_pointers")
            + _count_regex(text, r"\+\s*[A-Za-z_][A-Za-z0-9_]*")
            + _count_regex(text, r"\*\s*[A-Za-z_][A-Za-z0-9_]*")
        ),
    }


def _safe_reduction(baseline_count, submitted_count):
    if baseline_count <= 0:
        return 0.0
    return (baseline_count - submitted_count) / baseline_count


def _repo_root():
    return pathlib.Path(__file__).resolve().parents[1]


def _run_cmd(cmd, *, cwd=None, env=None, capture=False):
    if capture:
        return subprocess.check_output(cmd, cwd=cwd, env=env, text=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _clear_generated_cache():
    for cache in (pathlib.Path("/root/.ninetoothed"), pathlib.Path.home() / ".ninetoothed"):
        if not cache.exists():
            continue

        for pattern in (
            "bench_speedup_*.cpp",
            "bench_speedup_*.so",
            "bench_speedup_*.h",
            "bench_speedup_*.c",
            "bench_speedup_*_metric_src.py",
        ):
            for path in cache.glob(pattern):
                path.unlink(missing_ok=True)


def _run_worker_subprocess(root, tag, output_json):
    env = os.environ.copy()

    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = "src" if not old_pythonpath else f"src:{old_pythonpath}"

    cmd = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        "--worker",
        "--tag",
        tag,
        "--output",
        str(output_json),
    ]

    _run_cmd(cmd, cwd=root, env=env)


def _benchmark_driver():
    """Benchmark committed HEAD against master/origin/master.

    Original version compared uncommitted working-tree modifications by stashing
    SOURCE_FILES.  This version is commit-friendly:
      - baseline: checkout SOURCE_FILES from NT_BENCH_BASE, default origin/master
      - submitted: checkout SOURCE_FILES from NT_BENCH_SUBMITTED, default HEAD
      - benchmark file itself is kept from the current branch, so it can exist
        even if master does not have this test file.
    """
    root = _repo_root()

    base_ref = os.environ.get("NT_BENCH_BASE", "origin/master")
    submitted_ref = os.environ.get("NT_BENCH_SUBMITTED", "HEAD")

    # Verify refs exist early, with clearer errors than checkout failures.
    _run_cmd(["git", "rev-parse", "--verify", base_ref], cwd=root, capture=True)
    _run_cmd(["git", "rev-parse", "--verify", submitted_ref], cwd=root, capture=True)

    changed_files = _run_cmd(
        ["git", "diff", "--name-only", f"{base_ref}...{submitted_ref}", "--", *SOURCE_FILES],
        cwd=root,
        capture=True,
    )

    if not changed_files.strip():
        raise RuntimeError(
            f"没有检测到 {base_ref}...{submitted_ref} 中 "
            "src/ninetoothed/aot.py 或 src/ninetoothed/generation.py 的修改。"
        )

    # We will overwrite SOURCE_FILES twice with git checkout, so do not allow
    # local uncommitted source changes to be accidentally lost.
    dirty_source_files = _run_cmd(
        ["git", "status", "--short", "--", *SOURCE_FILES],
        cwd=root,
        capture=True,
    )
    if dirty_source_files.strip():
        raise RuntimeError(
            "检测到 aot.py/generation.py 有未提交修改。"
            "请先 commit/stash 后再运行 master 对比 benchmark。\n"
            f"{dirty_source_files}"
        )

    tmp_dir = pathlib.Path("/tmp") / f"nt_speedup_{int(time.time())}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    baseline_json = tmp_dir / "baseline.json"
    submitted_json = tmp_dir / "submitted.json"

    restored_submitted = False

    try:
        print(f"=== Phase 1: checkout {base_ref} source files and run baseline ===")

        _run_cmd(
            ["git", "checkout", base_ref, "--", *SOURCE_FILES],
            cwd=root,
        )

        _clear_generated_cache()
        _run_worker_subprocess(root, "baseline", baseline_json)

        print(f"\n=== Phase 2: checkout {submitted_ref} source files and run submitted ===")

        _run_cmd(
            ["git", "checkout", submitted_ref, "--", *SOURCE_FILES],
            cwd=root,
        )
        restored_submitted = True

        _clear_generated_cache()
        _run_worker_subprocess(root, "submitted", submitted_json)

        _compare_benchmark_results(baseline_json, submitted_json, root)

    finally:
        if not restored_submitted:
            print("尝试恢复 submitted 修改...")
            try:
                _run_cmd(
                    ["git", "checkout", submitted_ref, "--", *SOURCE_FILES],
                    cwd=root,
                )
            except Exception as exc:
                print("WARNING: 自动恢复失败，请手动恢复 SOURCE_FILES。", exc)

def _compare_benchmark_results(baseline_json, submitted_json, root):
    baseline = json.loads(pathlib.Path(baseline_json).read_text())
    submitted = json.loads(pathlib.Path(submitted_json).read_text())

    base = {item["scenario"]: item for item in baseline["results"]}
    sub = {item["scenario"]: item for item in submitted["results"]}

    rows = []
    speedups = []
    hit_speedups = []
    fallback_speedups = []
    speedups_by_ndim = {1: [], 2: [], 3: []}
    hit_speedups_by_ndim = {1: [], 2: [], 3: []}
    fallback_speedups_by_ndim = {1: [], 2: [], 3: []}

    print("\n=== Runtime Median Summary ===")
    print(
        f"{'scenario':<42} "
        f"{'base_med':>10} "
        f"{'sub_med':>10} "
        f"{'speedup':>9} "
        f"{'correct':>7} "
        f"{'hit':>6} "
        f"{'expected':>29} "
        f"{'actual':>29} "
        f"{'match':>7} "
        f"{'mask':>9} "
        f"{'stride':>9} "
        f"{'ptr':>9}"
    )

    for name in base:
        if name not in sub:
            print("missing submitted result:", name)
            continue

        b = base[name]
        t = sub[name]

        if not t.get("correct", False):
            raise AssertionError(f"Benchmark correctness failed: {name}")
        if t.get("dispatch_match") is not True:
            raise AssertionError(
                f"Benchmark dispatch failed: {name}: "
                f"expected={t.get('expected_variant')}, "
                f"actual={t.get('actual_variant')}"
            )

        b_ms = b["runtime_ms"]
        t_ms = t["runtime_ms"]
        speedup = b_ms / t_ms if t_ms > 0 else 0.0
        speedups.append(speedup)

        ndim = t.get("ndim")
        if ndim in speedups_by_ndim:
            speedups_by_ndim[ndim].append(speedup)

        if t.get("specialization_hit", False):
            hit_speedups.append(speedup)
            if ndim in hit_speedups_by_ndim:
                hit_speedups_by_ndim[ndim].append(speedup)
        else:
            fallback_speedups.append(speedup)
            if ndim in fallback_speedups_by_ndim:
                fallback_speedups_by_ndim[ndim].append(speedup)

        mask_reduction = _safe_reduction(
            b.get("mask_expr_count", 0), t.get("mask_expr_count", 0)
        )
        stride_reduction = _safe_reduction(
            b.get("stride_expr_count", 0), t.get("stride_expr_count", 0)
        )
        pointer_reduction = _safe_reduction(
            b.get("pointer_expr_count", 0), t.get("pointer_expr_count", 0)
        )

        row = {
            "scenario": name,
            "baseline_runtime_ms": b_ms,
            "submitted_runtime_ms": t_ms,
            "baseline_runtime_p25_ms": b.get("runtime_p25_ms"),
            "baseline_runtime_p75_ms": b.get("runtime_p75_ms"),
            "submitted_runtime_p25_ms": t.get("runtime_p25_ms"),
            "submitted_runtime_p75_ms": t.get("runtime_p75_ms"),
            "baseline_runtime_samples_ms": b.get("runtime_samples_ms", []),
            "submitted_runtime_samples_ms": t.get("runtime_samples_ms", []),
            "speedup": speedup,
            "runtime_score_hint": _score_hint(speedup),
            "correct": t["correct"],
            "expected_specialization_hit": t.get(
                "expected_specialization_hit", False
            ),
            "specialization_hit": t["specialization_hit"],
            "expected_variant": t.get("expected_variant"),
            "actual_variant": t.get("actual_variant"),
            "dispatch_match": t.get("dispatch_match"),
            "variant_name": t["variant_name"],
            "expected_runtime_path": t["expected_runtime_path"],
            "baseline_mask_expr_count": b.get("mask_expr_count", 0),
            "submitted_mask_expr_count": t.get("mask_expr_count", 0),
            "mask_reduction": mask_reduction,
            "baseline_stride_expr_count": b.get("stride_expr_count", 0),
            "submitted_stride_expr_count": t.get("stride_expr_count", 0),
            "stride_reduction": stride_reduction,
            "baseline_pointer_expr_count": b.get("pointer_expr_count", 0),
            "submitted_pointer_expr_count": t.get("pointer_expr_count", 0),
            "pointer_reduction": pointer_reduction,
            "baseline_source_bytes": b.get("source_bytes", 0),
            "submitted_source_bytes": t.get("source_bytes", 0),
            "source_bytes_reduction": _safe_reduction(
                b.get("source_bytes", 0), t.get("source_bytes", 0)
            ),
            "baseline_kernel_param_count": b.get("kernel_param_count", 0),
            "submitted_kernel_param_count": t.get("kernel_param_count", 0),
            "baseline_stride_param_count": b.get("stride_param_count", 0),
            "submitted_stride_param_count": t.get("stride_param_count", 0),
            "baseline": b,
            "submitted": t,
        }
        rows.append(row)

        print(
            f"{name:<42} "
            f"{b_ms:>10.6f} "
            f"{t_ms:>10.6f} "
            f"{speedup:>8.3f}x "
            f"{str(t['correct']):>7} "
            f"{str(t['specialization_hit']):>6} "
            f"{t.get('expected_variant', 'unknown'):>29} "
            f"{t.get('actual_variant', 'unknown'):>29} "
            f"{str(t.get('dispatch_match')):>7} "
            f"{b.get('mask_expr_count', 0)}->{t.get('mask_expr_count', 0):<4} "
            f"{b.get('stride_expr_count', 0)}->{t.get('stride_expr_count', 0):<4} "
            f"{b.get('pointer_expr_count', 0)}->{t.get('pointer_expr_count', 0):<4}"
        )

    def _avg(values):
        return sum(values) / len(values) if values else 0.0

    def _median(values):
        return statistics.median(values) if values else 0.0

    average_speedup = _avg(speedups)
    average_hit_speedup = _avg(hit_speedups)
    average_fallback_speedup = _avg(fallback_speedups)
    median_speedup = _median(speedups)
    median_hit_speedup = _median(hit_speedups)
    median_fallback_speedup = _median(fallback_speedups)

    average_speedup_by_ndim = {
        str(ndim): _avg(values) for ndim, values in speedups_by_ndim.items()
    }
    average_hit_speedup_by_ndim = {
        str(ndim): _avg(values) for ndim, values in hit_speedups_by_ndim.items()
    }
    average_fallback_speedup_by_ndim = {
        str(ndim): _avg(values) for ndim, values in fallback_speedups_by_ndim.items()
    }
    median_speedup_by_ndim = {
        str(ndim): _median(values) for ndim, values in speedups_by_ndim.items()
    }
    median_hit_speedup_by_ndim = {
        str(ndim): _median(values) for ndim, values in hit_speedups_by_ndim.items()
    }
    median_fallback_speedup_by_ndim = {
        str(ndim): _median(values)
        for ndim, values in fallback_speedups_by_ndim.items()
    }

    print()
    print(f"median_speedup = {median_speedup:.4f}x")
    print(f"median_hit_speedup = {median_hit_speedup:.4f}x")
    print(f"median_fallback_speedup = {median_fallback_speedup:.4f}x")
    print(f"average_speedup = {average_speedup:.4f}x")
    print(f"average_hit_speedup = {average_hit_speedup:.4f}x")
    print(f"average_fallback_speedup = {average_fallback_speedup:.4f}x")
    print()
    print("=== Runtime Median/Average By Dimension ===")
    for ndim in (1, 2, 3):
        key = str(ndim)
        print(
            f"{ndim}D: "
            f"median(all/hit/fallback)="
            f"{median_speedup_by_ndim[key]:.4f}x/"
            f"{median_hit_speedup_by_ndim[key]:.4f}x/"
            f"{median_fallback_speedup_by_ndim[key]:.4f}x, "
            f"average(all/hit/fallback)="
            f"{average_speedup_by_ndim[key]:.4f}x/"
            f"{average_hit_speedup_by_ndim[key]:.4f}x/"
            f"{average_fallback_speedup_by_ndim[key]:.4f}x"
        )

    print()
    print("=== Triton Source Diagnostic Counts, not a local score ===")
    for row in rows:
        print(
            f"{row['scenario']:<42} "
            f"mask {row['baseline_mask_expr_count']}->"
            f"{row['submitted_mask_expr_count']} ({row['mask_reduction']:.1%}), "
            f"stride {row['baseline_stride_expr_count']}->"
            f"{row['submitted_stride_expr_count']} ({row['stride_reduction']:.1%}), "
            f"ptr {row['baseline_pointer_expr_count']}->"
            f"{row['submitted_pointer_expr_count']} ({row['pointer_reduction']:.1%}), "
            f"bytes {row['baseline_source_bytes']}->"
            f"{row['submitted_source_bytes']} ({row['source_bytes_reduction']:.1%}), "
            f"params {row['baseline_kernel_param_count']}->"
            f"{row['submitted_kernel_param_count']}, "
            f"stride_params {row['baseline_stride_param_count']}->"
            f"{row['submitted_stride_param_count']}"
        )

    baseline_source_diag = {
        item["scenario"]: item for item in baseline.get("source_diagnostics", [])
    }
    submitted_source_diag = {
        item["scenario"]: item for item in submitted.get("source_diagnostics", [])
    }

    if baseline_source_diag and submitted_source_diag:
        print()
        print("=== 1D/2D/3D Source Diagnostic Coverage Compare, not runtime ===")
        for name in baseline_source_diag:
            if name not in submitted_source_diag:
                print("missing submitted source diagnostic:", name)
                continue

            b = baseline_source_diag[name]
            t = submitted_source_diag[name]
            mask_reduction = _safe_reduction(
                b.get("mask_expr_count", 0), t.get("mask_expr_count", 0)
            )
            stride_reduction = _safe_reduction(
                b.get("stride_expr_count", 0), t.get("stride_expr_count", 0)
            )
            pointer_reduction = _safe_reduction(
                b.get("pointer_expr_count", 0), t.get("pointer_expr_count", 0)
            )

            print(
                f"{name:<48} "
                f"variant={t['variant_name']:<30} "
                f"mask {b.get('mask_expr_count', 0)}->"
                f"{t.get('mask_expr_count', 0)} ({mask_reduction:.1%}), "
                f"stride {b.get('stride_expr_count', 0)}->"
                f"{t.get('stride_expr_count', 0)} ({stride_reduction:.1%}), "
                f"ptr {b.get('pointer_expr_count', 0)}->"
                f"{t.get('pointer_expr_count', 0)} ({pointer_reduction:.1%}), "
                f"bytes {b.get('source_bytes', 0)}->"
                f"{t.get('source_bytes', 0)}, "
                f"params {b.get('kernel_param_count', 0)}->"
                f"{t.get('kernel_param_count', 0)}, "
                f"stride_params {b.get('stride_param_count', 0)}->"
                f"{t.get('stride_param_count', 0)}"
            )

        source_reductions_by_ndim = {1: [], 2: [], 3: []}
        for name in baseline_source_diag:
            if name not in submitted_source_diag:
                continue
            b = baseline_source_diag[name]
            t = submitted_source_diag[name]
            ndim = t.get("ndim")
            if ndim not in source_reductions_by_ndim:
                continue
            reductions = (
                _safe_reduction(
                    b.get("mask_expr_count", 0), t.get("mask_expr_count", 0)
                ),
                _safe_reduction(
                    b.get("stride_expr_count", 0), t.get("stride_expr_count", 0)
                ),
                _safe_reduction(
                    b.get("pointer_expr_count", 0), t.get("pointer_expr_count", 0)
                ),
            )
            source_reductions_by_ndim[ndim].append(max(reductions))

        print()
        print("=== Source Diagnostic Average Reduction By Dimension, not runtime ===")
        for ndim in (1, 2, 3):
            values = source_reductions_by_ndim[ndim]
            avg_red = sum(values) / len(values) if values else 0.0
            print(f"{ndim}D: best-count-reduction-average={avg_red:.1%}")

    out = {
        "benchmark_name": (
            "T1-2-1 AOT specialization median runtime, actual dispatch, "
            "and generated-source diagnostics"
        ),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_json": str(baseline_json),
        "submitted_json": str(submitted_json),
        "median_speedup": median_speedup,
        "median_hit_speedup": median_hit_speedup,
        "median_fallback_speedup": median_fallback_speedup,
        "average_speedup": average_speedup,
        "average_hit_speedup": average_hit_speedup,
        "average_fallback_speedup": average_fallback_speedup,
        "median_speedup_by_ndim": median_speedup_by_ndim,
        "median_hit_speedup_by_ndim": median_hit_speedup_by_ndim,
        "median_fallback_speedup_by_ndim": median_fallback_speedup_by_ndim,
        "average_speedup_by_ndim": average_speedup_by_ndim,
        "average_hit_speedup_by_ndim": average_hit_speedup_by_ndim,
        "average_fallback_speedup_by_ndim": average_fallback_speedup_by_ndim,
        "results": rows,
        "baseline_source_diagnostics": list(baseline_source_diag.values())
        if baseline_source_diag
        else [],
        "submitted_source_diagnostics": list(submitted_source_diag.values())
        if submitted_source_diag
        else [],
    }

    output_path = root / "benchmarks" / "bench_aot_speedup_compare_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))
    print("Wrote:", output_path)

def _benchmark_worker(tag, output):
    import torch
    import ninetoothed

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    cache_dir = pathlib.Path.home() / ".ninetoothed"

    def bench_cuda(fn, warmup=50, repeat=500):
        for _ in range(warmup):
            fn()

        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(repeat):
            fn()
        end.record()

        torch.cuda.synchronize()
        return start.elapsed_time(end) / repeat

    def benchmark_distribution(fn):
        sample_count = int(os.environ.get("NT_BENCH_SAMPLES", "9"))
        if sample_count < 3:
            raise ValueError("NT_BENCH_SAMPLES must be at least 3")

        samples = [bench_cuda(fn) for _ in range(sample_count)]
        q1, _, q3 = statistics.quantiles(samples, n=4, method="inclusive")
        return {
            "runtime_ms": statistics.median(samples),
            "runtime_p25_ms": q1,
            "runtime_p75_ms": q3,
            "runtime_samples_ms": samples,
        }

    def actual_variant_name(kernel):
        getter = getattr(kernel, "get_last_variant", None)
        return getter() if getter is not None else "unavailable"

    def make_add_kernel(kernel_name, ndim, dtype_nt, tile_profile=None):
        arrangement, application, tensors = _make_add_components(
            ndim, dtype_nt, tile_profile=tile_profile
        )

        kernel = ninetoothed.make(
            arrangement,
            application,
            tensors,
            caller="cuda",
            kernel_name=kernel_name,
            output_dir=cache_dir,
        )

        return kernel, application

    def make_scalar_kernel(kernel_name):
        arrangement, application, tensors = _make_scalar_components()

        kernel = ninetoothed.make(
            arrangement,
            application,
            tensors,
            caller="cuda",
            kernel_name=kernel_name,
            output_dir=cache_dir,
        )

        return kernel, application

    def make_tensor(shape, dtype, noncontiguous):
        if not noncontiguous:
            return torch.randn(shape, device="cuda", dtype=dtype)

        if len(shape) == 1:
            base = torch.randn((shape[0] * 2,), device="cuda", dtype=dtype)
            return base[::2]

        if len(shape) == 2:
            base = torch.randn((shape[0] * 2, shape[1]), device="cuda", dtype=dtype)
            return base[::2, :]

        if len(shape) == 3:
            base = torch.randn(
                (shape[0] * 2, shape[1], shape[2]),
                device="cuda",
                dtype=dtype,
            )
            return base[::2, :, :]

        raise ValueError(f"Unsupported shape: {shape}")

    def make_empty_like_strided(tensor, noncontiguous):
        if not noncontiguous:
            return torch.empty_like(tensor)

        shape = tuple(tensor.shape)
        dtype = tensor.dtype

        if len(shape) == 1:
            base = torch.empty((shape[0] * 2,), device="cuda", dtype=dtype)
            return base[::2]

        if len(shape) == 2:
            base = torch.empty((shape[0] * 2, shape[1]), device="cuda", dtype=dtype)
            return base[::2, :]

        if len(shape) == 3:
            base = torch.empty(
                (shape[0] * 2, shape[1], shape[2]),
                device="cuda",
                dtype=dtype,
            )
            return base[::2, :, :]

        raise ValueError(f"Unsupported tensor ndim: {tensor.ndim}")

    def run_add_case(
        name,
        ndim,
        shape,
        dtype_torch,
        dtype_nt,
        *,
        expected_variant,
        noncontiguous=False,
        tile_profile=None,
    ):
        print(f"[{tag}] compile/run {name}", flush=True)
        kernel_name = f"bench_speedup_{tag}_{name}"
        kernel, application = make_add_kernel(
            kernel_name, ndim, dtype_nt, tile_profile=tile_profile
        )

        x = make_tensor(shape, dtype_torch, noncontiguous)
        y = make_tensor(shape, dtype_torch, noncontiguous)
        out = make_empty_like_strided(x, noncontiguous)

        kernel(x, y, out)
        torch.cuda.synchronize()

        expected = x + y
        kernel(x, y, out)
        torch.cuda.synchronize()

        correct = torch.allclose(out, expected)
        max_diff = (out - expected).abs().max().item()

        timing = benchmark_distribution(lambda: kernel(x, y, out))
        actual_variant = actual_variant_name(kernel)
        dispatch_match = (
            None
            if actual_variant == "unavailable"
            else actual_variant == expected_variant
        )

        if tag == "submitted" and dispatch_match is not True:
            raise AssertionError(
                f"Dispatch mismatch for {name}: "
                f"expected={expected_variant}, actual={actual_variant}"
            )

        diagnostic_variant = (
            actual_variant if actual_variant != "unavailable" else expected_variant
        )
        diagnostic_counts = _triton_source_diagnostic_counts(
            application, kernel_name, diagnostic_variant
        )
        specialization_hit = actual_variant in (
            "flatten_contiguous_divisible",
            "flatten_contiguous_masked",
        )

        return {
            "scenario": name,
            "tag": tag,
            "ndim": ndim,
            "shape": str(shape),
            "dtype": str(dtype_torch),
            "noncontiguous": noncontiguous,
            "tile_profile": tile_profile,
            "expected_runtime_path": (
                "legacy_or_fallback"
                if noncontiguous or ndim > 2
                else f"{expected_variant}_runtime"
            ),
            "runtime_ms": round(timing["runtime_ms"], 6),
            "runtime_p25_ms": round(timing["runtime_p25_ms"], 6),
            "runtime_p75_ms": round(timing["runtime_p75_ms"], 6),
            "runtime_samples_ms": [round(value, 6) for value in timing["runtime_samples_ms"]],
            "correct": bool(correct),
            "max_diff": max_diff,
            "expected_specialization_hit": expected_variant.startswith(
                "flatten_contiguous_"
            ),
            "specialization_hit": specialization_hit,
            "expected_variant": expected_variant,
            "actual_variant": actual_variant,
            "dispatch_match": dispatch_match,
            "variant_name": (
                actual_variant if actual_variant != "unavailable" else expected_variant
            ),
            **diagnostic_counts,
        }

    def run_scalar_case(name, n):
        print(f"[{tag}] compile/run {name}", flush=True)
        kernel_name = f"bench_speedup_{tag}_{name}"
        kernel, application = make_scalar_kernel(kernel_name)

        x = torch.randn((n,), device="cuda", dtype=torch.float32)
        scale = 0.125
        out = torch.empty_like(x)

        kernel(x, scale, out)
        torch.cuda.synchronize()

        expected = x * scale
        kernel(x, scale, out)
        torch.cuda.synchronize()

        correct = torch.allclose(out, expected)
        max_diff = (out - expected).abs().max().item()

        timing = benchmark_distribution(lambda: kernel(x, scale, out))
        expected_variant = "legacy"
        actual_variant = actual_variant_name(kernel)
        dispatch_match = (
            None
            if actual_variant == "unavailable"
            else actual_variant in ("legacy", "fallback")
        )

        if tag == "submitted" and dispatch_match is not True:
            raise AssertionError(
                f"Dispatch mismatch for {name}: actual={actual_variant}"
            )

        diagnostic_counts = _triton_source_diagnostic_counts(
            application, kernel_name, "fallback_scalar_arg"
        )

        return {
            "scenario": name,
            "tag": tag,
            "ndim": 1,
            "n": n,
            "dtype": "torch.float32",
            "noncontiguous": False,
            "expected_runtime_path": "legacy_or_fallback",
            "runtime_ms": round(timing["runtime_ms"], 6),
            "runtime_p25_ms": round(timing["runtime_p25_ms"], 6),
            "runtime_p75_ms": round(timing["runtime_p75_ms"], 6),
            "runtime_samples_ms": [round(value, 6) for value in timing["runtime_samples_ms"]],
            "correct": bool(correct),
            "max_diff": max_diff,
            "expected_specialization_hit": False,
            "specialization_hit": False,
            "expected_variant": expected_variant,
            "actual_variant": actual_variant,
            "dispatch_match": dispatch_match,
            "variant_name": (
                actual_variant if actual_variant != "unavailable" else expected_variant
            ),
            **diagnostic_counts,
        }

    def collect_source_diagnostic_coverage():
        """Collect 1D/2D/3D generated-source diagnostic counts.

        These cases are NOT runtime benchmark cases and are NOT included in
        average_speedup.  They exist only so the local output still shows
        1D/2D/3D mask/stride/pointer evidence while the runtime average stays
        focused on true runtime-hit and fallback cases.
        """
        source_cases = []

        source_profiles = [
            (ndim, tile_profile)
            for ndim in (1, 2, 3)
            for tile_profile in TILE_SWEEP_PROFILES[ndim]
        ]

        for ndim, tile_profile in source_profiles:
            tile_label = _tile_profile_label(tile_profile)
            print(
                f"[{tag}] source-diagnostic {ndim}d {tile_label}",
                flush=True,
            )
            _, application, _ = _make_add_components(
                ndim, ninetoothed.bfloat16, tile_profile=tile_profile
            )
            source_cases.append(
                (
                    f"source_diag_divisible_{ndim}d_{tile_label}_bf16",
                    ndim,
                    "flatten_contiguous_divisible",
                    application,
                )
            )
            source_cases.append(
                (
                    f"source_diag_masked_{ndim}d_{tile_label}_bf16",
                    ndim,
                    "flatten_contiguous_masked",
                    application,
                )
            )

        _, application_3d, _ = _make_add_components(
            3, ninetoothed.float32, tile_profile="balanced"
        )
        source_cases.append(
            (
                "source_diag_fallback_noncontiguous_3d_fp32",
                3,
                "fallback_non_contiguous",
                application_3d,
            )
        )

        _, scalar_application, _ = _make_scalar_components()
        source_cases.append(
            (
                "source_diag_fallback_scalar_arg_fp32",
                1,
                "fallback_scalar_arg",
                scalar_application,
            )
        )

        diagnostics = []
        for scenario, ndim, variant, application in source_cases:
            kernel_name = f"bench_speedup_{tag}_{scenario}"
            counts = _triton_source_diagnostic_counts(
                application, kernel_name, variant
            )
            diagnostics.append(
                {
                    "scenario": scenario,
                    "tag": tag,
                    "ndim": ndim,
                    "variant_name": variant,
                    **counts,
                }
            )

        return diagnostics

    print("GPU:", torch.cuda.get_device_name(0))
    print("tag:", tag)

    # Runtime benchmark cases, separated by ndim and tile profile:
    # - Runtime uses RUNTIME_TILE_SWEEP_PROFILES, not full TILE_SWEEP_PROFILES.
    # - Full tile-profile coverage is kept in source diagnostics only.
    # - This avoids nvcc variant explosion for heavy 2D/3D tiles while still
    #   reporting 1D/2D/3D runtime averages and source diagnostics separately.
    divisible_shapes = {
        1: (262144,),
        2: (1024, 1024),
        3: (128, 128, 64),
    }
    masked_shapes = {
        1: (262151,),
        2: (1025, 1023),
        3: (129, 127, 63),
    }

    results = []

    for ndim in (1, 2, 3):
        for tile_profile in RUNTIME_TILE_SWEEP_PROFILES[ndim]:
            tile_label = _tile_profile_label(tile_profile)

            results.append(
                run_add_case(
                    f"runtime_{ndim}d_{tile_label}_flatten_divisible_bf16",
                    ndim,
                    divisible_shapes[ndim],
                    torch.bfloat16,
                    ninetoothed.bfloat16,
                    expected_variant=(
                        "legacy"
                        if ndim > 2
                        else "flatten_contiguous_divisible"
                    ),
                    noncontiguous=False,
                    tile_profile=tile_profile,
                )
            )

            results.append(
                run_add_case(
                    f"runtime_{ndim}d_{tile_label}_flatten_masked_bf16",
                    ndim,
                    masked_shapes[ndim],
                    torch.bfloat16,
                    ninetoothed.bfloat16,
                    expected_variant=(
                        "legacy"
                        if ndim > 2
                        else "flatten_contiguous_masked"
                    ),
                    noncontiguous=False,
                    tile_profile=tile_profile,
                )
            )

    for ndim in (1, 2, 3):
        for tile_profile in FALLBACK_TILE_SWEEP_PROFILES[ndim]:
            tile_label = _tile_profile_label(tile_profile)

            results.append(
                run_add_case(
                    f"fallback_noncontiguous_{ndim}d_{tile_label}_fp32",
                    ndim,
                    divisible_shapes[ndim],
                    torch.float32,
                    ninetoothed.float32,
                    expected_variant="legacy",
                    noncontiguous=True,
                    tile_profile=tile_profile,
                )
            )

    results.append(
        run_scalar_case(
            "fallback_scalar_arg_fp32",
            262144,
        )
    )

    for item in results:
        print(
            f"{item['scenario']:<42} "
            f"median={item['runtime_ms']:.6f} ms  "
            f"iqr=[{item['runtime_p25_ms']:.6f}, {item['runtime_p75_ms']:.6f}]  "
            f"correct={item['correct']}  "
            f"max_diff={item['max_diff']}  "
            f"expected={item['expected_variant']}  "
            f"actual={item['actual_variant']}  "
            f"dispatch_match={item['dispatch_match']}  "
            f"specialization_hit={item['specialization_hit']}  "
            f"mask={item['mask_expr_count']}  "
            f"stride={item['stride_expr_count']}  "
            f"ptr={item['pointer_expr_count']}"
        )

    source_diagnostics = collect_source_diagnostic_coverage()

    print()
    print("=== 1D/2D/3D Source Diagnostic Coverage, not runtime ===")
    for item in source_diagnostics:
        print(
            f"{item['scenario']:<48} "
            f"variant={item['variant_name']:<30} "
            f"mask={item['mask_expr_count']}  "
            f"stride={item['stride_expr_count']}  "
            f"ptr={item['pointer_expr_count']}"
        )

    output_obj = {
        "tag": tag,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": torch.cuda.get_device_name(0),
        "results": results,
        "source_diagnostics": source_diagnostics,
    }

    pathlib.Path(output).write_text(json.dumps(output_obj, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.worker:
        if args.tag is None or args.output is None:
            raise RuntimeError("--worker requires --tag and --output")
        _benchmark_worker(args.tag, args.output)
    elif args.benchmark:
        _benchmark_driver()
    else:
        print(
            "This file is normally used by pytest. "
            "Run with --benchmark to measure baseline vs submitted speedup."
        )


if __name__ == "__main__":
    main()
