#!/usr/bin/env python3
"""
debug_arrangement.py — verify a NineToothed arrangement before compiling a kernel.

Wraps `ninetoothed.debugging.simulate_arrangement` to give human-readable
output: for each tensor, prints the source-index grid shape and a summary
of which source elements each tile position reads.

This script is designed to be called BY an agent during step 5 of the
SKILL.md workflow ("Verify arrangement") and takes no arguments — the
arrangement under test must be passed as a Python import or monkey-patched.

Usage (the agent writes a small driver file and calls this as a module):

    # In your kernel file (e.g. kernel.py):
    from ninetoothed.debugging import simulate_arrangement
    src, tgt = simulate_arrangement(arrangement, tensors)
    # Then call debug_arrangement.summarise(arrangement, tensors) for a report.

Or import this module directly from agent code:

    from scripts.debug_arrangement import summarise, check_no_oob

Requirements: torch (CUDA), ninetoothed
No extra dependencies beyond the core requirements.txt.
"""

from __future__ import annotations

import sys


def summarise(arrangement, tensors, device: str = "cuda") -> dict:
    """
    Run simulate_arrangement and return a human-readable summary dict.

    Returns
    -------
    dict with keys:
        "tensors": list of per-tensor dicts, each containing:
            "source_shape": tuple
            "target_shape": tuple   (full arranged shape incl. program dim)
            "num_programs": int     (number of parallel GPU programs)
            "tile_shape":   tuple   (shape seen inside each program)
            "oob_count":    int     (number of -1 sentinel values in target)
            "unique_sources": int   (how many distinct source elements are read)
        "all_oob_free": bool
        "all_elements_covered": bool  (every source element is read at least once)
    """
    try:
        import torch  # noqa: F401  # availability guard for the error message below

        from ninetoothed.debugging import simulate_arrangement
    except ImportError as e:
        raise ImportError(
            f"Debug_arrangement requires torch + ninetoothed: {e}."
        ) from e

    src_tensors, tgt_tensors = simulate_arrangement(arrangement, tensors, device=device)

    results = []

    for i, (src, tgt) in enumerate(zip(src_tensors, tgt_tensors)):
        oob = int((tgt == -1).sum().item()) if tgt.numel() > 0 else 0
        unique = int(tgt[tgt >= 0].unique().numel()) if tgt.numel() > 0 else 0
        total_src = int(src.numel())

        # Tile_shape = shape inside one program (all dims after the first).
        tile_shape = tuple(tgt.shape[1:]) if tgt.dim() > 1 else ()

        results.append(
            {
                "tensor_index": i,
                "source_shape": tuple(src.shape),
                "target_shape": tuple(tgt.shape),
                "num_programs": int(tgt.shape[0]) if tgt.dim() >= 1 else 1,
                "tile_shape": tile_shape,
                "oob_count": oob,
                "unique_sources": unique,
                "total_source_elements": total_src,
            }
        )

    all_oob_free = all(r["oob_count"] == 0 for r in results)
    all_covered = all(
        r["unique_sources"] == r["total_source_elements"] for r in results
    )

    return {
        "tensors": results,
        "all_oob_free": all_oob_free,
        "all_elements_covered": all_covered,
    }


def print_summary(info: dict) -> None:
    """Print the summarise() result in a readable format."""
    print("=== arrangement debug summary ===")

    for r in info["tensors"]:
        idx = r["tensor_index"]
        oob_flag = " *** OOB DETECTED ***" if r["oob_count"] > 0 else ""
        cov_flag = (
            " *** NOT ALL ELEMENTS READ ***"
            if r["unique_sources"] < r["total_source_elements"]
            else ""
        )
        print(
            f"  tensor[{idx}]  source={r['source_shape']}  "
            f"target={r['target_shape']}  "
            f"programs={r['num_programs']}  tile={r['tile_shape']}"
        )
        print(
            f"           oob={r['oob_count']}{oob_flag}  "
            f"unique_read={r['unique_sources']}/{r['total_source_elements']}{cov_flag}"
        )

    status = "OK" if info["all_oob_free"] else "FAIL (OOB)"
    print(f"  status: {status}  all_covered={info['all_elements_covered']}")


def check_no_oob(arrangement, tensors, device: str = "cuda") -> bool:
    """Return True if the arrangement produces no out-of-bounds accesses."""
    info = summarise(arrangement, tensors, device=device)
    print_summary(info)

    return info["all_oob_free"]


def visualize_and_save(
    arrangement, tensors, save_dir: str = ".", device: str = "cuda"
) -> list[str]:
    """
    Generate one arrangement PNG per tensor and save to save_dir.

    Requires matplotlib (optional dep). Falls back gracefully if not installed.
    Returns list of saved file paths.
    """
    try:
        import pathlib

        import torch  # noqa: F401  # availability guard for the error message below

        from ninetoothed.debugging import simulate_arrangement
        from ninetoothed.visualization import visualize
    except ImportError as e:
        print(
            f"[debug_arrangement] visualize skipped: {e}\n"
            "  Install optional deps: pip install matplotlib\n"
            "  Or: bash setup.sh --with-viz",
            file=sys.stderr,
        )

        return []

    _, tgt_tensors = simulate_arrangement(arrangement, tensors, device=device)
    saved = []
    out_dir = pathlib.Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, tgt in enumerate(tgt_tensors):
        # Build a symbolic Tensor from the target shape for visualize().
        # visualize() accepts a ninetoothed Tensor; we pass the arranged
        # tensor's shape description via a dummy Tensor of matching ndim.
        from ninetoothed import Tensor as NTTensor

        dummy = NTTensor(tgt.dim())
        path = str(out_dir / f"arrangement_tensor{i}.png")

        try:
            # Visualize(dummy) shows tile structure; save_path writes PNG.
            visualize(dummy, save_path=path)
            saved.append(path)
            print(f"[debug_arrangement] saved arrangement viz: {path}")
        except Exception as exc:
            print(f"[debug_arrangement] visualize({i}) failed: {exc}", file=sys.stderr)

    return saved


# ---------------------------------------------------------------------------
# CLI usage: python debug_arrangement.py <module_path>:<arrangement_name>.
# ---------------------------------------------------------------------------
def _cli(argv: list[str]) -> int:
    """CLI helper.

    Expects one argument: module_dotpath:arrangement_fn_name
    e.g.:
        python scripts/debug_arrangement.py examples.02_reduction_parameterized.kernel:arrangement.
    """
    if not argv:
        print("usage: python debug_arrangement.py module.path:arrangement_fn")

        return 1

    spec = argv[0]

    if ":" not in spec:
        print(f"bad spec {spec!r}; expected module.path:fn_name")

        return 1

    mod_path, fn_name = spec.rsplit(":", 1)
    import importlib

    try:
        mod = importlib.import_module(mod_path)
    except ImportError as e:
        print(f"cannot import {mod_path!r}: {e}", file=sys.stderr)

        return 1

    fn = getattr(mod, fn_name, None)

    if fn is None:
        print(f"{fn_name!r} not found in {mod_path!r}", file=sys.stderr)

        return 1

    # Try to find the tensors used with this arrangement.
    tensors = getattr(mod, "_TENSORS", None) or getattr(mod, "tensors", None)

    if tensors is None:
        print(
            f"cannot find _TENSORS or tensors in {mod_path!r}; "
            "define _TENSORS at module level or pass tensors directly.",
            file=sys.stderr,
        )

        return 1

    info = summarise(fn, tensors)
    print_summary(info)

    return 0 if info["all_oob_free"] else 1


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
