#!/usr/bin/env python3
"""Debug a NineToothed arrangement before compiling a kernel.

Wraps `ninetoothed.debugging.simulate_arrangement` to detect:
- Out-of-bounds (OOB) accesses: tiles reading beyond tensor boundaries
- Element coverage: whether every source element is read at least once
- Tile shape summary: programs, tile dims, unique sources

Usage (CLI):
    python scripts/debug_arrangement.py module.path:arrangement_fn

Usage (Python import):
    from scripts.debug_arrangement import summarise, check_no_oob, print_summary

    info = summarise(arrangement, tensors)
    print_summary(info)
    assert info["all_oob_free"], "OOB detected!"
    assert info["all_elements_covered"], "Not all elements read!"

Requires: torch, ninetoothed
"""
from __future__ import annotations

import sys


def summarise(arrangement, tensors, device: str = "cuda") -> dict:
    """Run simulate_arrangement and return a human-readable summary dict.

    Returns
    -------
    dict with keys:
        "tensors": list of per-tensor dicts containing:
            "source_shape": tuple
            "target_shape": tuple (full arranged shape incl. program dim)
            "num_programs": int
            "tile_shape": tuple (shape seen inside each program)
            "oob_count": int (number of -1 sentinel values in target)
            "unique_sources": int (distinct source elements read)
            "total_source_elements": int
        "all_oob_free": bool
        "all_elements_covered": bool
    """
    try:
        import torch
        from ninetoothed.debugging import simulate_arrangement
    except ImportError as e:
        raise ImportError(f"debug_arrangement requires torch + ninetoothed: {e}") from e

    src_tensors, tgt_tensors = simulate_arrangement(arrangement, tensors, device=device)

    results = []
    for i, (src, tgt) in enumerate(zip(src_tensors, tgt_tensors)):
        oob = int((tgt == -1).sum().item()) if tgt.numel() > 0 else 0
        unique = int(tgt[tgt >= 0].unique().numel()) if tgt.numel() > 0 else 0
        total_src = int(src.numel())
        tile_shape = tuple(tgt.shape[1:]) if tgt.dim() > 1 else ()

        results.append({
            "tensor_index": i,
            "source_shape": tuple(src.shape),
            "target_shape": tuple(tgt.shape),
            "num_programs": int(tgt.shape[0]) if tgt.dim() >= 1 else 1,
            "tile_shape": tile_shape,
            "oob_count": oob,
            "unique_sources": unique,
            "total_source_elements": total_src,
        })

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
    """Generate arrangement visualization PNGs (requires matplotlib).

    Returns list of saved file paths, or empty list if matplotlib unavailable.
    """
    try:
        from ninetoothed.visualization import visualize
        from ninetoothed import Tensor as NTTensor
        from ninetoothed.debugging import simulate_arrangement
        import pathlib
        import torch
    except ImportError as e:
        print(
            f"[debug_arrangement] visualize skipped: {e}\n"
            "  Install optional deps: pip install matplotlib",
            file=sys.stderr,
        )
        return []

    _, tgt_tensors = simulate_arrangement(arrangement, tensors, device=device)
    saved = []
    out_dir = pathlib.Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, tgt in enumerate(tgt_tensors):
        dummy = NTTensor(tgt.dim())
        path = str(out_dir / f"arrangement_tensor{i}.png")
        try:
            visualize(dummy, save_path=path)
            saved.append(path)
            print(f"[debug_arrangement] saved: {path}")
        except Exception as exc:
            print(f"[debug_arrangement] visualize({i}) failed: {exc}", file=sys.stderr)

    return saved


def _cli(argv: list[str]) -> int:
    """CLI: python debug_arrangement.py module.path:arrangement_fn"""
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

    tensors = getattr(mod, "_TENSORS", None) or getattr(mod, "tensors", None)
    if tensors is None:
        print(
            f"cannot find _TENSORS or tensors in {mod_path!r}; "
            "define _TENSORS at module level.",
            file=sys.stderr,
        )
        return 1

    info = summarise(fn, tensors)
    print_summary(info)
    return 0 if info["all_oob_free"] else 1


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
