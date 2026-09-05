"""Inspect generated Triton source code from NineToothed cache.

Reads the compiled kernel from ~/.ninetoothed/ cache directory and
analyzes it for potential issues:
- Load/store counts (memory access efficiency)
- num_warps / num_stages configuration
- dtype usage patterns
- Potential private memory issues

Usage:
    python scripts/inspect_generated.py --op add
    python scripts/inspect_generated.py --op matmul --verbose
    python scripts/inspect_generated.py --cache-dir ~/.ninetoothed --list
"""

import argparse
import ast
import os
import re
import sys
from collections import Counter
from pathlib import Path


def find_cache_files(cache_dir: Path):
    """Find all generated .py files in the ninetoothed cache."""
    if not cache_dir.exists():
        print(f"Cache directory not found: {cache_dir}")
        return []
    return sorted(cache_dir.glob("*.py"))


def digest_file(filepath: Path, verbose: bool = False) -> dict:
    """Parse a generated Triton kernel file and extract key metrics."""
    source = filepath.read_text()
    sha = filepath.stem

    stats = {
        "sha": sha,
        "file": str(filepath),
        "lines": len(source.splitlines()),
        "tl_load": len(re.findall(r"tl\.load\(", source)),
        "tl_store": len(re.findall(r"tl\.store\(", source)),
        "tl_dot": len(re.findall(r"tl\.dot\(", source)),
        "tl_exp": len(re.findall(r"tl\.math\.exp\(", source)),
        "tl_exp2": len(re.findall(r"tl\.math\.exp2\(", source)),
        "tl_sqrt": len(re.findall(r"tl\.math\.(rsqrt|sqrt)\(", source)),
        "tl_libdevice": len(re.findall(r"tl\.math\.libdevice\.\w+\(", source)),
        "num_warps": None,
        "num_stages": None,
        "has_mask": "mask=" in source,
        "has_cast_to_fp32": "tl.cast(" in source and "tl.float32" in source,
    }

    # Extract num_warps and num_stages
    warps_match = re.search(r"num_warps\s*=\s*(\d+)", source)
    if warps_match:
        stats["num_warps"] = int(warps_match.group(1))

    stages_match = re.search(r"num_stages\s*=\s*(\d+)", source)
    if stages_match:
        stats["num_stages"] = int(stages_match.group(1))

    # Check for potential issues
    issues = []

    # Load/Store ratio
    if stats["tl_load"] > 0 and stats["tl_store"] > 0:
        ratio = stats["tl_load"] / stats["tl_store"]
        if ratio > 10:
            issues.append(f"HIGH_LOAD_RATIO: {ratio:.1f}x loads vs stores (possible redundant loads)")

    # No mask
    if not stats["has_mask"]:
        issues.append("NO_MASK: No boundary masking detected (may fail on non-aligned shapes)")

    # fp32 cast check for libdevice
    if stats["tl_libdevice"] > 0 and not stats["has_cast_to_fp32"]:
        issues.append("FP32_CAST_MISSING: libdevice calls without float32 cast (precision risk)")

    # exp vs exp2
    if stats["tl_exp"] > 0 and stats["tl_exp2"] == 0:
        issues.append("EXP_NOT_EXP2: Using tl.math.exp instead of tl.math.exp2 (slower on some GPUs)")

    stats["issues"] = issues

    if verbose:
        print(f"\n{'='*60}")
        print(f"File: {filepath.name}")
        print(f"{'='*60}")
        print(f"  Lines:          {stats['lines']}")
        print(f"  tl.load:        {stats['tl_load']}")
        print(f"  tl.store:       {stats['tl_store']}")
        print(f"  tl.dot:         {stats['tl_dot']}")
        print(f"  tl.exp:         {stats['tl_exp']}")
        print(f"  tl.exp2:        {stats['tl_exp2']}")
        print(f"  tl.sqrt/rsqrt:  {stats['tl_sqrt']}")
        print(f"  libdevice:      {stats['tl_libdevice']}")
        print(f"  num_warps:      {stats['num_warps']}")
        print(f"  num_stages:     {stats['num_stages']}")
        print(f"  has_mask:       {stats['has_mask']}")
        print(f"  has_fp32_cast:  {stats['has_cast_to_fp32']}")
        if issues:
            print(f"\n  ⚠️  ISSUES:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print(f"\n  ✅ No issues detected")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Inspect NineToothed generated Triton source")
    parser.add_argument("--op", type=str, default=None, help="Operator name (searches cache by keyword)")
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory (default: ~/.ninetoothed)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed analysis per file")
    parser.add_argument("--list", action="store_true", help="List all cached kernel files")
    args = parser.parse_args()

    if args.cache_dir:
        cache_dir = Path(args.cache_dir).expanduser()
    else:
        cache_dir = Path.home() / ".ninetoothed"

    files = find_cache_files(cache_dir)

    if not files:
        print(f"No generated files found in {cache_dir}")
        sys.exit(1)

    if args.op:
        # Filter by operator name (search in file content)
        matched = []
        for f in files:
            content = f.read_text().lower()
            if args.op.lower() in content:
                matched.append(f)
        if not matched:
            print(f"No cached files matching operator '{args.op}'")
            print(f"Showing all {len(files)} files:")
            files_to_show = files
        else:
            print(f"Found {len(matched)} file(s) matching '{args.op}':")
            files_to_show = matched
    else:
        files_to_show = files

    if args.list:
        for f in files_to_show:
            print(f"  {f.name}")
        return

    # Digest all matched files
    all_stats = []
    for f in files_to_show:
        stats = digest_file(f, verbose=args.verbose)
        all_stats.append(stats)

    # Summary
    if not args.verbose and all_stats:
        print(f"\n{'SHA':<12s} {'Lines':>6s} {'Loads':>6s} {'Stores':>6s} {'Dots':>5s} "
              f"{'Warps':>6s} {'Issues':>7s}")
        print("-" * 60)
        for s in all_stats:
            warps = str(s["num_warps"]) if s["num_warps"] else "?"
            issue_count = len(s["issues"])
            marker = "⚠️" if issue_count > 0 else "✅"
            print(f"{s['sha'][:12]:<12s} {s['lines']:>6d} {s['tl_load']:>6d} "
                  f"{s['tl_store']:>6d} {s['tl_dot']:>5d} {warps:>6s} "
                  f"{marker} {issue_count}")

    # Aggregate issues
    total_issues = sum(len(s["issues"]) for s in all_stats)
    if total_issues > 0:
        print(f"\n⚠️  Found {total_issues} potential issue(s) across {len(all_stats)} file(s).")
        print("   Run with --verbose for details.")
    else:
        print(f"\n✅ No issues found across {len(all_stats)} file(s).")


if __name__ == "__main__":
    main()
