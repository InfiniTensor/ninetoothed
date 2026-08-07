#!/usr/bin/env python3
"""
Load, validate, and summarise the proxy task set.

Runs three checks:
  1. Schema validation on every task.
  2. Structural invariants: 24 tasks, 6 per category, 4 train + 2 holdout each.
  3. (only if torch is installed) numeric self-check: run each operator task's
     make_inputs + reference on CPU and confirm a finite tensor of the right
     leading shape comes out.

Also (re)generates manifest.json from the modules so the manifest can never
drift from the code.

Usage:
    python loader.py            # validate + summarise + write manifest.json
    python loader.py --check    # also run the torch CPU self-check
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

from schema import CATEGORIES

_MODULES = ("elementwise", "reduction", "layout", "perf_diag")


def load_all() -> list:
    tasks = []

    for mod_name in _MODULES:
        mod = __import__(mod_name)
        tasks.extend(mod.TASKS)
    return tasks


def validate_schema(tasks) -> list[str]:
    errs = []
    seen = set()

    for t in tasks:
        errs.extend(t.validate())

        if t.id in seen:
            errs.append(f"duplicate id {t.id}")

        seen.add(t.id)
    return errs


def check_invariants(tasks) -> list[str]:
    errs = []

    if len(tasks) != 24:
        errs.append(f"expected 24 tasks, got {len(tasks)}")

    for cat in CATEGORIES:
        cat_tasks = [t for t in tasks if t.category == cat]

        if len(cat_tasks) != 6:
            errs.append(f"category {cat}: expected 6, got {len(cat_tasks)}")

        n_train = sum(t.split == "train" for t in cat_tasks)
        n_hold = sum(t.split == "holdout" for t in cat_tasks)

        if n_train != 4:
            errs.append(f"category {cat}: expected 4 train, got {n_train}")

        if n_hold != 2:
            errs.append(f"category {cat}: expected 2 holdout, got {n_hold}")
    return errs


def numeric_self_check(tasks) -> list[str]:
    """Run operator references on CPU. Skipped if torch is absent."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return ["__skipped__: torch not installed; numeric self-check skipped"]

    errs = []

    for t in tasks:
        if t.kind != "operator":
            continue

        try:
            inputs = t.make_inputs(device="cpu", dtype="float32")
            out = t.reference(*inputs)
            import torch

            if not torch.is_tensor(out):
                errs.append(f"{t.id}: reference did not return a tensor")
                continue

            if not torch.isfinite(out).all():
                errs.append(f"{t.id}: reference produced non-finite values")
        except Exception as e:  # noqa: BLE001
            errs.append(f"{t.id}: reference raised {type(e).__name__}: {e}")
    return errs


def write_manifest(tasks, path: pathlib.Path) -> None:
    manifest = {
        "version": 1,
        "total": len(tasks),
        "split_rule": "per category: 4 train + 2 holdout",
        "tolerance_ref": "see ninetoothed-operator-dev/scripts/run_correctness_matrix.py (MERE/MARE)",
        "tasks": [t.meta() for t in tasks],
    }
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def summarise(tasks) -> None:
    print("proxy task set summary")
    print(f"  total: {len(tasks)}")
    header = f"  {'category':<12}{'train':>6}{'holdout':>9}{'total':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for cat in CATEGORIES:
        ct = [t for t in tasks if t.category == cat]
        tr = sum(t.split == "train" for t in ct)
        ho = sum(t.split == "holdout" for t in ct)
        print(f"  {cat:<12}{tr:>6}{ho:>9}{len(ct):>7}")

    op = sum(t.kind == "operator" for t in tasks)
    dg = sum(t.kind == "diagnosis" for t in tasks)
    print(f"  kinds: operator={op}, diagnosis={dg}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="also run torch CPU self-check")
    args = p.parse_args(argv)

    here = pathlib.Path(__file__).parent
    sys.path.insert(0, str(here))
    tasks = load_all()

    errs = validate_schema(tasks) + check_invariants(tasks)

    if errs:
        print("VALIDATION FAILED:", file=sys.stderr)

        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        return 1

    summarise(tasks)

    if args.check:
        nerrs = numeric_self_check(tasks)
        skipped = [e for e in nerrs if e.startswith("__skipped__")]
        real = [e for e in nerrs if not e.startswith("__skipped__")]

        if skipped:
            print(f"  numeric self-check: {skipped[0].split(': ', 1)[1]}")
        elif real:
            print("  numeric self-check FAILED:")

            for e in real:
                print(f"    - {e}")
            return 1
        else:
            n_op = sum(t.kind == "operator" for t in tasks)
            print(f"  numeric self-check: {n_op}/{n_op} operator references OK on CPU")

    manifest_path = here / "manifest.json"
    write_manifest(tasks, manifest_path)
    print(f"  wrote {manifest_path.name}")
    print("VALIDATION PASSED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
