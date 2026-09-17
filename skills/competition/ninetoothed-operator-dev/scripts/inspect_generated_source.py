#!/usr/bin/env python3
"""Inspect the Triton source NineToothed caches at ~/.ninetoothed/<sha256>.py.

Read-only. Parses with `ast` (no execution of the cached code). Reports the
Triton ops used, tile / num_warps / num_stages hints, and load/store counts —
the evidence you cite when judging whether a kernel is reasonable or regressed.

`--contract <name>` additionally checks a SEMANTIC CONTRACT: does the
generated source actually contain the primitive the claimed semantics
requires? A numerically green kernel can still be semantically wrong — the
canonical case is a scatter-class kernel that passes on collision-free test
data while using plain stores (race-prone), or a "reduction" that copies
through without any fold. Correctness matrices sample inputs; the contract
check reads what was compiled.

Usage:
    python inspect_generated_source.py                 # newest cached kernel
    python inspect_generated_source.py --digest <hex>  # a specific kernel
    python inspect_generated_source.py --list          # list cached kernels
    python inspect_generated_source.py --cache-dir DIR  # override cache dir
    python inspect_generated_source.py --contract reduction   # semantic check

Exit codes: 0 = ok / contract satisfied, 1 = no cached kernel found,
2 = contract violated.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import sys
from collections import Counter


def default_cache_dir() -> pathlib.Path:
    return pathlib.Path.home() / ".ninetoothed"


def list_cached(cache_dir: pathlib.Path) -> list[pathlib.Path]:
    if not cache_dir.is_dir():
        return []
    return sorted(cache_dir.glob("*.py"), key=lambda p: p.stat().st_mtime, reverse=True)


def pick_file(cache_dir: pathlib.Path, digest: str | None) -> pathlib.Path | None:
    files = list_cached(cache_dir)

    if not files:
        return None

    if digest is None:
        return files[0]

    for f in files:
        if f.stem == digest or f.stem.startswith(digest):
            return f
    return None


class _Analyzer(ast.NodeVisitor):
    """Collect call names, decorator kwargs, and numeric constants."""

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.kw_consts: dict[str, list] = {}
        self.int_consts: Counter[int] = Counter()
        self.masked_loads = 0
        self.masked_stores = 0

    def visit_Call(self, node: ast.Call) -> None:
        name = _dotted_name(node.func)

        if name:
            self.calls[name] += 1
        # Capture num_warps=/num_stages=/BLOCK_*=... style kwargs.

        for kw in node.keywords:
            if kw.arg and isinstance(kw.value, ast.Constant):
                self.kw_consts.setdefault(kw.arg, []).append(kw.value.value)
        # Boundary-safety evidence: mask= on load/store, any value shape.

        if name and any(kw.arg == "mask" for kw in node.keywords):
            if name.endswith(".load"):
                self.masked_loads += 1
            elif name.endswith(".store"):
                self.masked_stores += 1

        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            self.int_consts[node.value] += 1

        self.generic_visit(node)


def _dotted_name(node: ast.AST) -> str:
    parts: list[str] = []

    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value

    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


# Namespaces that count as in-kernel DSL / device primitives in the generated
# source. `ninetoothed_libdevice` is the alias ninetoothed's generator emits for
# libdevice math (`src/ninetoothed/generation.py`: `LIBDEVICE_ALIAS =
# `ninetoothed_libdevice``; the Tritonizer only rewrites the `ninetoothed`
# name, so the alias survives verbatim). Without it, a correct kernel whose
# `exp`/`rsqrt`/etc. route through libdevice — which the reference docs
# actively steer agents toward — would be invisible to the contract check and
# score a false violation.
_KERNEL_OP_PREFIXES = ("tl.", "ntl.", "triton", "ninetoothed_libdevice.")


def analyze(path: pathlib.Path) -> dict:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    a = _Analyzer()
    a.visit(tree)

    tl_ops = {k: v for k, v in a.calls.items() if k.startswith(_KERNEL_OP_PREFIXES)}
    loads = sum(v for k, v in a.calls.items() if k.endswith(".load"))
    stores = sum(v for k, v in a.calls.items() if k.endswith(".store"))
    dots = sum(v for k, v in a.calls.items() if k.endswith(".dot"))

    interesting = {}

    for key in ("num_warps", "num_stages"):
        if key in a.kw_consts:
            interesting[key] = a.kw_consts[key]

    return {
        "path": str(path),
        "lines": src.count("\n") + 1,
        "tl_ops": dict(sorted(tl_ops.items(), key=lambda kv: -kv[1])),
        "loads": loads,
        "stores": stores,
        "dots": dots,
        "masked_loads": a.masked_loads,
        "masked_stores": a.masked_stores,
        "config_kwargs": interesting,
        "top_int_consts": a.int_consts.most_common(8),
    }


# ---------------------------------------------------------------------------
# Semantic contracts
#
# Each contract states what the generated source MUST contain for the claimed
# semantics to be real, in terms of leaf op names on the tl.*/ntl. namespace.
# `require_any`  : at least one of these leaves must appear
# `require_all`  : every one of these leaves must appear
# `forbid`       : none of these leaves may appear
# `require_prefix`: at least one leaf starting with this prefix must appear
# Matching is on the leaf (last dotted component), counted only over the
# tl.*/ntl.*/triton-namespaced calls the analyzer already extracts.
# ---------------------------------------------------------------------------
SEMANTIC_CONTRACTS: dict[str, dict] = {
    "reduction": {
        "require_any": ("sum", "max", "min", "dot"),
        "why": (
            "a reduction must fold — generated source with no fold primitive "
            "is a pass-through, however green the matrix looks"
        ),
    },
    "matmul": {
        "require_any": ("dot",),
        "why": "a matmul without tl.dot is either a fallback or a fake",
    },
    "atomic": {
        "require_prefix": "atomic_",
        "why": (
            "many-to-one writes need atomic RMW; plain stores are race-prone "
            "and only look correct on collision-free test data"
        ),
    },
    "stable_softmax": {
        "require_all": ("exp", "max"),
        "why": (
            "numerically stable softmax needs BOTH the exp and the row-max "
            "subtraction; exp alone overflows fp16 on large inputs"
        ),
    },
    "elementwise": {
        "forbid": ("dot",),
        "require_loads_stores": True,
        "why": (
            "an elementwise kernel must move data (loads+stores) and has no "
            "business containing a matmul primitive"
        ),
    },
}


def check_contract(info: dict, name: str) -> dict:
    """Evaluate one semantic contract against analyze() output.

    Returns {"name", "satisfied": bool, "evidence": [...], "missing": [...]}.
    """
    spec = SEMANTIC_CONTRACTS[name]
    leaves: Counter[str] = Counter()

    for op, n in info["tl_ops"].items():
        leaves[op.rsplit(".", 1)[-1]] += n

    evidence: list[str] = []
    missing: list[str] = []

    if "require_any" in spec:
        found = [
            f"{leaf} x{leaves[leaf]}" for leaf in spec["require_any"] if leaves[leaf]
        ]

        if found:
            evidence.extend(found)
        else:
            missing.append("none of: " + "/".join(spec["require_any"]))

    if "require_all" in spec:
        for leaf in spec["require_all"]:
            if leaves[leaf]:
                evidence.append(f"{leaf} x{leaves[leaf]}")
            else:
                missing.append(leaf)

    if "require_prefix" in spec:
        found = [
            f"{leaf} x{n}"
            for leaf, n in leaves.items()
            if leaf.startswith(spec["require_prefix"])
        ]

        if found:
            evidence.extend(found)
        else:
            missing.append(
                f"no {spec['require_prefix']}* call (plain stores: {info['stores']})"
            )

    if "forbid" in spec:
        for leaf in spec["forbid"]:
            if leaves[leaf]:
                missing.append(f"forbidden {leaf} present x{leaves[leaf]}")

    if spec.get("require_loads_stores"):
        if info["loads"] >= 1 and info["stores"] >= 1:
            evidence.append(f"loads={info['loads']} stores={info['stores']}")
        else:
            missing.append(
                f"loads={info['loads']} stores={info['stores']} (needs >=1 of each)"
            )

    return {
        "name": name,
        "satisfied": not missing,
        "evidence": evidence,
        "missing": missing,
        "why": spec["why"],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--digest", default=None, help="sha256 (or prefix) of a cached kernel"
    )
    p.add_argument("--cache-dir", default=None, help="override ~/.ninetoothed")
    p.add_argument("--list", action="store_true", help="list cached kernels and exit")
    p.add_argument(
        "--contract",
        default=None,
        choices=sorted(SEMANTIC_CONTRACTS),
        help="also check a semantic contract; exits 2 if violated",
    )
    args = p.parse_args(argv)

    cache_dir = pathlib.Path(args.cache_dir) if args.cache_dir else default_cache_dir()

    if args.list:
        files = list_cached(cache_dir)

        if not files:
            print(f"no cached kernels under {cache_dir}")

            return 1

        for f in files:
            print(f"{f.stem}\t{f.stat().st_size:>8} B\t{f.name}")
        return 0

    target = pick_file(cache_dir, args.digest)

    if target is None:
        print(
            f"no matching cached kernel under {cache_dir} "
            f"(build/run a kernel first, then re-run)",
            file=sys.stderr,
        )

        return 1

    info = analyze(target)
    print(f"# generated source: {info['path']}  ({info['lines']} lines)")
    print(
        f"loads={info['loads']}  stores={info['stores']}  dots={info['dots']}  "
        f"masked={info['masked_loads']}/{info['masked_stores']} (of loads/stores)"
    )

    if info["config_kwargs"]:
        print(f"config kwargs: {info['config_kwargs']}")

    print("triton ops:")

    for op, n in info["tl_ops"].items():
        print(f"  {op:<28} x{n}")

    print(f"top int constants (tile/size hints): {info['top_int_consts']}")

    if args.contract:
        result = check_contract(info, args.contract)

        if result["satisfied"]:
            print(
                f"contract[{args.contract}]: SATISFIED — "
                + ", ".join(result["evidence"])
            )

            return 0

        print(f"contract[{args.contract}]: VIOLATED — " + "; ".join(result["missing"]))
        print(f"  why it matters: {result['why']}")

        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
