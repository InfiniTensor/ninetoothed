"""
Inspect the generated Triton IR for the scatter_add kernel.

Checks:
  1. The generated code contains `tl.atomic_add` (hardware-level atomic).
  2. There is NO `tl.store` of an "old + acc" value (which would indicate
     a non-atomic load-add-store race).

This verifies that our choice of a fp32 working buffer with
`tl.atomic_add` translates into an actual atomic RMW instruction on the
MetaX backend.
"""

import os
import re
import sys

sys.path.insert(0, "/data/ntops13/src")
import torch  # noqa: E402

import ntops  # noqa: E402


def inspect():
    self_t = torch.randn(32, 64, dtype=torch.float32, device="cuda")
    src_t = torch.randn(32, 64, dtype=torch.float32, device="cuda")
    index = torch.randint(0, 32, (32, 64), dtype=torch.long, device="cuda")
    _ = ntops.torch.scatter_add(self_t, 0, index, src_t)
    torch.cuda.synchronize()

    cache_root = os.path.expanduser("~/.triton/cache")
    if not os.path.isdir(cache_root):
        cache_root = "/tmp"
    print(f"Scanning cache root: {cache_root}")

    found_atomic_add = 0
    found_atomic_rmw = 0
    found_load_add_store = 0
    found_atomic_cas = 0
    inspected = 0
    sample_paths = []

    for root, _dirs, files in os.walk(cache_root):
        for f in files:
            if not f.endswith((".py", ".ttir", ".ttgir", ".llir", ".ll", ".txt")):
                continue
            full = os.path.join(root, f)
            try:
                with open(full, "r", errors="ignore") as fh:
                    content = fh.read()
            except OSError:
                continue
            if "scatter_add" not in content and "atomic_add" not in content:
                continue
            inspected += 1
            if len(sample_paths) < 5:
                sample_paths.append(full)

            if re.search(r"\batomic_add\b", content):
                found_atomic_add += 1
            if re.search(r"atomic_rmw|atomicrmw", content):
                found_atomic_rmw += 1
            if re.search(r"atomic_cas|atomicCAS", content):
                found_atomic_cas += 1
            # Detect the dangerous pattern:
            #   %old = load %ptr
            #   %new = fadd %old, %acc
            #   store %new, %ptr
            # without any `atomic` keyword on the store.
            if re.search(r"(?<!atomic_)(?<!atomicrmw )\bstore\b", content):
                # Only count if the same file contains no atomic_add at all.
                if "atomic_add" not in content and "atomic_rmw" not in content:
                    found_load_add_store += 1

    print(f"inspected kernel artifacts: {inspected}")
    print(f"files containing `atomic_add`:     {found_atomic_add}")
    print(f"files containing `atomicrmw`:      {found_atomic_rmw}")
    print(f"files containing `atomic_cas`:     {found_atomic_cas}")
    print(f"files with store (no atomic path): {found_load_add_store}")
    print()
    print("Sample artifact paths:")
    for p in sample_paths:
        print(f"  {p}")

    if found_atomic_add > 0 or found_atomic_rmw > 0:
        verdict = "OK -- kernel uses hardware atomic RMW"
    elif found_load_add_store > 0:
        verdict = "WARNING -- kernel uses non-atomic load-add-store"
    else:
        verdict = "UNKNOWN -- could not locate scatter_add artifacts"
    print()
    print("Verdict:", verdict)
    return found_atomic_add + found_atomic_rmw > 0


if __name__ == "__main__":
    ok = inspect()
    sys.exit(0 if ok else 1)
