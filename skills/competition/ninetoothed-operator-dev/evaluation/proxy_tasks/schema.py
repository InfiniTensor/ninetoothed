"""
Task schema for the proxy task set.

A proxy task stands in for a hidden evaluation task so the skill can be measured
offline (A/B) and the Stage 3 optimizer has a reward signal. Two kinds:

  operator  : the agent must implement a NineToothed operator. Carries a PyTorch
              reference and an input generator; correctness is judged by MERE/MARE
              against the reference (thresholds in the skill's
              run_correctness_matrix.py: fp32 1.22e-4, fp16 9.77e-4, bf16 7.81e-3).
  diagnosis : the agent must diagnose a failing or slow kernel. Carries a scenario
              and a checklist of findings the diagnosis must hit.

Modules are import-safe without torch installed (torch is imported lazily inside
reference / make_inputs), so the manifest and counts can be validated on any
machine; the numeric self-check runs only where torch is present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

CATEGORIES = ("elementwise", "reduction", "layout", "perf_diag")
SPLITS = ("train", "holdout")
KINDS = ("operator", "diagnosis")
DIFFICULTIES = ("easy", "medium", "hard")


@dataclass
class TaskSpec:
    id: str
    category: str
    split: str
    difficulty: str
    name: str
    kind: str
    prompt: str
    # Operator kind.
    reference: Optional[Callable] = None
    make_inputs: Optional[Callable] = None
    dtypes: tuple = ("float32", "float16")
    # Diagnosis kind.
    scenario: Optional[str] = None
    expected_findings: tuple = ()
    buggy_snippet: Optional[str] = None

    def validate(self) -> list[str]:
        errs = []

        if self.category not in CATEGORIES:
            errs.append(f"{self.id}: bad category {self.category!r}")

        if self.split not in SPLITS:
            errs.append(f"{self.id}: bad split {self.split!r}")

        if self.kind not in KINDS:
            errs.append(f"{self.id}: bad kind {self.kind!r}")

        if self.difficulty not in DIFFICULTIES:
            errs.append(f"{self.id}: bad difficulty {self.difficulty!r}")

        if not self.prompt or len(self.prompt) < 10:
            errs.append(f"{self.id}: prompt too short")

        if self.kind == "operator":
            if self.reference is None or self.make_inputs is None:
                errs.append(f"{self.id}: operator task needs reference + make_inputs")
        elif self.kind == "diagnosis":
            if not self.scenario or not self.expected_findings:
                errs.append(
                    f"{self.id}: diagnosis task needs scenario + expected_findings"
                )
        return errs

    def meta(self) -> dict:
        """JSON-serialisable metadata (no callables)."""
        d = {
            "id": self.id,
            "category": self.category,
            "split": self.split,
            "difficulty": self.difficulty,
            "name": self.name,
            "kind": self.kind,
            "prompt": self.prompt,
        }

        if self.kind == "operator":
            d["dtypes"] = list(self.dtypes)
        else:
            d["expected_findings"] = list(self.expected_findings)
        return d


def randn_inputs(*shapes):
    """Make inputs that return randn tensors of the given shapes."""

    def maker(device="cpu", dtype="float32"):
        import torch

        dt = getattr(torch, dtype)

        return tuple(torch.randn(*s, device=device, dtype=dt) for s in shapes)

    return maker
