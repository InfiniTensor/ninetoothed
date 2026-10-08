"""Classify a NineToothed kernel failure as code_error or guidance_error.

failure_classifier.py follows the Ascend slide's insight:

  Code Repair   → fix kernel.py        (agent made a known mistake)
  Prompt Repair → fix references/*.md  (skill text has a gap or ambiguity)

The distinction matters for Stage 3 MOO edit targeting:
  code_error    → prune/substitute target: kernel.py, wrapper.py
  guidance_error → prune/substitute target: references/<family>.md or SKILL.md

A third class is terminal — it has NO repair target on either side:
  dsl_limit     → the task itself needs a feature outside the DSL's
                  expressible set (ninetoothed 0.25.0). Iterating on the
                  kernel or the skill text cannot converge; the correct
                  action is a *declared* PyTorch fallback per SKILL.md
                  Stop Rules ("Declared fallback — never silent").

Classification is heuristic — deliberately simple and auditable.
It is NOT an LLM call; it is a deterministic rule-based classifier so
the Stage 3 loop can attribute failures without spending inference budget.

Heuristics (applied in order, first match wins):
  0. A task feature is in KNOWN_INEXPRESSIBLE → dsl_limit (terminal;
     also available pre-attempt via screen_task_features())
  1. debug_arrangement reports OOB → code_error (arrangement mistake)
  2. Error text matches a known pattern in common-errors.md → code_error
  3. Error text contains an API signature mismatch (unexpected keyword,
     wrong ndim, wrong number of args) → guidance_error (reference
     example may show wrong call signature)
  4. Error text carries an unambiguous expressiveness signature
     (reaching for a missing atomic primitive) → dsl_limit
  5. Same wrong symptom appears ≥ REPEAT_THRESHOLD times across
     different shapes/dtypes → guidance_error (skill didn't prevent it)
  6. Fallback → unknown (escalate to human)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FailureType(str, Enum):
    CODE_ERROR = "code_error"  # Fix kernel.py / wrapper.py.
    GUIDANCE_ERROR = "guidance_error"  # Fix references/<family>.md or SKILL.md.
    DSL_LIMIT = "dsl_limit"  # Terminal: declared fallback, stop iterating.
    UNKNOWN = "unknown"  # Escalate.


# Minimum times a symptom must repeat across distinct (shape, dtype) pairs
# before we consider it a guidance gap rather than a one-off code mistake.
REPEAT_THRESHOLD = 2


# ---------------------------------------------------------------------------
# DSL expressiveness limits (ninetoothed 0.25.0)
#
# Each entry names a task *feature* the DSL cannot express, with the reason
# grounded in the 0.25.0 source (Tensor meta-ops in tensor.py; the AST-traced
# application path; make()'s static symbolic-shape contract; the ntl surface).
# The subjective step — deciding a task spec has one of these features — is
# the agent's; the flag definitions below are crisp so that step is auditable.
# If a flagged feature is present, no amount of kernel or skill-text iteration
# converges: the correct action is a DECLARED PyTorch fallback (SKILL.md
# Stop Rules), never a silent one.
# ---------------------------------------------------------------------------
KNOWN_INEXPRESSIBLE: dict[str, str] = {
    "data_dependent_indexing": (
        "read/write addresses depend on tensor VALUES (gather/scatter driven "
        "by an index tensor). The arrangement is fixed at compile time from "
        "symbolic shapes; ntl exposes no gather/scatter primitive."
    ),
    "many_to_one_scatter": (
        "multiple source elements may write the same output element "
        "(scatter-add with colliding indices, bincount, histogram). The tile "
        "model is one-writer-per-output-tile and no atomic RMW primitive is "
        "exposed — collisions are a data race, not a slow path. A kernel can "
        "look numerically correct on collision-free test data and still be "
        "semantically wrong."
    ),
    "value_dependent_control_flow": (
        "loop bounds or branches depend on tensor VALUES. application() is "
        "AST-traced to Triton source; only tile-shape/meta-driven control "
        "flow is representable."
    ),
    "dynamic_output_shape": (
        "output shape is a function of input values (nonzero, unique, "
        "compacting boolean select). make() requires every tensor's symbolic "
        "shape up front."
    ),
    "cross_tile_communication": (
        "programs must exchange partial results beyond the built-in fold "
        "along a tiled axis (running prefix scan across tiles, global argmax "
        "with index). No inter-program primitive is exposed."
    ),
}

# Error-text signatures that unambiguously indicate an expressiveness wall
# rather than a fixable mistake. Deliberately tight: the primary detector is
# screen_task_features() on the spec, not the error text — so this list holds
# only signatures that CANNOT be produced by ordinary agent mistakes.
#
# Note on what is deliberately NOT here: a `NotImplementedError` naming
# arrangement/application is NOT a dsl_limit signature. ninetoothed 0.25.0
# never raises NotImplementedError from its lowering path (it raises
# ValueError/TypeError/RuntimeError), so that text can only come from an
# agent's own placeholder stub — a trivially-completable state, not an
# expressiveness wall. Classifying it terminal would tell the agent to give up
# on a task it hasn't started. Reaching for a nonexistent `atomic*` primitive,
# by contrast, is a genuine tell that the task needs many-to-one RMW.
_DSL_LIMIT_PATTERNS: list[tuple[str, str]] = [
    (
        r"has no attribute ['\"]atomic",
        "an atomic primitive was needed and does not exist in the ntl surface",
    ),
]


# ---------------------------------------------------------------------------
# Known-error pattern bank  (mirrors common-errors.md)
# If a failure message matches one of these, it is a CODE error — the agent
# failed to follow existing, clearly documented guidance.
# ---------------------------------------------------------------------------
_CODE_ERROR_PATTERNS: list[tuple[str, str]] = [
    # (Regex_pattern, description).
    (
        r"TypeError.*not iterable",
        "arrangement returned bare Tensor, not tuple (error #1)",
    ),
    (
        r"F841.*output.*assigned.*never used",
        "missing # noqa: F841 on output assignment (error #2)",
    ),
    (r"RecursionError", "output over-arranged to mirror input blocks (error #6)"),
    (r"unsqueeze.*eval", "unsqueeze in arrangement eval failure (error #7)"),
    (
        r"rank.*mismatch|ndim.*mismatch|shape.*mismatch.*Tensor\(\d\)",
        "multi-dim tensor passed to wrong-rank kernel (error #8)",
    ),
    (
        r"BLOCK_SIZE.*constexpr.*block_size",
        "constexpr vs block_size confusion (error #9)",
    ),
    (r"NaN.*output|nan.*values", "fp16 accumulation without upcast (error #5)"),
    (
        r"allclose.*failed|assert.*close.*fail",
        "correctness failure — check dtype tolerance and upcast",
    ),
    (
        r"other.*float.*inf|mask.*fill",
        "missing other=float('-inf') on softmax/max input (error #4)",
    ),
]

# ---------------------------------------------------------------------------
# Guidance-error API-mismatch patterns  (skill reference shows wrong idiom).
# ---------------------------------------------------------------------------
_GUIDANCE_ERROR_PATTERNS: list[tuple[str, str]] = [
    (
        r"offsets.*takes \d+ positional argument|takes \d+ positional argument.*offsets",
        "references use offsets(dim) — but 0.25.0 offsets() takes no args; call offsets() with no arguments",
    ),
    (
        r"takes \d+ positional argument.*\d+ given",
        "reference example has wrong argument count — fix references/",
    ),
    (
        r"has no attribute.*tile|has no attribute.*expand|has no attribute.*squeeze",
        "meta-op not available on this Tensor subtype — reference example may be wrong version",
    ),
    (
        r"TypeError.*block_size\(\).*constexpr",
        "block_size() vs Symbol(constexpr=True) confused in reference example",
    ),
    # Generic NineToothed calling-convention gaps — these recur when the skill does not
    # teach the exact launch signature, so they are guidance (skill) gaps, not code slips.
    (
        r"unexpected keyword argument",
        "kernel launched with a keyword arg it doesn't accept — pass ALL tensors "
        "positionally, e.g. kernel(a, b, output, BLOCK_SIZE=...), never output=output",
    ),
    (
        r"got multiple values for argument",
        "mixed positional/keyword launch — pass tensors positionally in arrangement order",
    ),
    (
        r"(ndim|rank|dimension).*(mismatch|match)|expected \d+ dim|got \d+ dim",
        "tensor rank does not match the kernel's tile rank — tile every dim of the real "
        "shape (a 2-D input needs a 2-D tile, e.g. tile((BM, BN)))",
    ),
    (
        r"MERE=nan|nan.*thr|got nan",
        "NaN in output usually means part of the tensor was never written — the kernel's "
        "tile rank/grid does not cover the full (multi-dim) shape",
    ),
]


@dataclass
class FailureClassification:
    failure_type: FailureType
    evidence: str  # What triggered the rule.
    repair_target: (
        str  # Repair targets: `kernel.py` | "references/<X>.md" | `SKILL.md`.
    )
    repair_hint: str  # One-line suggestion for what to change.
    repeat_count: int = 0  # How many times this symptom was seen.
    matched_rule: str = ""  # Which rule fired.

    def __str__(self) -> str:
        return (
            f"[{self.failure_type.value.upper()}] {self.evidence}\n"
            f"  repair_target : {self.repair_target}\n"
            f"  hint          : {self.repair_hint}\n"
            f"  rule          : {self.matched_rule}"
        )


def screen_task_features(
    task_features: set[str] | frozenset[str] | list[str],
) -> Optional[FailureClassification]:
    """
    Pre-attempt expressibility screen (SKILL.md workflow step 2).

    Call BEFORE writing any kernel, with the feature flags extracted from the
    task spec. Returns a terminal DSL_LIMIT classification when any flag is in
    KNOWN_INEXPRESSIBLE, else None (proceed to implement). Deterministic given
    the flags; the flag definitions live in KNOWN_INEXPRESSIBLE.
    """
    hits = sorted(set(task_features) & set(KNOWN_INEXPRESSIBLE))

    if not hits:
        return None

    reasons = "; ".join(f"{h}: {KNOWN_INEXPRESSIBLE[h]}" for h in hits)

    return FailureClassification(
        failure_type=FailureType.DSL_LIMIT,
        evidence=f"task feature(s) outside the DSL's expressible set: {', '.join(hits)}",
        repair_target="wrapper.py (declared fallback) + task report",
        repair_hint=(
            "Stop kernel iteration. Write the declared-fallback note (SKILL.md "
            "Stop Rules) naming the feature and why it is inexpressible, then "
            "implement the PyTorch fallback in wrapper.py with the "
            "'# ninetoothed-fallback:' marker so the legality gate can tell a "
            f"declared fallback from a disguised one. Basis: {reasons}"
        ),
        matched_rule="expressibility_screen:" + "+".join(hits),
    )


def classify(
    error_text: str,
    oob_count: int = 0,
    symptom_history: Optional[list[str]] = None,
    family: str = "unknown",
    task_features: Optional[set[str]] = None,
) -> FailureClassification:
    """
    Classify a kernel failure.

    Args:
        error_text       : Full error / assertion message from pytest or kernel run.
        oob_count        : Output of debug_arrangement (0 = no OOB).
        symptom_history  : List of error messages from PREVIOUS iterations of the
                           SAME operator (same session). Used for repeat detection.
        family           : Operator family ("elementwise" | "reduction" | "layout" | "perf-diag").
        task_features    : Optional feature flags extracted from the task spec
                           (see KNOWN_INEXPRESSIBLE). If any flagged feature is
                           inexpressible, the failure is terminal (dsl_limit)
                           regardless of the error text — fixing an incidental
                           bug cannot make an inexpressible task converge.

    Returns:
        FailureClassification
    """
    history = symptom_history or []

    # ---- Rule 0: spec-level expressibility screen → terminal dsl_limit ----
    if task_features:
        screened = screen_task_features(task_features)

        if screened is not None:
            return screened

    # ---- Rule 1: OOB from debug_arrangement → always code error ----
    if oob_count > 0:
        return FailureClassification(
            failure_type=FailureType.CODE_ERROR,
            evidence=f"debug_arrangement reported {oob_count} OOB accesses",
            repair_target="kernel.py (arrangement)",
            repair_hint=(
                "Fix the tile hierarchy: check squeeze/expand dims, "
                "ensure output arrangement is independent of input block structure."
            ),
            matched_rule="oob_from_debug_arrangement",
        )

    # ---- Rule 2: known common-errors.md pattern → code error ----
    for pattern, description in _CODE_ERROR_PATTERNS:
        if re.search(pattern, error_text, re.IGNORECASE):
            return FailureClassification(
                failure_type=FailureType.CODE_ERROR,
                evidence=f"matched known error pattern: {description}",
                repair_target="kernel.py",
                repair_hint=(
                    f"Apply the fix documented in references/common-errors.md: {description}"
                ),
                matched_rule=f"known_pattern:{pattern[:40]}",
            )

    # ---- Rule 3: guidance/API mismatch → guidance error ----
    for pattern, description in _GUIDANCE_ERROR_PATTERNS:
        if re.search(pattern, error_text, re.IGNORECASE):
            return FailureClassification(
                failure_type=FailureType.GUIDANCE_ERROR,
                evidence=f"API mismatch pattern: {description}",
                repair_target=f"references/{family}.md",
                repair_hint=(
                    f"The reference example in references/{family}.md likely shows "
                    f"an incorrect API call. Fix: {description}"
                ),
                matched_rule=f"api_mismatch:{pattern[:40]}",
            )

    # ---- Rule 3b: unambiguous expressiveness signature → dsl_limit ----
    # Checked BEFORE the repeat rule: an inexpressible construct repeats too,
    # and misclassifying it as a guidance gap would send the Stage 3 loop
    # editing skill text that can never fix it.
    for pattern, description in _DSL_LIMIT_PATTERNS:
        if re.search(pattern, error_text, re.IGNORECASE):
            return FailureClassification(
                failure_type=FailureType.DSL_LIMIT,
                evidence=f"expressiveness signature: {description}",
                repair_target="wrapper.py (declared fallback) + task report",
                repair_hint=(
                    "This is a DSL expressiveness wall, not a fixable bug. "
                    "Stop iterating; write the declared-fallback note "
                    "(SKILL.md Stop Rules) and a PyTorch fallback in "
                    "wrapper.py carrying the '# ninetoothed-fallback:' marker."
                ),
                matched_rule=f"dsl_limit:{pattern[:40]}",
            )

    # ---- Rule 4: repeat symptom → guidance error ----
    # If the SAME class of error (first 80 chars) has appeared ≥ REPEAT_THRESHOLD
    # times across different iterations, the skill text didn't prevent it.
    symptom_key = error_text.strip()[:80]
    repeat_count = sum(1 for h in history if h.strip()[:80] == symptom_key)

    if repeat_count >= REPEAT_THRESHOLD:
        return FailureClassification(
            failure_type=FailureType.GUIDANCE_ERROR,
            evidence=(
                f"Same symptom appeared {repeat_count + 1} times across iterations "
                f"(threshold={REPEAT_THRESHOLD}) — skill text did not prevent it"
            ),
            repair_target=f"references/{family}.md",
            repair_hint=(
                "Add a more prominent warning or concrete example to "
                f"references/{family}.md covering this failure pattern. "
                "Consider making it the first pitfall in the 'Pitfalls' section."
            ),
            repeat_count=repeat_count + 1,
            matched_rule="repeat_symptom",
        )

    # ---- Rule 5: fallback ----
    return FailureClassification(
        failure_type=FailureType.UNKNOWN,
        evidence=f"no rule matched; error: {error_text[:120]}",
        repair_target="human review",
        repair_hint=(
            "Neither a known code pattern nor a repeated symptom. "
            "Add to common-errors.md after root-cause analysis."
        ),
        matched_rule="fallback",
    )


def classify_batch(
    failures: list[dict],
    family: str = "unknown",
) -> list[FailureClassification]:
    """
    Classify a list of failure dicts from run_correctness_matrix output.

    Each dict should have keys: 'error_text', 'oob_count' (optional).
    Builds the symptom_history incrementally so repeat detection works.
    """
    results = []
    history: list[str] = []

    for f in failures:
        clf = classify(
            error_text=f.get("error_text", ""),
            oob_count=f.get("oob_count", 0),
            symptom_history=history,
            family=family,
        )
        results.append(clf)
        history.append(f.get("error_text", ""))
    return results


def summary(classifications: list[FailureClassification]) -> dict:
    """Return counts by failure type — useful for the A/B comparison table."""
    from collections import Counter

    counts = Counter(c.failure_type.value for c in classifications)
    repair_targets = Counter(c.repair_target for c in classifications)

    return {
        "total": len(classifications),
        "code_errors": counts.get("code_error", 0),
        "guidance_errors": counts.get("guidance_error", 0),
        "dsl_limits": counts.get("dsl_limit", 0),
        "unknown": counts.get("unknown", 0),
        "top_repair_targets": repair_targets.most_common(3),
    }
