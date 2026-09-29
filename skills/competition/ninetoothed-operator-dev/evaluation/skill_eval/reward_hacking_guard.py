"""reward_hacking_guard.py — KernelSwift-inspired static + dynamic analysis, plus a MusaCoder/KernelBench-style legality (anti-cheat) guard.

KernelSwift (Shanghai AI Lab) identifies three techniques for detecting
kernels that appear faster than they should be:
  1. Static code analysis via AST
  2. Dynamic runtime analysis
  3. NCU roofline sanity check

This module implements 1 and 2 without ncu (which requires root / profiling
permissions). An optional ncu wrapper is provided as a stub.

Static checks (AST-based) that can detect hacking patterns:
  - 'return None' or early return before any store
  - Zero-tensor shortcut (output never written)
  - Hardcoded constant store (output = torch.zeros / torch.ones)
  - Kernel with 0 tl.store calls (for trivially optimised-away stores)

Dynamic checks (runtime, no GPU profiler required):
  - Output is all-zero when input is non-zero (data-independent output)
  - Output shape mismatch (returned wrong buffer)
  - Output dtype mismatch
  - Output is identical to input (no-op)

Legality (anti-cheat) check — see `banned_fallback_analysis` below — is a
fourth, independent technique borrowed from KernelBench's task contract
(`ModelNew.forward()` must not fall back to the reference op) and MusaCoder's
MooreEval sandbox (静态分析 + 运行时 profiling 检测被禁 `aten::*`，命中即零奖励):
a solution that reaches for a high-level PyTorch/aten matmul, conv, or
reduction call instead of driving the NineToothed kernel can pass the
numeric-correctness oracle without ever exercising the kernel it was asked
to write. This module only implements the static half (AST scan); there is
no dynamic/profiler-based confirmation, unlike KernelSwift checks 1–2 above.
"""

from __future__ import annotations

import ast
import pathlib
import re
import warnings
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HackingReport:
    static_flags: list[str] = field(default_factory=list)
    dynamic_flags: list[str] = field(default_factory=list)
    ncu_flags: list[str] = field(default_factory=list)
    legality_flags: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not (
            self.static_flags
            or self.dynamic_flags
            or self.ncu_flags
            or self.legality_flags
        )

    def __str__(self) -> str:
        if self.is_clean:
            return "reward_hacking: CLEAN"

        lines = ["reward_hacking: SUSPICIOUS"]

        for f in self.static_flags:
            lines.append(f"  [static]   {f}")

        for f in self.dynamic_flags:
            lines.append(f"  [dynamic]  {f}")

        for f in self.ncu_flags:
            lines.append(f"  [ncu]      {f}")

        for f in self.legality_flags:
            lines.append(f"  [legality] {f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Static analysis (AST) — applied to the generated Triton source.
# ---------------------------------------------------------------------------


class _StaticAnalyzer(ast.NodeVisitor):
    def __init__(self) -> None:
        self.tl_store_count = 0
        self.tl_load_count = 0
        self.zeros_patterns = 0  # Count of tl.zeros / torch.zeros appearing as stores.
        self.early_returns = 0
        self.constexpr_outputs = 0  # Outputs that are constants.

    def visit_Call(self, node: ast.Call) -> None:
        name = _dotted(node.func)

        if name.endswith((".store", "tl.store")):
            self.tl_store_count += 1
            # Check if the value being stored is a data-independent constant.

            if node.args:
                val = node.args[-1] if len(node.args) >= 2 else None

                if val is not None:
                    # Case 1: tl.zeros(...) / tl.ones(...) / tl.full(...).
                    if isinstance(val, ast.Call):
                        vname = _dotted(val.func)

                        if "zeros" in vname or "ones" in vname or "full" in vname:
                            self.zeros_patterns += 1
                    # Case 2: literal constant (0.0, 0, 1.0, True, False).
                    elif isinstance(val, ast.Constant) and isinstance(
                        val.value, (int, float, bool)
                    ):
                        self.zeros_patterns += 1
                    # Case 3: unary minus of a literal (-1.0, -0).
                    elif isinstance(val, ast.UnaryOp) and isinstance(val.op, ast.USub):
                        if isinstance(val.operand, ast.Constant):
                            self.zeros_patterns += 1

        if name.endswith((".load", "tl.load")):
            self.tl_load_count += 1

        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        # An early return (before stores) is suspicious in a kernel context.
        if node.value is None:
            self.early_returns += 1

        self.generic_visit(node)


def _dotted(node: ast.AST) -> str:
    parts = []

    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value

    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def static_analysis(
    source_path: Optional[pathlib.Path] = None, source_code: Optional[str] = None
) -> list[str]:
    """
    Run AST-based static analysis on a generated Triton kernel.

    Pass either a path to the .py file or the source code string.
    Returns a list of flag strings (empty = clean).
    """
    if source_path is not None:
        source_code = pathlib.Path(source_path).read_text(encoding="utf-8")

    if not source_code:
        return ["no_source: cannot analyse"]

    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        return [f"syntax_error: {e}"]

    a = _StaticAnalyzer()
    a.visit(tree)

    flags = []
    has_kernel = (
        a.tl_load_count > 0
        or a.tl_store_count > 0
        or "triton.jit" in source_code
        or "@jit" in source_code
    )

    if has_kernel:
        # Only flag missing stores if we have evidence of a real kernel body.
        if a.tl_store_count == 0:
            flags.append(
                "no_tl_store: kernel has no tl.store calls — output never written"
            )

        if a.tl_load_count == 0:
            flags.append("no_tl_load: kernel has no tl.load calls — input never read")

    if a.zeros_patterns > 0:
        flags.append(
            f"constant_store: {a.zeros_patterns} store(s) write zeros/ones/full "
            "constants — output may be data-independent"
        )

    ratio = a.tl_load_count / max(a.tl_store_count, 1)

    if a.tl_store_count > 0 and ratio > 32:
        flags.append(
            f"high_load_store_ratio: {a.tl_load_count} loads vs {a.tl_store_count} stores "
            f"(ratio {ratio:.1f}) — possible redundant loads"
        )
    return flags


# ---------------------------------------------------------------------------
# 2. Dynamic analysis (runtime, no profiler).
# ---------------------------------------------------------------------------


def dynamic_analysis(
    kernel_fn,  # Callable: () -> None (in-place).
    output_tensor,  # The pre-allocated output tensor (post-call).
    input_tensor=None,  # First input tensor (for no-op detection).
    rtol: float = 0.0,
    atol: float = 1e-6,
) -> list[str]:
    """
    Run the kernel and check for data-independence / no-op patterns.

    Args:
        kernel_fn    : the kernel callable (run ONCE to produce output_tensor).
        output_tensor: the output tensor AFTER the kernel has run.
        input_tensor : optional first input (for no-op check).
        rtol, atol   : tolerance for "all-close to zero/input" checks.

    Returns list of flag strings (empty = clean).

    Important: this checks the OUTPUT TENSOR already filled by the caller.
    It does NOT call kernel_fn again — to avoid timing interference.
    """
    try:
        import torch
    except ImportError:
        return ["no_torch: dynamic analysis requires torch"]

    flags = []
    out = output_tensor

    # Check 1: output is all-zero when we expect non-zero values.
    if out.numel() > 0:
        max_abs = out.abs().max().item()

        if max_abs < atol:
            flags.append(
                f"all_zero_output: max|output|={max_abs:.2e} — "
                "kernel may have skipped computation"
            )

    # Check 2: output is NaN or Inf (failed numerical op).
    if torch.isnan(out).any():
        flags.append("nan_in_output: kernel produced NaN values")

    if torch.isinf(out).any():
        flags.append("inf_in_output: kernel produced Inf values")

    # Check 3: no-op check — output is identical to input.
    if input_tensor is not None and input_tensor.shape == out.shape:
        try:
            if torch.allclose(out, input_tensor.to(out.dtype), rtol=0, atol=atol):
                flags.append(
                    "no_op_output: output == input within tolerance — "
                    "kernel may not be computing anything"
                )
        except Exception:
            pass

    return flags


# ---------------------------------------------------------------------------
# 3. NCU roofline stub (requires ncu profiler, hardware-specific).
# ---------------------------------------------------------------------------


def ncu_roofline_check(
    kernel_fn,
    expected_flops: int,
    expected_bytes: int,
    gpu: Optional[str] = None,
    tolerance: float = 1.1,
) -> list[str]:
    """
    Stub: run NCU roofline sanity check if ncu is available.

    KernelSwift uses ncu roofline as the third layer of reward hacking
    detection. This stub attempts to run ncu but gracefully returns an
    'unchecked' notice if ncu is not available.

    Full implementation would: run `ncu --metrics ...` in a subprocess,
    parse achieved_occupancy / sm_throughput / dram_read_throughput, and
    compare against the roofline bound.

    Args:
        tolerance: if achieved FLOPs > expected * tolerance, flag as hacking.
    """
    import shutil

    if not shutil.which("ncu"):
        return ["ncu_unavailable: install CUDA profiler for ncu roofline check"]

    # Placeholder: in production, build the ncu command and parse XML output.
    warnings.warn(
        "The ncu_roofline_check found ncu but full implementation is a stub. "
        "Implement subprocess ncu call and output parser for your environment.",
        stacklevel=2,
    )

    return ["ncu_stub: ncu available but parser not implemented"]


# ---------------------------------------------------------------------------
# 4. Legality guard — banned high-level fallback ops
#    (MusaCoder/KernelBench convention: matmul/conv/reduction/normalization
#    must go through the NineToothed kernel under test, not a PyTorch/aten
#    fallback that would make correctness pass for the wrong reason).
# ---------------------------------------------------------------------------

# High-level compute ops that would let a solution pass correctness without
# ever invoking a NineToothed kernel. Mirrors MusaCoder's aten::* ban (matmul/
# conv/reduce family all banned); shape query, view/reshape, copy, and tensor
# creation stay legal (musacoder.md: "白名单仅放行非计算工具").
BANNED_FALLBACK_OPS = {
    # Matmul family.
    "matmul",
    "mm",
    "bmm",
    "addmm",
    "baddbmm",
    "einsum",
    # Conv family.
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    # Reduction family.
    "sum",
    "mean",
    "prod",
    "amax",
    "amin",
    "var",
    "std",
    "cumsum",
    "cumprod",
    "logsumexp",
    "norm",
    # Fused high-level ops that would trivially satisfy correctness.
    "softmax",
    "log_softmax",
    "layer_norm",
    "batch_norm",
    "scaled_dot_product_attention",
    "linear",
}

# NineToothed's own kernel DSL is accessed as `ntl.sum(...)` /
# `ninetoothed.language.sum(...)` inside a kernel's `application()` (see any
# examples/*/kernel.py) — that is the REQUIRED in-kernel form of a reduction,
# not a fallback, so calls qualified by these names are never flagged.
_NT_DSL_QUALIFIERS = ("ntl", "ninetoothed")

# Declared-fallback marker (SKILL.md Stop Rules, "Declared fallback — never
# silent"). When a task is outside the DSL's expressible set, the wrapper may
# legitimately implement the op in PyTorch — but MUST say so, on a comment
# line of this form, naming the inexpressible feature:
#
#     # ninetoothed-fallback: <feature> — <one-line reason>
#
# The legality gate then reports the torch.* usage with a `declared_fallback:`
# prefix instead of `banned_fallback:` — a signal to route the file to human
# adjudication, NOT an automatic pass. Two guards keep the marker from becoming
# a one-line compliance-dodge:
#   (1) <feature> must be one of the recognised inexpressible features
#       (_DECLARABLE_FEATURES, the keys of failure_classifier.KNOWN_INEXPRESSIBLE
#       — kept in sync by test). A garbage feature name (`laziness`) is treated
#       as no declaration at all: the hits stay `banned_fallback:`.
#   (2) A declared fallback still fails the "implemented a NineToothed kernel"
#       bar, so completion stays capped low regardless — there is no score to
#       be gained by declaring on an expressible task, only a low one to accept.
# Whether a declared fallback is *appropriate* for THIS task (vs. a lazy dodge
# on an expressible one) is a judgement the static scan cannot make and does
# not pretend to: see tests/verifier_spec.md, which routes it to review.
_DECLARED_FALLBACK_RE = re.compile(
    r"^\s*#\s*ninetoothed-fallback\s*:\s*(?P<feature>[\w\-]+)\s*[—-]\s*(?P<reason>.+)$",
    re.MULTILINE,
)

# Mirror of failure_classifier.KNOWN_INEXPRESSIBLE keys. Kept here (not imported)
# because failure_classifier lives under scripts/ and this module under
# evaluation/skill_eval/; a cross-package import would be fragile. The two are
# asserted equal by the offline test suite, so drift is caught, not silent.
_DECLARABLE_FEATURES = frozenset(
    {
        "data_dependent_indexing",
        "many_to_one_scatter",
        "value_dependent_control_flow",
        "dynamic_output_shape",
        "cross_tile_communication",
    }
)


def declared_fallback_note(source_code: str) -> Optional[str]:
    """Return '<feature> — <reason>' if the source carries a VALID declared-fallback marker (feature in _DECLARABLE_FEATURES), else None.

    An unrecognised feature name is deliberately treated as no declaration:
    the marker only earns the softer `declared_fallback:` prefix when it names
    an actually-inexpressible feature, so it cannot launder a torch fallback on
    an expressible task.
    """
    for m in _DECLARED_FALLBACK_RE.finditer(source_code):
        if m.group("feature") in _DECLARABLE_FEATURES:
            return f"{m.group('feature')} — {m.group('reason').strip()}"
    return None


class _LegalityAnalyzer(ast.NodeVisitor):
    def __init__(self) -> None:
        self.hits: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        dotted = _dotted(node.func)

        if dotted and "." in dotted:
            qualifier, _, leaf = dotted.rpartition(".")

            if leaf in BANNED_FALLBACK_OPS and not (
                qualifier == "ntl" or qualifier.split(".")[0] in _NT_DSL_QUALIFIERS
            ):
                self.hits.append(f"{dotted}(...) at line {node.lineno}")

        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.MatMult):
            self.hits.append(f"'@' matmul operator at line {node.lineno}")

        self.generic_visit(node)


def banned_fallback_analysis(
    source_path: Optional[pathlib.Path] = None, source_code: Optional[str] = None
) -> list[str]:
    """Scan solution source for banned high-level fallback ops (matmul/conv/reduction/normalization/attention), mirroring MusaCoder/KernelBench's `aten::*` ban and its "命中即零奖励" (a hit zeroes the reward) verdict.

    Heuristic and AST-based, in the same spirit as static_analysis() above —
    no type inference, so a bare `x.sum(...)` is flagged regardless of what
    `x` actually is. The one deliberate exception is the `ntl.*` /
    `ninetoothed.*` namespace, which is how a legitimate in-kernel reduction
    is written and must never be flagged.

    Returns a list of flag strings (empty = clean / legal). Hits are prefixed
    `banned_fallback:` normally, or `declared_fallback:` when the file carries
    the `# ninetoothed-fallback:` marker (see declared_fallback_note) — the
    caller can then keep the completion cap but skip the compliance penalty
    for a disclosed inability.
    """
    if source_path is not None:
        source_code = pathlib.Path(source_path).read_text(encoding="utf-8")

    if not source_code:
        return []

    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        return [f"syntax_error: {e}"]

    a = _LegalityAnalyzer()
    a.visit(tree)
    note = declared_fallback_note(source_code)

    if note is not None:
        return [f"declared_fallback: {h} (declared: {note})" for h in a.hits]
    return [f"banned_fallback: {h}" for h in a.hits]


# ---------------------------------------------------------------------------
# Full guard: combine all three checks.
# ---------------------------------------------------------------------------


def full_guard(
    kernel_fn,
    output_tensor,
    input_tensor=None,
    generated_source_path: Optional[pathlib.Path] = None,
    solution_source_paths: Optional[list[pathlib.Path]] = None,
    bytes_moved: int = 0,
    flops: int = 0,
    gpu: Optional[str] = None,
    run_ncu: bool = False,
) -> HackingReport:
    """
    Run all reward-hacking + legality checks and return a HackingReport.

    Intended to be called once after correctness validation and before
    submitting benchmark results to the evaluator.

    Args:
        solution_source_paths: the agent-produced .py files (e.g. kernel.py,
            wrapper.py) to scan for banned `aten::*`-style fallback ops. Pass
            all of them — the fallback is as likely to hide in the wrapper as
            in the kernel itself.
    """
    report = HackingReport()

    if generated_source_path is not None:
        report.static_flags = static_analysis(source_path=generated_source_path)

    report.dynamic_flags = dynamic_analysis(kernel_fn, output_tensor, input_tensor)

    if run_ncu and bytes_moved > 0 and flops > 0:
        report.ncu_flags = ncu_roofline_check(kernel_fn, flops, bytes_moved, gpu)

    for p in solution_source_paths or []:
        report.legality_flags.extend(banned_fallback_analysis(source_path=p))

    return report
