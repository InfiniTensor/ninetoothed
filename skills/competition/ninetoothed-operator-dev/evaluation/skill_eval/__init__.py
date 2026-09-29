"""
skill_eval — KernelSwift + Ascend-slide inspired evaluation utilities for NineToothed.

Modules:
    robust_bench         - outlier-removed timing + reward-hacking bandwidth check
    reward_hacking_guard - AST static + dynamic + ncu stub

Note: failure_classifier lives in the skill's own `scripts/failure_classifier.py`
(it is an agent-runtime helper, not evaluation-side tooling). Consumers that need
it should import it from there, not from this package.
"""

from .reward_hacking_guard import (
    HackingReport,
    dynamic_analysis,
    full_guard,
    static_analysis,
)
from .robust_bench import (
    GPU_RIDGE,
    BenchResult,
    RewardHackingError,
    compare_robust,
    detect_gpu,
    measure_peak_bw,
    robust_benchmark,
)

__all__ = [
    "HackingReport",
    "dynamic_analysis",
    "full_guard",
    "static_analysis",
    "GPU_RIDGE",
    "BenchResult",
    "RewardHackingError",
    "compare_robust",
    "detect_gpu",
    "measure_peak_bw",
    "robust_benchmark",
]
