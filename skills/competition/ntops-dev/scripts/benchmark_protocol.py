#!/usr/bin/env python3
"""Reusable fixed-protocol CUDA benchmark helpers for ntops candidates."""

import argparse
import json
import statistics


def _measure_ms(torch, fn, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def benchmark_pair(torch, baseline, candidate, *, warmup=20, iterations=200, rounds=3):
    """Measure equivalent callables in alternating order and return raw/median data."""
    if warmup < 0 or iterations < 1 or rounds < 3:
        raise ValueError("Require warmup >= 0, iterations >= 1, and rounds >= 3")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark protocol")

    for _ in range(warmup):
        baseline()
        candidate()
    torch.cuda.synchronize()

    rows = []
    for round_index in range(rounds):
        order = ("baseline", "candidate") if round_index % 2 == 0 else (
            "candidate",
            "baseline",
        )
        measured = {}
        for name in order:
            fn = baseline if name == "baseline" else candidate
            measured[name] = _measure_ms(torch, fn, iterations)
        rows.append(
            {
                "round": round_index + 1,
                "order": list(order),
                "baseline_ms": measured["baseline"],
                "candidate_ms": measured["candidate"],
            }
        )

    baseline_median = statistics.median(row["baseline_ms"] for row in rows)
    candidate_median = statistics.median(row["candidate_ms"] for row in rows)
    improvement = (baseline_median - candidate_median) / baseline_median
    return {
        "warmup": warmup,
        "iterations": iterations,
        "rounds": rows,
        "baseline_median_ms": baseline_median,
        "candidate_median_ms": candidate_median,
        "improvement_fraction": improvement,
    }


def promotion_decision(result, *, minimum_improvement=0.05, correctness=True):
    """Return a conservative keep/revert decision for one benchmark result."""
    if not correctness:
        return {"promote": False, "reason": "correctness failed"}
    improvement = result["improvement_fraction"]
    if improvement < minimum_improvement:
        return {
            "promote": False,
            "reason": (
                f"improvement {improvement:.2%} is below "
                f"the {minimum_improvement:.2%} threshold"
            ),
        }
    return {
        "promote": True,
        "reason": f"improvement {improvement:.2%} meets the threshold",
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Provide importable CUDA-event benchmark_pair() and promotion_decision() "
            "helpers. Run --self-test to validate decision logic without CUDA."
        )
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        parser.print_help()
        return 0

    result = {
        "baseline_median_ms": 1.0,
        "candidate_median_ms": 0.94,
        "improvement_fraction": 0.06,
    }
    decision = promotion_decision(result)
    assert decision["promote"] is True
    assert promotion_decision(result, correctness=False)["promote"] is False
    print(json.dumps({"result": result, "decision": decision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
