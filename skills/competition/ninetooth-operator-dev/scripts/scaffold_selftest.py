#!/usr/bin/env python3
"""Emit a self-test task skeleton."""

from __future__ import annotations

import sys

TEMPLATE = """# Self-Test Task: {title}

## Input Task

- operator requirement:
- inputs, outputs, shape, dtype, broadcast, boundary, layout:
- unsupported scope:

## Agent Execution Summary

- status:
- steps completed:

## Repository Files Inspected

- arrangement/application anchors:
- tensor metadata or load/store anchors:
- generated-source/AOT/benchmark anchors, if relevant:

## Patch Summary

- operator/test/example/fix files:
- implementation shape:

## Correctness Command

## Correctness Result

- command:
- environment:
- cases passed:
- failures:
- fix and rerun, if any:

## Benchmark Command

## Benchmark Result

- baseline:
- input sizes:
- command:
- result or blocker:

## Performance Conclusion

- conclusion:
- generated-source/AOT fallback evidence, if benchmark is blocked:

## Failure Diagnosis

- symptom:
- diagnosis path:
- root cause or blocker:
- minimal fix or workaround:
- rerun command:
- rerun result:

## Risks and Unsupported Scope

- unsupported dtype, shape, layout, device, benchmark, AOT, or generated-source
  scope:
"""


def main() -> int:
    title = " ".join(sys.argv[1:]).strip() or "Untitled Operator Task"
    print(TEMPLATE.format(title=title))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
