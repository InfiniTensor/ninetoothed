#!/usr/bin/env python3
"""Create a self-test task record template."""

from __future__ import annotations

import argparse
from pathlib import Path

TEMPLATE = """# Self-Test: {name}

## Category

{kind}

## Input Task Statement

TODO

## AI Agent Execution Summary

TODO

## Produced Patch Summary

TODO

## Correctness Command

```shell
TODO
```

## Correctness Result

```text
TODO
```

## Benchmark Command

```shell
TODO or not applicable
```

## Benchmark Result

```text
TODO or not applicable
```

## Failure or Regression Diagnosis

Symptom:

```text
TODO or not applicable
```

Root cause:

```text
TODO or not applicable
```

Fix or mitigation:

```text
TODO or not applicable
```

Verification:

```text
TODO or not applicable
```

## Unsupported Cases

TODO
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--kind", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "task.md"

    if path.exists():
        raise SystemExit(f"refusing to overwrite existing file: {path}")

    path.write_text(TEMPLATE.format(name=args.name, kind=args.kind), encoding="utf-8")
    print(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
