#!/usr/bin/env python3
"""Emit a structured operator task card from task.md / task.yaml; fail on TBD."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from _task_spec import (
    find_tbd_fields,
    load_task_spec,
    load_task_spec_from_markdown,
    validate_card_text,
)

TEMPLATE = """# Task card

- Generated at (UTC): {timestamp}
- Source: {source}

## Operator

{operator}

## Math definition

{math}

## Inputs

| Tensor | Shape | dtype | Layout |
|--------|-------|-------|--------|
{inputs_rows}

## Outputs

| Tensor | Shape | dtype |
|--------|-------|-------|
{outputs_rows}

## Broadcast

{broadcast}

## Layout / stride / offset

{layout}

## Boundary cases

{boundaries}

## Reference implementation

{reference}

## Tests required

{tests}

## Benchmark required

{benchmark}

## Unsupported / risks

{risks}
"""


def build_card_from_spec(spec) -> str:
    return TEMPLATE.format(
        timestamp=datetime.now(timezone.utc).isoformat(),
        source=spec.source_label,
        operator=spec.operator,
        math=spec.math,
        inputs_rows=spec.inputs_rows,
        outputs_rows=spec.outputs_rows,
        broadcast=spec.broadcast,
        layout=spec.layout,
        boundaries=spec.boundaries,
        reference=spec.reference,
        tests=spec.tests,
        benchmark=spec.benchmark,
        risks=spec.risks,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate task_card.md from task spec."
    )
    parser.add_argument(
        "--task-file", type=Path, help="Path to task.md (default: stdin)"
    )
    parser.add_argument("--output", type=Path, help="Output path (default: stdout)")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate an existing task_card.md at --output",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Exit 1 if any required field is TBD (default: on)",
    )
    parser.add_argument(
        "--no-strict",
        action="store_false",
        dest="strict",
        help="Allow TBD placeholders (not for gates)",
    )
    args = parser.parse_args()

    if args.validate_only:
        if not args.output or not args.output.is_file():
            print(
                "validate-only requires --output pointing to task_card.md",
                file=sys.stderr,
            )
            return 1
        bad = validate_card_text(args.output.read_text(encoding="utf-8"))
        if bad:
            for item in bad:
                print(f"INVALID: {item}", file=sys.stderr)
            return 1
        print(f"OK: {args.output}")
        return 0

    if args.task_file:
        spec = load_task_spec(args.task_file)
    else:
        text = sys.stdin.read()
        spec = load_task_spec_from_markdown(text, source_label="stdin")

    card = build_card_from_spec(spec)
    bad = find_tbd_fields(spec) + validate_card_text(card)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(card, encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(card, end="")

    if args.strict and bad:
        for item in bad:
            print(f"TBD/invalid: {item}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
