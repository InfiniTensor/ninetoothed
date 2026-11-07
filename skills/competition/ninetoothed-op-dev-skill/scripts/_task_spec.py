"""Load operator task specs from task.md or structured task.yaml (generic)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore

TODO_VERIFY = "TODO: verify"


@dataclass
class TaskSpec:
    task_id: str
    operator: str
    math: str
    inputs_rows: str
    outputs_rows: str
    broadcast: str
    layout: str
    boundaries: str
    reference: str
    tests: str
    benchmark: str
    benchmark_required: bool
    risks: str
    source_label: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "math": self.math,
            "inputs_rows": self.inputs_rows,
            "outputs_rows": self.outputs_rows,
            "broadcast": self.broadcast,
            "layout": self.layout,
            "boundaries": self.boundaries,
            "reference": self.reference,
            "tests": self.tests,
            "benchmark": self.benchmark,
            "benchmark_required": self.benchmark_required,
            "risks": self.risks,
        }


_SECTION_HEADING = re.compile(
    r"^##\s+(?:(\d+)[.)]\s*)?(.+?)\s*$",
    re.MULTILINE,
)
_SEMANTICS = re.compile(r"semantics\s*=\s*`([^`]+)`", re.IGNORECASE)
_OUTPUT_EQ = re.compile(r"output\s*=\s*`([^`]+)`", re.IGNORECASE)
_IO_TENSOR = re.compile(r"`?(\w+)`?\s*\(([^)]+)\)")
_IMPLEMENT = re.compile(r"implement \*\*([^*]+)\*\*", re.IGNORECASE)
_INCOMPLETE = re.compile(
    r"\bTBD\b|TODO:\s*verify|see contract|see task\.md|pytest per task\.md",
    re.IGNORECASE,
)
_KNOWN_OPS = frozenset(
    {
        "mul",
        "add",
        "sub",
        "div",
        "where",
        "relu",
        "softmax",
        "gelu",
        "silu",
        "amin",
        "amax",
        "cast",
        "float",
        "empty",
        "zeros",
        "ones",
        "leaky_relu",
        "input",
        "output",
        "out",
    }
)


def _section_map(text: str) -> dict[str, str]:
    matches = list(_SECTION_HEADING.finditer(text))
    sections: dict[str, str] = {}
    for i, match in enumerate(matches):
        title = match.group(2).strip().lower()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections[title] = body
    return sections


def _first_line(block: str) -> str:
    for line in block.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return block.strip()


def _benchmark_not_required(block: str) -> bool:
    """Return True when the task explicitly says benchmark is not required / N/A."""
    low = block.lower().strip()
    if not low:
        return False
    if re.search(r"\bnot required\b", low):
        return True
    if re.search(r"\bn/?a\b", low) or low in {"na", "na.", "n/a", "n/a."}:
        return True
    return False


def _benchmark_required(block: str) -> bool:
    low = block.lower()
    if _benchmark_not_required(block):
        return False
    if "required" in low:
        return True
    return False


def _explicit_dtype(text: str) -> str | None:
    low = text.lower()
    if "float16" in low or "fp16" in low:
        return "float16"
    if "bfloat16" in low or "bf16" in low:
        return "bfloat16"
    if "float64" in low or "fp64" in low:
        return "float64"
    if "float32" in low or "fp32" in low:
        return "float32"
    if re.search(r"\bbool\b", low):
        return "bool"
    if "int64" in low or "int32" in low or "int16" in low or "int8" in low:
        for tok in ("int64", "int32", "int16", "int8"):
            if tok in low:
                return tok
    return None


def _explicit_layout(text: str) -> str | None:
    low = text.lower()
    if "non-contiguous" in low or "noncontiguous" in low or "strided" in low:
        return "non-contiguous"
    if "contiguous" in low:
        return "contiguous"
    return None


def _local_clause(text: str, match_start: int, match_end: int) -> str:
    """Return the local statement/table-row/clause around a tensor match.

    Never use the whole contract: dtype/layout must come from this local span.
    Prefer span limited by separators (; . newline bullet table).
    """
    line_start = text.rfind("\n", 0, match_start) + 1
    line_end = text.find("\n", match_end)
    if line_end < 0:
        line_end = len(text)
    line = text[line_start:line_end]
    relative_start = match_start - line_start
    # Split on ';' or '.' (sentence end before next tensor / Output).
    parts = re.split(r"\s*[;.]\s*", line)
    if len(parts) > 1:
        pos = 0
        for part in parts:
            idx = line.find(part, pos)
            if idx < 0:
                continue
            if idx <= relative_start < idx + len(part):
                return part.strip()
            pos = idx + len(part)
    if "|" in line:
        return line.strip()
    comma_parts = re.split(r"\s*,\s*(?=`?\w+`?\s*\()", line)
    if len(comma_parts) > 1:
        pos = 0
        for part in comma_parts:
            idx = line.find(part, pos)
            if idx < 0:
                continue
            if idx <= relative_start < idx + len(part):
                return part.strip()
            pos = idx + len(part)
    return line.strip()


def _attrs_from_local(name: str, local: str) -> tuple[str, str]:
    """Extract dtype/layout from a tensor-local span only."""
    dtype = _explicit_dtype(local) or TODO_VERIFY
    layout = _explicit_layout(local) or TODO_VERIFY
    # mask: explicit bool in local span must win over later float leakage.
    if name.lower() == "mask" and re.search(r"\bbool\b", local, re.IGNORECASE):
        dtype = "bool"
    return dtype, layout


def _parse_md_io_tables(text: str) -> tuple[list[str], list[str]]:
    """Parse markdown I/O tables; dtype/layout come from each row only."""
    inputs: list[str] = []
    outputs: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if "|" not in line:
            i += 1
            continue
        headers = [c.strip().lower() for c in line.strip().strip("|").split("|")]
        if not any("tensor" in h or h == "name" for h in headers):
            i += 1
            continue
        if not any("shape" in h for h in headers):
            i += 1
            continue
        i += 1
        if i < len(lines) and re.match(r"^\|?\s*-+", lines[i]):
            i += 1
        has_layout = any("layout" in h for h in headers)
        direction_headers = {
            h
            for h in headers
            if h in {"role", "direction", "io", "i/o", "kind"} or "direction" in h
        }
        while (
            i < len(lines) and "|" in lines[i] and not re.match(r"^\|?\s*-+", lines[i])
        ):
            cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            row = {
                headers[j]: (cells[j] if j < len(cells) else "")
                for j in range(len(headers))
            }
            name = ""
            for key in ("tensor", "name"):
                for h, v in row.items():
                    if key in h and v:
                        name = v.strip("`")
                        break
                if name:
                    break
            if not name or name.lower() in {"tensor", "name"}:
                i += 1
                continue
            shape = TODO_VERIFY
            for h, v in row.items():
                if "shape" in h and v:
                    shape = v
                    break
            dtype_cell = ""
            for h, v in row.items():
                if "dtype" in h:
                    dtype_cell = v
                    break
            dtype = _explicit_dtype(dtype_cell) or (
                dtype_cell if dtype_cell.strip() else TODO_VERIFY
            )
            if name.lower() == "mask" and re.search(
                r"\bbool\b", dtype_cell, re.IGNORECASE
            ):
                dtype = "bool"
            direction = ""
            for header in direction_headers:
                value = row.get(header, "").strip().lower()
                if re.search(r"\b(output|result|return|returns)\b", value):
                    direction = "output"
                    break
                if re.search(r"\b(input|operand|argument|arg)\b", value):
                    direction = "input"
                    break
            if direction == "output":
                outputs.append(f"| {name} | {shape} | {dtype} |")
            elif direction == "input" or has_layout:
                layout_cell = ""
                for h, v in row.items():
                    if "layout" in h:
                        layout_cell = v
                        break
                layout = _explicit_layout(layout_cell) or TODO_VERIFY
                inputs.append(f"| {name} | {shape} | {dtype} | {layout} |")
            else:
                outputs.append(f"| {name} | {shape} | {dtype} |")
            i += 1
        continue
    return inputs, outputs


def _parse_bullet_io(text: str) -> tuple[list[str], list[str]]:
    """Parse bullet I/O lines; dtype/layout from that bullet only."""
    inputs: list[str] = []
    outputs: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("-"):
            continue
        body = stripped[1:].strip()
        low = body.lower()
        is_output = (
            low.startswith("output")
            or low.startswith("result")
            or low.startswith("return")
            or low.startswith("out ")
            or re.match(r"out\b", low) is not None
        )
        # Prefer explicit output/out label; bare `out` name handled below.
        m_shape = re.search(r"\(([^)]+)\)", body)
        shape = (
            "(" + m_shape.group(1).replace(" ", "") + ")" if m_shape else TODO_VERIFY
        )
        names = re.findall(r"`(\w+)`", body)
        if not names:
            m_name = re.match(r"(?:input|output)?\s*`?([A-Za-z_]\w*)`?\s*\(", body)
            if m_name and m_name.group(1).lower() not in _KNOWN_OPS - {
                "input",
                "output",
                "out",
            }:
                names = [m_name.group(1)]
        if is_output or (names and names[0].lower() in {"output", "out"}):
            name = names[0] if names else "output"
            dtype, _layout = _attrs_from_local(name, body)
            outputs.append(f"| {name} | {shape} | {dtype} |")
        elif "scalar" in low and names:
            dtype, layout = _attrs_from_local(names[0], body)
            inputs.append(f"| {names[0]} | scalar | {dtype} | {layout} |")
        else:
            for name in names or []:
                if name.lower() in _KNOWN_OPS - {"input"}:
                    continue
                dtype, layout = _attrs_from_local(name, body)
                inputs.append(f"| {name} | {shape} | {dtype} | {layout} |")
    return inputs, outputs


def _parse_inline_tensors(text: str) -> tuple[list[str], list[str]]:
    """Parse `name (shape)` mentions; attrs from [match, next_match) only."""
    inputs: list[str] = []
    outputs: list[str] = []
    # Strip math/semantics so operator calls are less likely to leak.
    scrubbed = re.sub(r"semantics\s*=\s*`[^`]+`", "", text, flags=re.IGNORECASE)
    scrubbed = re.sub(r"torch\.\w+\([^)]*\)", "", scrubbed, flags=re.IGNORECASE)

    matches: list[re.Match[str]] = []
    for match in _IO_TENSOR.finditer(scrubbed):
        name, shape = match.group(1), match.group(2).replace(" ", "")
        if "e.g" in shape.lower() or name.lower() in {"configs", "config", "e"}:
            continue
        if name.lower() in _KNOWN_OPS - {"input", "output", "out"}:
            continue
        matches.append(match)

    for i, match in enumerate(matches):
        name = match.group(1)
        shape = match.group(2).replace(" ", "")
        prefix_start = matches[i - 1].end() if i > 0 else 0
        prefix = scrubbed[prefix_start : match.start()]
        has_output_label = bool(
            re.search(
                r"(?:^|[.;]\s*)(?:outputs?|results?|returns?)\s*:?[\s`]*$",
                prefix,
                re.IGNORECASE,
            )
        )
        # Attribute span: current match → next tensor match (exclusive).
        span_end = matches[i + 1].start() if i + 1 < len(matches) else len(scrubbed)
        local = scrubbed[match.start() : span_end]
        # Also trim at sentence / bullet boundaries inside the span.
        local = re.split(r"(?<=[.])\s+(?=[A-Z`])", local, maxsplit=1)[0]
        dtype, layout = _attrs_from_local(name, local)
        if name.lower() in {"output", "out"} or has_output_label:
            outputs.append(f"| {name} | ({shape}) | {dtype} |")
        else:
            inputs.append(f"| {name} | ({shape}) | {dtype} | {layout} |")
    return inputs, outputs


def _parse_io_table(
    contract: str, background: str, operator: str
) -> tuple[str, str, str]:
    del operator  # operator name must not drive invented I/O fallbacks
    math = ""
    sem = _SEMANTICS.search(contract)
    out_eq = _OUTPUT_EQ.search(contract)
    if sem:
        math = sem.group(1).strip()
    elif out_eq:
        math = f"output = {out_eq.group(1).strip()}"
    elif "`where(" in contract:
        math = contract.strip()
    elif "out=a+b" in contract.replace(" ", "").lower():
        math = "out = a + b (elementwise)"
    elif "=" in contract:
        math = contract.split(";", 1)[0].strip()

    inputs: list[str] = []
    outputs: list[str] = []

    table_in, table_out = _parse_md_io_tables(contract)
    if table_in or table_out:
        inputs = table_in
        outputs = table_out
    else:
        bullet_in, bullet_out = _parse_bullet_io(contract)
        if bullet_in or bullet_out:
            inputs = bullet_in
            outputs = bullet_out
        else:
            inputs, outputs = _parse_inline_tensors(contract)

    if not outputs:
        m = re.search(
            r"(?:output|result|returns?)\s*\(([^)]+)\)",
            contract,
            re.IGNORECASE,
        )
        if m:
            local = _local_clause(contract, m.start(), m.end())
            dtype = _explicit_dtype(local) or TODO_VERIFY
            outputs.append(f"| output | {m.group(1).replace(' ', '')} | {dtype} |")
    if not inputs:
        m = re.search(r"input\s*\(([^)]+)\)", contract, re.IGNORECASE)
        if m:
            local = _local_clause(contract, m.start(), m.end())
            dtype = _explicit_dtype(local) or TODO_VERIFY
            layout = _explicit_layout(local) or TODO_VERIFY
            inputs.append(
                f"| input | {m.group(1).replace(' ', '')} | {dtype} | {layout} |"
            )

    inputs_rows = (
        "\n".join(inputs)
        if inputs
        else f"| input | {TODO_VERIFY} | {TODO_VERIFY} | {TODO_VERIFY} |"
    )
    outputs_rows = (
        "\n".join(outputs) if outputs else f"| output | {TODO_VERIFY} | {TODO_VERIFY} |"
    )
    if not math:
        impl = _IMPLEMENT.search(background)
        math = impl.group(1).strip() if impl else (contract.strip() or TODO_VERIFY)
    return inputs_rows, outputs_rows, math


def _operator_name(background: str, contract: str, *, task_id: str = "") -> str:
    del task_id  # never infer operator semantics from task_id prefixes
    impl = _IMPLEMENT.search(background)
    if impl:
        return impl.group(1).strip()
    line = _first_line(contract)
    return line if line else TODO_VERIFY


def _resolve_benchmark(benchmark_block: str, deliverables: str) -> tuple[str, bool]:
    """Return (benchmark_text, benchmark_required)."""
    if _benchmark_not_required(benchmark_block) or _benchmark_not_required(
        deliverables
    ):
        return "Not required.", False
    if not benchmark_block.strip():
        return TODO_VERIFY, False
    required = _benchmark_required(benchmark_block) or _benchmark_required(deliverables)
    return _first_line(benchmark_block), required


def load_task_spec_from_markdown(
    text: str, *, source_label: str, task_id: str = ""
) -> TaskSpec:
    sections = _section_map(text)
    math_block = sections.get("math definition", "")
    io_block = sections.get("inputs and outputs", "")
    constraints_block = sections.get("constraints", "")
    deliverables = sections.get("deliverables", "")

    background = sections.get("background", math_block)
    contract = sections.get("operator contract", "")
    if not contract and io_block:
        contract = f"{io_block}\n{math_block}".strip()
    layout_block = sections.get(
        "layout / broadcast / dtype",
        sections.get("layout", constraints_block),
    )
    correctness = deliverables or sections.get("correctness requirements", "")
    # Prefer dedicated correctness section when present.
    if sections.get("correctness requirements"):
        correctness = sections["correctness requirements"]
    benchmark_block = sections.get("benchmark requirements", "")
    if not benchmark_block:
        for ln in deliverables.splitlines():
            if "benchmark" in ln.lower():
                benchmark_block = ln.strip()
                break

    operator = _operator_name(background, contract or math_block, task_id=task_id)
    io_source = io_block.strip() if io_block.strip() else contract
    inputs_rows, outputs_rows, parsed_math = _parse_io_table(
        io_source, background, operator
    )
    math = _first_line(math_block) if math_block else parsed_math
    if not math or math == operator:
        math = parsed_math or _first_line(math_block) or TODO_VERIFY
    reference = ""
    if math and math != TODO_VERIFY:
        reference = math
    elif "torch." in (contract or math_block or io_block):
        reference = _first_line(math_block or contract or io_block)

    tests_lines = [
        ln.strip()
        for ln in correctness.splitlines()
        if ln.strip() and not ln.strip().startswith("```")
    ]
    tests = "\n".join(tests_lines[:6]) if tests_lines else TODO_VERIFY

    boundaries = (
        sections.get("boundary cases")
        or sections.get("boundaries")
        or sections.get("boundary conditions")
        or sections.get("constraints")
        or sections.get("likely failure modes")
        or TODO_VERIFY
    )
    if not str(boundaries).strip():
        boundaries = TODO_VERIFY
    layout = layout_block.strip() if layout_block.strip() else TODO_VERIFY

    broadcast_src = ""
    if any("broadcast" in title for title in sections):
        broadcast_src = _first_line(layout_block or constraints_block or "")
    if not broadcast_src:
        for block in (layout_block, constraints_block, contract, io_block):
            if block and "broadcast" in block.lower():
                broadcast_src = _first_line(block)
                break
    broadcast = broadcast_src if broadcast_src else TODO_VERIFY

    benchmark, benchmark_required = _resolve_benchmark(benchmark_block, deliverables)

    return TaskSpec(
        task_id=task_id,
        operator=operator,
        math=math or operator,
        inputs_rows=inputs_rows,
        outputs_rows=outputs_rows,
        broadcast=broadcast,
        layout=layout,
        boundaries=boundaries,
        reference=reference or TODO_VERIFY,
        tests=tests,
        benchmark=benchmark,
        benchmark_required=benchmark_required,
        risks=boundaries if boundaries != TODO_VERIFY else TODO_VERIFY,
        source_label=source_label,
    )


def load_task_spec_from_yaml(data: dict[str, Any], *, source_label: str) -> TaskSpec:
    def _rows(items: list[dict[str, Any]], *, with_layout: bool) -> str:
        lines = []
        for item in items:
            shape = str(item.get("shape") or TODO_VERIFY)
            dtype = str(item.get("dtype") or TODO_VERIFY)
            if with_layout:
                layout = str(item.get("layout") or TODO_VERIFY)
                lines.append(f"| {item['name']} | {shape} | {dtype} | {layout} |")
            else:
                lines.append(f"| {item['name']} | {shape} | {dtype} |")
        return "\n".join(lines)

    inputs = data.get("inputs", [])
    outputs = data.get("outputs", [])
    bench_raw = data.get("benchmark")
    if bench_raw is None:
        benchmark, benchmark_required = TODO_VERIFY, False
    else:
        benchmark, benchmark_required = _resolve_benchmark(
            str(bench_raw), str(data.get("deliverables", ""))
        )
        if "benchmark_required" in data:
            benchmark_required = bool(data["benchmark_required"])
    return TaskSpec(
        task_id=str(data.get("task_id", "")),
        operator=str(data.get("operator") or TODO_VERIFY),
        math=str(data.get("math") or data.get("operator") or TODO_VERIFY),
        inputs_rows=_rows(inputs, with_layout=True)
        if inputs
        else f"| input | {TODO_VERIFY} | {TODO_VERIFY} | {TODO_VERIFY} |",
        outputs_rows=_rows(outputs, with_layout=False)
        if outputs
        else f"| output | {TODO_VERIFY} | {TODO_VERIFY} |",
        broadcast=str(data.get("broadcast") or TODO_VERIFY),
        layout=str(data.get("layout") or TODO_VERIFY),
        boundaries=str(data.get("boundaries") or TODO_VERIFY),
        reference=str(data.get("reference") or TODO_VERIFY),
        tests=str(data.get("tests") or TODO_VERIFY),
        benchmark=benchmark,
        benchmark_required=benchmark_required,
        risks=str(data.get("risks") or TODO_VERIFY),
        source_label=source_label,
    )


def load_task_spec(path: Path) -> TaskSpec:
    yaml_path = path.with_suffix(".yaml")
    if yaml_path.is_file():
        if yaml is None:
            raise RuntimeError("PyYAML required to read task.yaml")
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"invalid YAML root in {yaml_path}")
        return load_task_spec_from_yaml(data, source_label=str(yaml_path))

    text = path.read_text(encoding="utf-8")
    task_id = path.parent.name if path.name == "task.md" else path.stem
    return load_task_spec_from_markdown(text, source_label=str(path), task_id=task_id)


REQUIRED_CARD_FIELDS = (
    "operator",
    "math",
    "inputs_rows",
    "outputs_rows",
    "broadcast",
    "layout",
    "boundaries",
    "reference",
    "tests",
    "benchmark",
)


def validate_card_text(card: str) -> list[str]:
    bad: list[str] = []
    for line in card.splitlines():
        if _INCOMPLETE.search(line):
            bad.append(f"incomplete field in line: {line.strip()}")
    _bogus_inputs = re.compile(
        r"^\|\s*(mul|add|sub|div|where|relu|softmax|gelu|silu)\s*\|",
        re.IGNORECASE,
    )
    in_inputs = False
    for line in card.splitlines():
        if line.strip().startswith("## Inputs"):
            in_inputs = True
            continue
        if in_inputs and line.strip().startswith("## "):
            in_inputs = False
        if in_inputs and _bogus_inputs.match(line.strip()):
            bad.append(f"bogus input tensor parsed from math expr: {line.strip()}")
    return bad


def find_tbd_fields(spec: TaskSpec) -> list[str]:
    bad: list[str] = []
    data = spec.as_dict()
    for key in REQUIRED_CARD_FIELDS:
        value = str(data[key])
        if not value.strip():
            bad.append(key)
        elif _INCOMPLETE.search(value):
            bad.append(key)
    return bad


# Capability-family probes keyed by operator text in the *spec*, never by task_id.
OPERATOR_PROBES: dict[str, dict[str, list[str]]] = {
    "gelu": {
        "required_any": ["gelu", "libdevice.erf", "F.gelu"],
        "forbidden": ["def silu", "F.silu", "ntl.sigmoid(ntl.cast"],
    },
    "broadcast_add": {
        "required_any": ["broadcast_add", ".expand("],
        "forbidden": [".expand(m, n).contiguous()", "expand(m, n).contiguous()"],
    },
}


def _probes_from_spec(spec: TaskSpec | None) -> dict[str, list[str]]:
    if spec is None:
        return {}
    blob = f"{spec.operator}\n{spec.math}\n{spec.reference}".lower()
    if "gelu" in blob:
        return OPERATOR_PROBES["gelu"]
    if "broadcast" in blob and ("add" in blob or "+" in blob):
        return OPERATOR_PROBES["broadcast_add"]
    return {}


def operator_contract_violations(
    task_id: str, test_source: str, spec: TaskSpec | None = None
) -> list[str]:
    """Check test source against explicit operator/spec text only (ignore task_id)."""
    del task_id
    low = test_source.lower()
    violations: list[str] = []
    probes = _probes_from_spec(spec)
    for token in probes.get("forbidden", []):
        if token.lower() in low:
            violations.append(f"forbidden token in test source: {token}")
    required = probes.get("required_any", [])
    if required and not any(tok.lower() in low for tok in required):
        violations.append(f"missing required operator markers: {required}")

    if spec:
        op_blob = f"{spec.operator} {spec.math}".lower()
        if "gelu" in op_blob:
            if "silu" in low and "gelu" not in low:
                violations.append("spec requires GELU but test implements SiLU")
        if "broadcast" in op_blob:
            if "expand(m, n).contiguous()" in low or "expand(m,n).contiguous()" in low:
                violations.append(
                    "broadcast materialized via expand().contiguous() in wrapper"
                )
    return violations
