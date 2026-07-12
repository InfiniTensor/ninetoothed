#!/usr/bin/env python3
"""Check project-specific Python style rules from ``CONTRIBUTING.md``."""

from __future__ import annotations

import argparse
import ast
import io
import re
import tokenize
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATHS = ("src", "tests", "docs/source", "scripts")
EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "venv",
}
SENTENCE_PUNCTUATION = ".!?:;"
CONTROL_FLOW_STATEMENTS = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Match,
)
EXCEPTION_NAMES = {
    "AssertionError",
    "AttributeError",
    "EOFError",
    "ImportError",
    "IndexError",
    "KeyError",
    "LookupError",
    "NameError",
    "NotImplementedError",
    "OSError",
    "RuntimeError",
    "StopIteration",
    "SyntaxError",
    "SystemError",
    "TypeError",
    "ValueError",
}
FRAMEWORK_MESSAGE_CALLS = {
    "pytest.importorskip",
    "pytest.skip",
    "pytest.xfail",
}
CODE_QUOTE_PATTERN = re.compile(
    r"(?<!`)['\"]([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?(?:\(\))?)['\"]"
)


@dataclass(frozen=True)
class Diagnostic:
    path: Path
    line: int
    column: int
    code: str
    message: str


@dataclass(frozen=True)
class CommentLine:
    line: int
    column: int
    text: str


@dataclass(frozen=True)
class Fixes:
    insertions: frozenset[int]
    deletions: frozenset[int]


def main() -> int:
    args = parse_args()
    paths = tuple(Path(path) for path in args.paths)
    files = tuple(discover_python_files(paths))
    diagnostics: list[Diagnostic] = []
    fixed_paths: list[Path] = []

    for path in files:
        text = path.read_text(encoding="utf-8")
        path_diagnostics, fixed_text = check_file(path, text, fix=args.fix)
        diagnostics.extend(path_diagnostics)

        if args.fix and fixed_text is not None and fixed_text != text:
            path.write_text(fixed_text, encoding="utf-8")
            fixed_paths.append(path)

    for diagnostic in diagnostics:
        print(
            f"{diagnostic.path}:{diagnostic.line}:{diagnostic.column + 1}: "
            f"{diagnostic.code} {diagnostic.message}"
        )

    if args.fix and fixed_paths:
        for path in fixed_paths:
            print(f"Fixed {path}")

    return 1 if diagnostics else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check project-specific Python style rules from CONTRIBUTING.md."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=DEFAULT_PATHS,
        help="Files or directories to check. Defaults to the Python source tree.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply mechanical blank-line fixes.",
    )

    return parser.parse_args()


def discover_python_files(paths: tuple[Path, ...]) -> list[Path]:
    files: list[Path] = []

    for path in paths:
        if not path.exists():
            continue

        if path.is_file():
            if path.suffix == ".py":
                files.append(path)

            continue

        for child in path.rglob("*.py"):
            if any(part in EXCLUDED_DIRS for part in child.parts):
                continue

            files.append(child)

    return sorted(dict.fromkeys(files))


def check_file(
    path: Path, text: str, *, fix: bool
) -> tuple[list[Diagnostic], str | None]:
    diagnostics: list[Diagnostic] = []
    lines = text.splitlines()

    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as error:
        diagnostics.append(
            Diagnostic(
                path=path,
                line=error.lineno or 1,
                column=error.offset or 0,
                code="PY000",
                message=error.msg,
            )
        )

        return diagnostics, None

    diagnostics.extend(check_comments(path, text))
    diagnostics.extend(check_error_messages(path, tree))
    blank_line_diagnostics, fixes = check_blank_lines(path, tree, lines)
    diagnostics.extend(blank_line_diagnostics)

    if fix:
        fixed_text = apply_fixes(text, fixes)

        if fixed_text != text:
            fixed_diagnostics, _ = check_file(path, fixed_text, fix=False)
            diagnostics = [
                diagnostic
                for diagnostic in fixed_diagnostics
                if diagnostic.code.startswith(("C", "E", "PY"))
            ]

        return diagnostics, fixed_text

    return diagnostics, None


def check_comments(path: Path, text: str) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    comments = collect_comment_lines(text)

    for block in group_comment_blocks(comments):
        if should_skip_comment_block(block):
            continue

        first = block[0]
        normalized = normalize_comment_block(block)

        if not normalized:
            continue

        sentence_error = validate_sentence(normalized)

        if sentence_error is not None:
            diagnostics.append(
                Diagnostic(
                    path=path,
                    line=first.line,
                    column=first.column,
                    code="C001",
                    message=sentence_error,
                )
            )

        if CODE_QUOTE_PATTERN.search(normalized):
            diagnostics.append(
                Diagnostic(
                    path=path,
                    line=first.line,
                    column=first.column,
                    code="C002",
                    message="Use Markdown backticks for code references in comments.",
                )
            )

    return diagnostics


def collect_comment_lines(text: str) -> list[CommentLine]:
    comments: list[CommentLine] = []

    try:
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)

        for token in tokens:
            if token.type != tokenize.COMMENT:
                continue

            comment_text = token.string.lstrip()[1:].strip()
            comments.append(
                CommentLine(
                    line=token.start[0],
                    column=token.start[1],
                    text=comment_text,
                )
            )
    except tokenize.TokenError:
        return comments

    return comments


def group_comment_blocks(comments: list[CommentLine]) -> list[list[CommentLine]]:
    blocks: list[list[CommentLine]] = []

    for comment in comments:
        if not blocks:
            blocks.append([comment])

            continue

        previous = blocks[-1][-1]

        if comment.line == previous.line + 1 and comment.column == previous.column:
            blocks[-1].append(comment)

            continue

        blocks.append([comment])

    return blocks


def should_skip_comment_block(block: list[CommentLine]) -> bool:
    meaningful = [line.text.strip() for line in block if line.text.strip()]

    if not meaningful:
        return True

    return all(is_skippable_comment_line(line) for line in meaningful)


def is_skippable_comment_line(text: str) -> bool:
    lower = text.lower()

    if text.startswith("!") or lower.startswith("-*- coding:"):
        return True

    if re.fullmatch(r"coding[:=]\s*[-\w.]+", lower):
        return True

    if lower.startswith(
        ("noqa", "type:", "pragma:", "fmt:", "isort:", "pylint:", "flake8:")
    ):
        return True

    if text.startswith(("http://", "https://")):
        return True

    return bool(
        re.fullmatch(r"-{2,}.*-{2,}", text) or re.fullmatch(r"[=#*-]{3,}", text)
    )


def normalize_comment_block(block: list[CommentLine]) -> str:
    parts: list[str] = []

    for line in block:
        text = line.text.strip()

        if not text or text.startswith(("http://", "https://")):
            continue

        if is_skippable_comment_line(text):
            continue

        parts.append(text)

    return " ".join(parts).strip()


def check_error_messages(path: Path, tree: ast.AST) -> list[Diagnostic]:
    checker = ErrorMessageChecker(path)
    checker.visit(tree)

    return checker.diagnostics


class ErrorMessageChecker(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.diagnostics: list[Diagnostic] = []

    def visit_Assert(self, node: ast.Assert) -> None:
        if node.msg is not None:
            self._check_message(node.msg)

        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        if isinstance(node.exc, ast.Call) and is_exception_call(node.exc):
            self._check_call_message(node.exc)

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        function_name = get_full_name(node.func)

        if function_name in FRAMEWORK_MESSAGE_CALLS:
            return

        if function_name in {"warnings.warn", "pytest.fail"}:
            self._check_call_message(node)

        self.generic_visit(node)

    def _check_call_message(self, node: ast.Call) -> None:
        if node.args:
            self._check_message(node.args[0])

    def _check_message(self, node: ast.AST) -> None:
        message = get_string_value(node)

        if message is None:
            return

        sentence_error = validate_sentence(message)

        if sentence_error is None:
            return

        self.diagnostics.append(
            Diagnostic(
                path=self.path,
                line=getattr(node, "lineno", 1),
                column=getattr(node, "col_offset", 0),
                code="E001",
                message=sentence_error,
            )
        )


def is_exception_call(node: ast.Call) -> bool:
    function_name = get_full_name(node.func)
    short_name = function_name.rsplit(".", 1)[-1]

    return short_name in EXCEPTION_NAMES or short_name.endswith(("Error", "Exception"))


def get_full_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        prefix = get_full_name(node.value)

        if prefix:
            return f"{prefix}.{node.attr}"

        return node.attr

    return ""


def get_string_value(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value

    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []

        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                parts.append("value")

        return "".join(parts)

    return None


def validate_sentence(text: str) -> str | None:
    normalized = " ".join(text.strip().split())

    if not normalized:
        return None

    first_alpha = next((char for char in normalized if char.isalpha()), "")

    if first_alpha and first_alpha.islower():
        return "Start the sentence with a capital letter."

    if normalized[-1] not in SENTENCE_PUNCTUATION:
        return "End the sentence with punctuation."

    return None


def check_blank_lines(
    path: Path, tree: ast.AST, lines: list[str]
) -> tuple[list[Diagnostic], Fixes]:
    diagnostics: list[Diagnostic] = []
    insertions: set[int] = set()
    deletions: set[int] = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            diagnostics.extend(
                check_function_signature_spacing(path, node, lines, deletions=deletions)
            )

        for statements in child_statement_lists(node):
            diagnostics.extend(
                check_statement_spacing(path, statements, lines, insertions=insertions)
            )

    return diagnostics, Fixes(frozenset(insertions), frozenset(deletions))


def check_function_signature_spacing(
    path: Path,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    lines: list[str],
    *,
    deletions: set[int],
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []

    if not node.body or ast.get_docstring(node, clean=False) is not None:
        return diagnostics

    first_statement = node.body[0]
    header_end_line = find_header_end_line(node, first_statement, lines)

    if header_end_line is None:
        return diagnostics

    gap_line_numbers = range(header_end_line + 1, first_statement.lineno)
    gap_lines = [lines[line_number - 1] for line_number in gap_line_numbers]

    if any(line.strip().startswith("#") for line in gap_lines):
        return diagnostics

    blank_lines = [
        line_number
        for line_number in gap_line_numbers
        if not lines[line_number - 1].strip()
    ]

    if not blank_lines:
        return diagnostics

    deletions.update(blank_lines)
    diagnostics.append(
        Diagnostic(
            path=path,
            line=blank_lines[0],
            column=0,
            code="F001",
            message="Remove the blank line between the function signature and body.",
        )
    )

    return diagnostics


def find_header_end_line(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    first_statement: ast.stmt,
    lines: list[str],
) -> int | None:
    for line_number in range(first_statement.lineno - 1, node.lineno - 1, -1):
        line = lines[line_number - 1].strip()

        if not line or line.startswith("#"):
            continue

        if line.endswith(":"):
            return line_number

    return None


def child_statement_lists(node: ast.AST) -> list[list[ast.stmt]]:
    statement_lists: list[list[ast.stmt]] = []

    for _field, value in ast.iter_fields(node):
        if not isinstance(value, list):
            continue

        if value and all(isinstance(item, ast.stmt) for item in value):
            statement_lists.append(value)

    return statement_lists


def check_statement_spacing(
    path: Path,
    statements: list[ast.stmt],
    lines: list[str],
    *,
    insertions: set[int],
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []

    for index, (previous, current) in enumerate(zip(statements, statements[1:])):
        if previous.end_lineno is None:
            continue

        if index == 0 and is_docstring_statement(previous):
            continue

        needs_blank = False
        code = ""
        message = ""

        if isinstance(current, ast.Return) and not isinstance(
            previous, CONTROL_FLOW_STATEMENTS
        ):
            needs_blank = True
            code = "R001"
            message = "Add a blank line before this return statement."
        elif isinstance(current, CONTROL_FLOW_STATEMENTS):
            needs_blank = True
            code = "B001"
            message = "Add a blank line before this control-flow statement."
        elif isinstance(previous, CONTROL_FLOW_STATEMENTS) and not isinstance(
            current, ast.Return
        ):
            needs_blank = True
            code = "B002"
            message = "Add a blank line after this control-flow statement."

        if not needs_blank or has_blank_line_between(
            lines, previous.end_lineno, current.lineno
        ):
            continue

        insertions.add(current.lineno)
        diagnostics.append(
            Diagnostic(
                path=path,
                line=current.lineno,
                column=current.col_offset,
                code=code,
                message=message,
            )
        )

    return diagnostics


def is_docstring_statement(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def has_blank_line_between(
    lines: list[str], previous_end_line: int, current_line: int
) -> bool:
    return any(
        not lines[line_number - 1].strip()
        for line_number in range(previous_end_line + 1, current_line)
    )


def apply_fixes(text: str, fixes: Fixes) -> str:
    if not fixes.insertions and not fixes.deletions:
        return text

    lines = text.splitlines()
    output: list[str] = []

    for line_number, line in enumerate(lines, start=1):
        if line_number in fixes.insertions:
            output.append("")

        if line_number not in fixes.deletions:
            output.append(line)

    if len(lines) + 1 in fixes.insertions:
        output.append("")

    fixed = "\n".join(output)

    if text.endswith("\n"):
        fixed += "\n"

    return fixed


if __name__ == "__main__":
    raise SystemExit(main())
