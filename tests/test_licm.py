import ast
import contextlib
import io

import pytest

from ninetoothed.licm import LICMTransformer


@pytest.mark.parametrize(
    "source, expected",
    (
        (
            """for i in range(10):
    x = 5 + 3
    y = i + x
    print(y)""",
            """x = 5 + 3
for i in range(10):
    y = i + x
    print(y)""",
        ),
        (
            """for i in range(5):
    for j in range(5):
        const = 10 * 20
        result = i + j + const
        print(result)""",
            """const = 10 * 20
for i in range(5):
    for j in range(5):
        result = i + j + const
        print(result)""",
        ),
        (
            """total = 0
for i in range(100):
    base = 1000
    offset = i * 2
    value = base + offset
    total += value
    print(total)""",
            """total = 0
base = 1000
for i in range(100):
    offset = i * 2
    value = base + offset
    total += value
    print(total)""",
        ),
    ),
)
def test_licm(source, expected):
    transformed = _apply_licm(source)

    with contextlib.redirect_stdout(io.StringIO()) as output:
        exec(source)

    output_with_source = output.getvalue()

    with contextlib.redirect_stdout(io.StringIO()) as output:
        exec(transformed)

    output_with_transformed = output.getvalue()

    assert transformed == expected and output_with_source == output_with_transformed


def _apply_licm(source_code):
    tree = ast.parse(source_code)
    tree = LICMTransformer().visit(tree)
    tree = ast.fix_missing_locations(tree)

    return ast.unparse(tree)
