import ast
import contextlib
import io

import pytest

from ninetoothed.cse import CSETransformer


@pytest.mark.parametrize(
    "source, expected",
    (
        (
            """a = 1
b = 2
result = (a + b) * (a + b) + (a + b)""",
            """a = 1
b = 2
ninetoothed_cse_0_ = a + b
ninetoothed_cse_1_ = ninetoothed_cse_0_ * ninetoothed_cse_0_
ninetoothed_cse_2_ = ninetoothed_cse_1_ + ninetoothed_cse_0_
result = ninetoothed_cse_2_""",
        ),
        (
            """for i in range(10):
    x = i * 2 + 5
    y = i * 2 + 5
    z = (i * 2 + 5) * 3
    print(x, y, z)""",
            """for i in range(10):
    ninetoothed_cse_0_ = i * 2
    ninetoothed_cse_1_ = ninetoothed_cse_0_ + 5
    x = ninetoothed_cse_1_
    y = ninetoothed_cse_1_
    ninetoothed_cse_2_ = ninetoothed_cse_1_ * 3
    z = ninetoothed_cse_2_
    print(x, y, z)""",
        ),
        (
            """def calculate(a, b):
    x = (a * b) + (a * b)
    y = (a * b) * 2
    return x + y
print(calculate(1, 2))""",
            """def calculate(a, b):
    ninetoothed_cse_0_ = a * b
    ninetoothed_cse_1_ = ninetoothed_cse_0_ + ninetoothed_cse_0_
    x = ninetoothed_cse_1_
    ninetoothed_cse_2_ = ninetoothed_cse_0_ * 2
    y = ninetoothed_cse_2_
    ninetoothed_cse_3_ = x + y
    return ninetoothed_cse_3_
print(calculate(1, 2))""",
        ),
        (
            """x = 1
y = 2
condition = False
if condition:
    result = (x + y) * (x + y)
else:
    result = (x + y) * 2
print(result)""",
            """x = 1
y = 2
condition = False
if condition:
    ninetoothed_cse_0_ = x + y
    ninetoothed_cse_1_ = ninetoothed_cse_0_ * ninetoothed_cse_0_
    result = ninetoothed_cse_1_
else:
    ninetoothed_cse_2_ = x + y
    ninetoothed_cse_3_ = ninetoothed_cse_2_ * 2
    result = ninetoothed_cse_3_
print(result)""",
        ),
        (
            """a = 5
b = 10
c = (a + b) * (a + b)
d = (a + b) + 20
e = (a + b) * 2
print(a, b, c, d, e)""",
            """a = 5
b = 10
ninetoothed_cse_0_ = a + b
ninetoothed_cse_1_ = ninetoothed_cse_0_ * ninetoothed_cse_0_
c = ninetoothed_cse_1_
ninetoothed_cse_2_ = ninetoothed_cse_0_ + 20
d = ninetoothed_cse_2_
ninetoothed_cse_3_ = ninetoothed_cse_0_ * 2
e = ninetoothed_cse_3_
print(a, b, c, d, e)""",
        ),
        (
            """print((lambda x, y: x + y)(1, 2))""",
            """print((lambda x, y: x + y)(1, 2))""",
        ),
        (
            """a = 1
b = 2
c = a + b
for i in range(10):
    d = a + b
e = a + b
print(a, b, c, d, e)""",
            """a = 1
b = 2
ninetoothed_cse_0_ = a + b
c = ninetoothed_cse_0_
for i in range(10):
    ninetoothed_cse_1_ = a + b
    d = ninetoothed_cse_1_
ninetoothed_cse_2_ = a + b
e = ninetoothed_cse_2_
print(a, b, c, d, e)""",
        ),
    ),
)
def test_cse(source, expected):
    transformed = _apply_cse(source)

    with contextlib.redirect_stdout(io.StringIO()) as output:
        exec(source)

    output_with_source = output.getvalue()

    with contextlib.redirect_stdout(io.StringIO()) as output:
        exec(transformed)

    output_with_transformed = output.getvalue()

    assert transformed == expected and output_with_source == output_with_transformed


def _apply_cse(source_code):
    tree = ast.parse(source_code)
    tree = CSETransformer().visit(tree)
    tree = ast.fix_missing_locations(tree)

    return ast.unparse(tree)
