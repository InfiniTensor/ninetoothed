"""Tests for Symbol arithmetic: __neg__, __pow__, __truediv__.

Bug found during operator development (pad, flashattention):
ninetoothed 0.25.0 Symbol lacked __neg__, __pow__, __truediv__.
__neg__ was used internally by __add__/__sub__ but missing, so
`Symbol(0) - Symbol(x)` would crash with TypeError.
__pow__ is needed for the 1/sqrt(d) factor in attention: `d ** -0.5`.
__truediv__ is needed for any "/" operation in user code.
"""

from ninetoothed import Symbol


def test_unary_negation():
    s = Symbol(2)
    assert repr(-s) == "-2"


def test_unary_negation_in_arithmetic():
    """__add__ uses -other internally; without __neg__ this crashes."""
    s = Symbol(5)
    # Symbol(0) + s triggers "if other == 0: return self" -- not the path.
    # Try the path: __sub__ uses "-other" when self == 0.
    z = Symbol(0)
    # 0 - s should produce -s
    result = z - s
    assert "-5" in repr(result)


def test_pow_with_int_exponent():
    s = Symbol(2)
    assert "2 ** 3" in repr(s**3)


def test_pow_with_negative_half_exponent():
    """The 1/sqrt(d) factor in attention."""
    d = Symbol(64)
    result = d**-0.5
    # repr should contain 64 and -0.5
    assert "64" in repr(result)
    assert "-0.5" in repr(result)


def test_rpow():
    s = Symbol(2)
    # 3 ** s: Python sees 3 is not a Symbol, calls Symbol.__rpow__(3, s),
    # which delegates to Symbol(3).__pow__(s) = BinOp(3, Pow, s) -> "3 ** 2"
    assert "3 ** 2" in repr(3**s)


def test_truediv():
    s = Symbol(4)
    assert "4 / 2" in repr(s / 2)


def test_rtruediv():
    s = Symbol(2)
    assert "10 / 2" in repr(10 / s)


def test_floordiv_still_works():
    """Regression: __floordiv__ predates our changes; must not break."""
    s = Symbol(7)
    assert "7 // 2" in repr(s // 2)


def test_add_sub_mul_still_work():
    s = Symbol(2)
    assert repr(s + 3) == "2 + 3"
    assert repr(s - 3) == "2 - 3"
    assert repr(s * 3) == "2 * 3"


def test_chained_arithmetic():
    """Realistic expression: scale = d ** -0.5."""
    d = Symbol(64)
    scale = d**-0.5
    # Symbol(1.0) * scale triggers `if self == 1: return other` in __mul__,
    # so the result is just `scale`, not `1.0 * scale`.
    assert "64" in repr(scale)
    assert "-0.5" in repr(scale)
    # Verify scale can be used in a non-trivial expression (no early return).
    e = Symbol(2.0)
    s = e * scale  # 2.0 * (64 ** -0.5) -> BinOp(2.0, Mult, ...)
    assert "2.0" in repr(s)
    assert "64" in repr(s)
