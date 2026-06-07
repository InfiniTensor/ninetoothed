"""Unit tests for the key-derivation helpers in ninetoothed._cache.

Covers:
  - hash_function_source: source sensitivity, functools.partial, fallback.
  - hash_tensor_signature: structural identity, NOT instance-bound names.
  - hash_value: stability across calls.
"""

import functools

import pytest

from ninetoothed._cache import (
    hash_function_source,
    hash_tensor_signature,
    hash_value,
)


# ---------- hash_function_source ----------


def test_returns_string():
    def f():
        pass

    h = hash_function_source(f)
    assert isinstance(h, str)
    assert len(h) > 0


def test_returns_src_prefix_for_normal_function():
    def f():
        return 1

    h = hash_function_source(f)
    assert h.startswith("src:")


def test_same_function_same_hash():
    def f(x):
        return x + 1

    assert hash_function_source(f) == hash_function_source(f)


def test_modified_function_different_hash():
    def f(x):
        return x + 1

    h1 = hash_function_source(f)

    def f(x):  # noqa: F811
        return x + 2

    h2 = hash_function_source(f)
    assert h1 != h2


def test_different_functions_different_hash():
    def f():
        return 1

    def g():
        return 2

    assert hash_function_source(f) != hash_function_source(g)


def test_functools_partial_unwrapped():
    """partial(jagged_dim=1) must differ from partial(jagged_dim=2)."""

    def base(x):
        return x

    p1 = functools.partial(base, jagged_dim=1)
    p2 = functools.partial(base, jagged_dim=2)

    assert hash_function_source(p1) != hash_function_source(p2)


def test_functools_partial_nested_keeps_outermost_layer():
    """Known limitation: hash_function_source only unwraps ONE level of partial.
    Nested partials hash DIFFERENTLY because the outer layer is what is recorded.
    Flattening nested partials to canonical (kwargs) form is a separate enhancement.
    """

    def base(x):
        return x

    p_ab = functools.partial(functools.partial(base, a=1), b=2)
    p_ba = functools.partial(functools.partial(base, b=2), a=1)

    # Hashes differ because the outer layer differs.
    assert hash_function_source(p_ab) != hash_function_source(p_ba)
def test_functools_partial_with_args():
    """partial(f, 1, 2) vs partial(f, 3, 4) differ because of bound args."""

    def base(x, y):
        return x + y

    p12 = functools.partial(base, 1, 2)
    p34 = functools.partial(base, 3, 4)

    assert hash_function_source(p12) != hash_function_source(p34)


def test_fallback_id_prefix_when_source_unavailable():
    """Lambdas defined in REPL/test scope may fail getsource; should still hash."""
    # Create a function whose source cannot be retrieved: a built-in
    h = hash_function_source(len)
    assert h.startswith("id:")
    # Same callable -> same hash
    assert hash_function_source(len) == h


def test_id_fallback_stable_across_calls():
    h1 = hash_function_source(len)
    h2 = hash_function_source(len)
    assert h1 == h2


# ---------- hash_tensor_signature ----------


class FakeTensor:
    """Mimics the ninetoothed.Tensor surface used by hash_tensor_signature."""

    def __init__(self, ndim, jagged_dim=None, other=0, name="t"):
        self.ndim = ndim
        self.jagged_dim = jagged_dim
        self.other = other
        self.name = name


def test_returns_tuple():
    t = FakeTensor(ndim=2)
    sig = hash_tensor_signature(t)
    assert isinstance(sig, tuple)


def test_same_structure_same_signature():
    """Two tensors with the same ndim/jagged_dim/other must hash equal
    even if they were constructed with different instance-bound names."""
    t1 = FakeTensor(ndim=2, jagged_dim=None, other=0, name="x_0")
    t2 = FakeTensor(ndim=2, jagged_dim=None, other=0, name="x_42")
    assert hash_tensor_signature(t1) == hash_tensor_signature(t2)


def test_different_ndim_different_signature():
    t1 = FakeTensor(ndim=2)
    t2 = FakeTensor(ndim=3)
    assert hash_tensor_signature(t1) != hash_tensor_signature(t2)


def test_different_jagged_dim_different_signature():
    t1 = FakeTensor(ndim=2, jagged_dim=0)
    t2 = FakeTensor(ndim=2, jagged_dim=1)
    assert hash_tensor_signature(t1) != hash_tensor_signature(t2)


def test_different_other_different_signature():
    t1 = FakeTensor(ndim=2, other=0)
    t2 = FakeTensor(ndim=2, other=1)
    assert hash_tensor_signature(t1) != hash_tensor_signature(t2)


def test_name_attribute_does_not_affect_signature():
    """Critical: Tensor.name is instance-counter-bound and must NOT influence
    the cache key, otherwise every fresh Tensor (even structurally identical)
    would miss the cache."""
    t_a = FakeTensor(ndim=2, name="input_0")
    t_b = FakeTensor(ndim=2, name="input_99999")
    assert hash_tensor_signature(t_a) == hash_tensor_signature(t_b)


# ---------- hash_value ----------


def test_hash_value_stable():
    assert hash_value(42) == hash_value(42)
    assert hash_value("abc") == hash_value("abc")
    assert hash_value([1, 2, 3]) == hash_value([1, 2, 3])


def test_hash_value_different_inputs_different_hash():
    assert hash_value(42) != hash_value(43)
    assert hash_value("abc") != hash_value("abd")


def test_hash_value_returns_hex_string():
    h = hash_value(42)
    assert isinstance(h, str)
    # SHA256 hex digest is 64 chars
    assert len(h) == 64
    int(h, 16)  # must be valid hex


def test_hash_value_handles_arbitrary_python_objects():
    """repr-based, so any object with a sensible repr works."""

    class Obj:
        def __repr__(self):
            return "Obj()"

    a = Obj()
    b = Obj()
    assert hash_value(a) == hash_value(b)
