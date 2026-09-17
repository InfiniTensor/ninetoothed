"""Geometry reuse keeps numeric reads, warnings and dynamic bindings live."""

import gc
import random
import weakref
from dataclasses import replace

import numpy as np
import pytest

import ninetoothed.language as ntl
from ninetoothed import Tensor, interpret
from ninetoothed.interpreter import expressions, interpret_program
from ninetoothed.interpreter.geometry import ExtractionGeometry, _bounds
from ninetoothed.interpreter.memory import TensorRef
from ninetoothed.ir import AccessMap, IndexExpr, LayoutLevel, TensorLayout, TensorSpec


def make_ref(
    array, *, index=None, predicate=None, symbols=None, shape=None, cache=None
):
    index = IndexExpr.parse("offset + value_0") if index is None else index
    access = AccessMap(
        source_indices=(index,),
        linear_index=index,
        predicate=IndexExpr.parse(True) if predicate is None else predicate,
    )
    layout = TensorLayout(
        source_shape=(IndexExpr.parse("n"),),
        source_strides=(IndexExpr.parse(1),),
        view_shape=(IndexExpr.parse(1),),
        application_shape=(IndexExpr.parse(1),),
        levels=(
            LayoutLevel(shape=(IndexExpr.parse("tile") if shape is None else shape,)),
        ),
        value_accesses=(access,),
    )

    return TensorRef(
        array,
        TensorSpec(name="x", ndim=1, layout=layout),
        {"n": len(array), "tile": 4, "offset": 0} if symbols is None else symbols,
        _geometry_cache=ExtractionGeometry() if cache is None else cache,
    )


def count_accesses(monkeypatch):
    calls = []
    original = TensorRef._access

    def counted(ref, extra_mask=True):
        calls.append(ref)

        return original(ref, extra_mask)

    monkeypatch.setattr(TensorRef, "_access", counted)

    return calls


def test_reads_fresh_data_and_rebuilds_changed_bindings(monkeypatch):
    calls = count_accesses(monkeypatch)
    x = np.arange(8, dtype=np.float32)
    ref = make_ref(x)
    assert ref.extract((1,)) == 1
    x[1] = 42
    assert ref.extract((1,)) == 42
    assert len(calls) == 1
    ref.symbols["offset"] = 2
    assert ref.extract((1,)) == 3
    assert len(calls) == 2
    # Equal integer/boolean values are not interchangeable proof inputs.
    ref.symbols["offset"] = True
    assert ref.extract((1,)) == 2
    assert ref.extract((1,)) == 2
    assert len(calls) == 4


def test_array_shape_change_is_revalidated():
    x = np.arange(8, dtype=np.float32)
    ref = make_ref(x)
    assert ref.extract((1,)) == 1
    x.shape = (2, 4)

    with pytest.raises(ValueError, match="Layout/source rank mismatch"):
        ref.extract((1,))


def test_cached_geometry_does_not_escape_reads_or_retain_user_array():
    x = np.arange(8, dtype=np.float32)
    cache = ExtractionGeometry()
    ref = make_ref(x, cache=cache)
    first = ref.extract((slice(None),))
    first[:] = -11
    np.testing.assert_array_equal(ref.extract((slice(None),)), np.arange(4))
    reference = weakref.ref(x)
    del ref, x
    gc.collect()
    assert reference() is None


def test_divide_by_zero_warning_policy_is_not_cached():
    ref = make_ref(
        np.arange(8, dtype=np.float32),
        index=IndexExpr.parse("value_0 // divisor"),
        symbols={"n": 8, "tile": 4, "divisor": 0},
    )

    with np.errstate(divide="ignore"):
        assert ref.extract((0,)) == 0

    with np.errstate(divide="raise"):
        with pytest.raises(FloatingPointError):
            ref.extract((0,))


def test_scalar_integer_overflow_stays_live_even_with_false_mask():
    ref = make_ref(
        np.arange(8, dtype=np.float32),
        index=IndexExpr.parse(f"(({2**63 - 1} // 1) * 2) + value_0"),
        predicate=IndexExpr.parse(False),
    )

    with np.errstate(over="ignore"):
        assert ref.extract((0,)) == 0

    with np.errstate(over="raise"):
        with pytest.raises(FloatingPointError):
            ref.extract((0,))


class MutableInteger(int):
    def __new__(cls):
        result = super().__new__(cls, 0)
        result.calls = 0

        return result

    def __add__(self, other):
        self.calls += 1

        return other + self.calls


def test_mutable_integer_constant_keeps_operator_effects():
    value = MutableInteger()
    expr = IndexExpr(
        op="add",
        operands=(IndexExpr(op="constant", value=value), IndexExpr.parse("value_0")),
    )
    ref = make_ref(np.arange(8, dtype=np.float32), index=expr)
    assert ref.extract((0,)) == 1
    assert ref.extract((0,)) == 2
    assert value.calls == 2


def test_precompiled_custom_operator_is_not_mistaken_for_pure_current_table(
    monkeypatch,
):
    calls = []
    original = expressions.BINARY["add"]

    def observed(a, b):
        calls.append(1)

        return original(a, b)

    expr = IndexExpr.parse("value_0 + 0")

    with monkeypatch.context() as patch:
        patch.setitem(expressions.BINARY, "add", observed)
        expressions.evaluate(expr, {"value_0": np.arange(4, dtype=np.int64)})

    calls.clear()
    ref = make_ref(np.arange(8, dtype=np.float32), index=expr)
    assert ref.extract((0,)) == 0
    assert ref.extract((0,)) == 0
    assert len(calls) == 2


def test_failed_map_is_not_cached(monkeypatch):
    calls = count_accesses(monkeypatch)
    ref = make_ref(np.arange(2, dtype=np.float32))

    for _ in range(2):
        with pytest.raises(IndexError, match="Active layout lane"):
            ref.extract((0,))

    assert len(calls) == 2


@pytest.mark.parametrize("divisor", (1, 2, 3, 7))
def test_integer_interval_proofs_cover_negative_and_positive_values(divisor):
    expression = IndexExpr.parse(f"((x * 3 - 7) // {divisor}) % 11")
    plan = expressions._compiled_expression(expressions._ExpressionIdentity(expression))
    low, high, boolean = _bounds(plan, {"x": (-17, 23, False)}, {})

    with np.errstate(all="raise"):
        actual = expressions.evaluate(
            expression, {"x": np.arange(-17, 24, dtype=np.int64)}
        )

    assert not boolean
    assert low <= actual.min() <= actual.max() <= high


def test_seeded_proven_trees_match_checked_numpy_ranges():
    rng = random.Random(918101)

    def tree(depth):
        if depth == 0 or rng.random() < 0.3:
            return (
                IndexExpr.parse("x")
                if rng.random() < 0.5
                else IndexExpr.parse(rng.randint(-4, 4))
            )

        op = rng.choice(("add", "sub", "mul", "floordiv", "mod", "lt"))
        right = (
            IndexExpr.parse(rng.randint(1, 5))
            if op in ("floordiv", "mod")
            else tree(depth - 1)
        )

        return IndexExpr(op=op, operands=(tree(depth - 1), right))

    proven = 0

    for _ in range(100):
        expression = tree(4)
        plan = expressions._compiled_expression(
            expressions._ExpressionIdentity(expression)
        )
        bounds = _bounds(plan, {"x": (-5, 6, False)}, {})

        if bounds is None:
            continue

        with np.errstate(all="raise"):
            value = expressions.evaluate(
                expression, {"x": np.arange(-5, 7, dtype=np.int64)}
            )

        assert bounds[0] <= np.min(value) <= np.max(value) <= bounds[1]
        proven += 1

    assert proven >= 20


def matrix_arrangement(a, b, out):
    return a.tile((4, 8)), b.tile((8, 4)), out.tile((4, 4))


def matrix_application(a, b, out):
    out = ntl.dot(a, b)  # noqa: F841


@pytest.mark.parametrize(
    "options",
    (
        {},
        {"trace": True},
        {"callback": lambda event: None},
        {"handlers": {"unused": lambda op, args: None}},
        {"watch": ("out",)},
        {"program_ids": ()},
        {"opcodes": ()},
    ),
)
def test_only_pure_untraced_matmul_uses_geometry_reuse(monkeypatch, options):
    kernel = interpret(
        matrix_arrangement,
        matrix_application,
        tuple(Tensor(2, name=n, dtype="float32") for n in ("a", "b", "out")),
        backend="triton",
    )
    a = np.arange(24, dtype=np.float32).reshape(3, 8) / 8
    b = np.arange(40, dtype=np.float32).reshape(8, 5) / 8
    data = {"a": a, "b": b, "out": np.empty((3, 5), dtype=np.float32)}
    calls = []
    original = ExtractionGeometry.access

    def observed(cache, ref):
        calls.append(1)

        return original(cache, ref)

    monkeypatch.setattr(ExtractionGeometry, "access", observed)
    result = interpret_program(
        kernel.program, data, tensors=kernel.tensors, symbols=kernel.meta, **options
    )
    np.testing.assert_allclose(result.outputs["out"], a @ b)
    assert bool(calls) == (options == {})


@pytest.mark.parametrize("symbol", ("outer_index", "extract_0_0"))
def test_shape_proof_uses_original_symbols_before_access_overrides(symbol):
    ref = make_ref(
        np.arange(8, dtype=np.float32),
        shape=IndexExpr.parse(symbol),
        index=IndexExpr.parse("value_0 // (3 - value_0)"),
        predicate=IndexExpr.parse(False),
        symbols={"n": 8, symbol: 4},
    )
    ref = replace(ref, outer_index=2, extracted=((2,),))

    with np.errstate(divide="ignore", invalid="ignore"):
        assert ref.extract((0,)) == 0

    with np.errstate(divide="raise", invalid="raise"):
        with pytest.raises(FloatingPointError):
            ref.extract((0,))


def test_recompiled_operator_invalidates_cached_geometry(monkeypatch):
    ref = make_ref(np.arange(8, dtype=np.float32))
    assert ref.extract((0,)) == 0
    calls = []
    original = expressions.BINARY["add"]

    def observed(a, b):
        calls.append(1)

        return original(a, b)

    expressions._compiled_expression.cache_clear()
    monkeypatch.setitem(expressions.BINARY, "add", observed)
    assert ref.extract((0,)) == 0
    assert ref.extract((0,)) == 0
    assert len(calls) == 2
    expressions._compiled_expression.cache_clear()


def test_equal_custom_symbol_after_cache_hit_keeps_operator_effects():
    ref = make_ref(np.arange(8, dtype=np.float32))
    assert ref.extract((0,)) == 0
    assert ref.extract((0,)) == 0
    value = MutableInteger()
    ref.symbols["offset"] = value
    assert ref.extract((0,)) == 1
    assert ref.extract((0,)) == 2
    assert value.calls == 2


def test_removed_used_symbol_never_reuses_prior_coordinates():
    ref = make_ref(np.arange(8, dtype=np.float32))
    assert ref.extract((0,)) == 0
    del ref.symbols["offset"]

    with pytest.raises(ValueError, match="Unbound layout symbol `offset`"):
        ref.extract((0,))


def test_nonstring_symbol_key_lookup_effects_stay_live():
    class OffsetKey:
        def __init__(self):
            self.calls = 0

        def __hash__(self):
            return hash("offset")

        def __eq__(self, other):
            self.calls += 1

            return other == "offset"

    key = OffsetKey()
    ref = make_ref(np.arange(8, dtype=np.float32), symbols={key: 0, "tile": 4})
    assert ref.extract((0,)) == 0
    first_calls = key.calls
    assert first_calls > 0
    assert ref.extract((0,)) == 0
    assert key.calls > first_calls
