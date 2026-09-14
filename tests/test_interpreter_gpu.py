"""Required CPU/real-Triton differential checks; missing GPU is not a skip."""

import ast
import hashlib
import operator
import re
from dataclasses import dataclass, replace

import numpy as np
import pytest

import ninetoothed.language as ntl
from ninetoothed import Tensor, interpret
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest
from ninetoothed.interpreter import interpret_program
from ninetoothed.ir import ir_to_dict, ssa

SEED = 2026
RTOL = ATOL = 1e-3


def vector_arrangement(x, y, out):
    return tuple(tensor.tile((256,)) for tensor in (x, y, out))


def vector_add(x, y, out):
    out = x + y  # noqa: F841


def vector_floor_divide(x, y, out):
    out = x // y  # noqa: F841


def vector_remainder(x, y, out):
    out = x % y  # noqa: F841


def broadcast_arrangement(x, bias, out):
    return x.tile((1, 256)), bias.tile((256,)), out.tile((1, 256))


def row_bias_arrangement(x, bias, out):
    return x.tile((1, 256)), bias.tile((1, 256)), out.tile((1, 256))


def column_bias_arrangement(x, bias, out):
    return x.tile((1, 256)), bias.tile((1, 1)), out.tile((1, 256))


def broadcast_add(x, bias, out):
    out = x + bias  # noqa: F841


def reduction_arrangement(x, out):
    return x.tile((1, 512)), out.tile((1,))


def row_sum(x, out):
    out = ntl.sum(x, axis=1)  # noqa: F841


def comparison_arrangement(x, out):
    return x.tile((256,)), out.tile((256,))


def comparison(x, out):
    out = (x >= -2) & (x < 5)  # noqa: F841


def control_arrangement(x, positive, out):
    return x.tile((256,)), positive, out.tile((256,))


def control_flow(x, positive, out):
    accumulator = x

    for i in range(3):
        if positive:
            accumulator = accumulator + i
        else:
            accumulator = accumulator - i

    out = accumulator  # noqa: F841


def softmax_arrangement(x, out):
    return x.tile((1, 512)), out.tile((1, 512))


def row_softmax(x, out):
    shifted = x - ntl.max(x, axis=1)[:, None]
    numerator = ntl.exp(shifted)
    out = numerator / ntl.sum(numerator, axis=1)[:, None]  # noqa: F841


def dot_arrangement(a, b, out):
    return a.tile((4, 4)), b.tile((4, 4)), out.tile((4, 4))


def matrix_dot(a, b, out):
    out = ntl.dot(a, b)  # noqa: F841


@dataclass(frozen=True)
class GPUCase:
    name: str
    category: str
    dtype: str = "float32"
    size: int = 1031
    positive: bool = True
    broadcast_mode: str = "vector"


GPU_CASES = (
    GPUCase("elementwise_float32_aligned", "elementwise", size=1024),
    GPUCase("masked_float32_tail", "masked_tail"),
    GPUCase("masked_int32_tail", "masked_tail", dtype="int32"),
    GPUCase("broadcast_float32_tail", "broadcast"),
    GPUCase("broadcast_row_matrix_float32_tail", "broadcast", broadcast_mode="row"),
    GPUCase(
        "broadcast_column_matrix_float32_tail", "broadcast", broadcast_mode="column"
    ),
    GPUCase("row_reduction_float32", "row_reduction"),
    GPUCase("row_reduction_int32", "row_reduction", dtype="int32"),
    GPUCase("comparison_bool_exact", "comparison", dtype="bool"),
    GPUCase("if_for_true", "if_for", positive=True),
    GPUCase("if_for_false", "if_for", positive=False),
    GPUCase("softmax_float32", "softmax"),
    GPUCase("floor_div_int32_mixed_sign_tail", "floor_div", dtype="int32"),
    GPUCase("remainder_int32_mixed_sign_tail", "remainder", dtype="int32"),
    GPUCase("dot_float32_multi_program_tail", "dot"),
)


def require_gpu(device_index=0):
    """Load GPU packages only when GPU validation is actually requested."""
    try:
        import torch
        import triton
    except ImportError as error:
        raise RuntimeError(
            "GPU validation UNVERIFIED: PyTorch and Triton must be installed."
        ) from error

    if not torch.cuda.is_available():
        raise RuntimeError("GPU validation UNVERIFIED: no usable CUDA GPU.")

    if not 0 <= device_index < torch.cuda.device_count():
        raise RuntimeError(f"GPU validation UNVERIFIED: invalid device {device_index}.")

    torch.cuda.set_device(device_index)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    return torch, triton


def _descriptor(name, ndim, dtype="float32", **kwargs):
    return Tensor(ndim, name=name, dtype=dtype, **kwargs)


def _signed_division_inputs(size):
    # All sign combinations, exact/inexact quotients, zero numerators, and
    # int32 extremes. Division by zero and INT_MIN / -1 are outside this test.
    pairs = np.array(
        [
            (-7, 3),
            (7, -3),
            (-7, -3),
            (7, 3),
            (-6, 3),
            (6, -3),
            (-6, -3),
            (6, 3),
            (0, 3),
            (0, -3),
            (-1, 3),
            (1, -3),
            (-2147483648, 3),
            (2147483647, -3),
            (-2147483648, -3),
            (2147483647, 3),
            (-2147483648, 2147483647),
            (2147483647, -2147483648),
            (-2147483647, -1),
            (-2147483648, 1),
            (-1, -2147483648),
            (1, 2147483647),
        ],
        dtype=np.int32,
    )
    selected = pairs[np.arange(size) % len(pairs)]

    return selected[:, 0].copy(), selected[:, 1].copy()


def case_inputs(case):
    """Construct fixed-seed inputs and an independent NumPy oracle."""
    rng = np.random.default_rng(SEED)
    dtype = np.dtype(case.dtype)

    if case.category == "dot":
        # Four independent M/N output tiles; K is complete within each tile.
        # This covers scalar float32 lowering, not Tensor Core instructions.
        a = rng.normal(size=(7, 3)).astype(np.float32)
        b = rng.normal(size=(3, 6)).astype(np.float32)

        return (
            dot_arrangement,
            matrix_dot,
            (
                _descriptor("a", 2, other=0),
                _descriptor("b", 2, other=0),
                _descriptor("out", 2),
            ),
            {"a": a, "b": b},
            a @ b,
        )

    if case.category in {"floor_div", "remainder"}:
        x, y = _signed_division_inputs(case.size)
        function = operator.floordiv if case.category == "floor_div" else operator.mod
        # Python integers are an independent, exact oracle: no GPU or NumPy
        # division, float conversion, or rounding assumption is involved.
        expected = np.array(
            [function(int(lhs), int(rhs)) for lhs, rhs in zip(x, y)],
            dtype=np.int32,
        )

        return (
            vector_arrangement,
            vector_floor_divide if case.category == "floor_div" else vector_remainder,
            (
                _descriptor("x", 1, "int32"),
                _descriptor("y", 1, "int32", other=1),
                _descriptor("out", 1, "int32"),
            ),
            {"x": x, "y": y},
            expected,
        )

    if case.category in {"elementwise", "masked_tail"}:
        if dtype.kind == "i":
            x = rng.integers(-100, 100, size=case.size, dtype=dtype)
            y = rng.integers(-100, 100, size=case.size, dtype=dtype)
        else:
            x = rng.normal(size=case.size).astype(dtype)
            y = rng.normal(size=case.size).astype(dtype)
        return (
            vector_arrangement,
            vector_add,
            tuple(_descriptor(name, 1, case.dtype) for name in ("x", "y", "out")),
            {"x": x, "y": y},
            x + y,
        )

    if case.category == "broadcast":
        x = rng.normal(size=(7, 257)).astype(np.float32)
        bias_shapes = {"vector": (257,), "row": (1, 257), "column": (7, 1)}
        bias = rng.normal(size=bias_shapes[case.broadcast_mode]).astype(np.float32)
        arrangement = {
            "vector": broadcast_arrangement,
            "row": row_bias_arrangement,
            "column": column_bias_arrangement,
        }[case.broadcast_mode]

        return (
            arrangement,
            broadcast_add,
            (
                _descriptor("x", 2),
                _descriptor("bias", bias.ndim),
                _descriptor("out", 2),
            ),
            {"x": x, "bias": bias},
            x + bias,
        )

    if case.category == "row_reduction":
        x = (
            rng.integers(-10, 10, size=(7, 257), dtype=dtype)
            if dtype.kind == "i"
            else rng.normal(size=(7, 257)).astype(dtype)
        )

        return (
            reduction_arrangement,
            row_sum,
            (
                _descriptor("x", 2, case.dtype, other=0),
                _descriptor("out", 1, case.dtype),
            ),
            {"x": x},
            x.sum(axis=1, dtype=dtype),
        )

    if case.category == "comparison":
        x = rng.integers(-10, 10, size=case.size, dtype=np.int32)

        return (
            comparison_arrangement,
            comparison,
            (_descriptor("x", 1, "int32"), _descriptor("out", 1, "bool")),
            {"x": x},
            (x >= -2) & (x < 5),
        )

    if case.category == "if_for":
        x = rng.normal(size=case.size).astype(np.float32)

        return (
            control_arrangement,
            control_flow,
            (
                _descriptor("x", 1),
                _descriptor("positive", 0, "bool", constexpr=True),
                _descriptor("out", 1),
            ),
            {"x": x, "positive": case.positive},
            x + (3 if case.positive else -3),
        )

    if case.category == "softmax":
        x = rng.normal(size=(7, 257)).astype(np.float32) * 3
        x[0] += 1000  # Stable subtraction must avoid exponential overflow.
        numerator = np.exp(x - np.max(x, axis=1, keepdims=True))
        expected = numerator / np.sum(numerator, axis=1, keepdims=True)

        return (
            softmax_arrangement,
            row_softmax,
            (_descriptor("x", 2, other=float("-inf")), _descriptor("out", 2)),
            {"x": x},
            expected,
        )

    raise ValueError(f"Unknown validation category: {case.category}.")


def _program_from_metadata(data):
    """Reconstruct the exact emitted SSA, after genuine compiler lowering."""

    def value(item):
        return ssa.Value(name=item["name"], type=ssa.Type(**item["type"]))

    def block(item):
        return ssa.Block(
            name=item["name"],
            args=tuple(value(arg) for arg in item["args"]),
            operations=tuple(
                ssa.Operation(
                    opcode=op["opcode"],
                    operands=op["operands"],
                    results=tuple(value(result) for result in op["results"]),
                    attrs=op["attrs"],
                    regions=tuple(block(region) for region in op["regions"]),
                    origins=tuple(op.get("origins", ())),
                )
                for op in item["operations"]
            ),
        )

    return ssa.verify_program(
        ssa.Program(
            kind=data["kind"],
            inputs=tuple(value(item) for item in data["inputs"]),
            outputs=tuple(value(item) for item in data["outputs"]),
            blocks=tuple(block(item) for item in data["blocks"]),
            metadata=data["metadata"],
        )
    )


def _assert_equal(actual, expected, label):
    assert actual.shape == expected.shape, label
    assert actual.dtype == expected.dtype, label

    if expected.dtype.kind == "f":
        np.testing.assert_allclose(
            actual, expected, rtol=RTOL, atol=ATOL, err_msg=label
        )
    else:
        np.testing.assert_array_equal(actual, expected, err_msg=label)


def _scalar_dot_offsets(program):
    """Require actual scalar-loop lowering, independently of metadata flags."""

    def walk(block):
        for operation in block.operations:
            yield operation

            for region in operation.regions:
                yield from walk(region)

    operations = tuple(op for block in program.blocks for op in walk(block))
    assert not any(op.opcode in {"linalg.dot", "linalg.matmul"} for op in operations)
    assert any(
        op.opcode == "scf.for" and op.attrs.get("decomposition") == "matmul"
        for op in operations
    )
    offsets = {
        op.attrs["dim"]: op.results[0].name
        for op in operations
        if op.opcode == "index.offset" and op.attrs.get("decomposition") == "matmul"
    }
    assert set(offsets) == {0, 1}

    return offsets


def run_gpu_case(case, torch, device_index=0):
    """Compare raw SSA, emitted target SSA, actual GPU output, and NumPy."""
    arrangement, application, tensors, source_inputs, expected = case_inputs(case)
    cpu = interpret(arrangement, application, tensors)
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=arrangement,
            application=application,
            tensors=tensors,
            backend="triton",
            kernel_name=f"gpu_validation_{case.name}",
            num_warps=4,
            max_num_configs=1,
        )
    )
    assert compilation.artifact.metadata["lowering_ir"] == "ssa.Program"
    assert compilation.artifact.metadata["generation_py_fallback"] is False
    assert ssa.render(
        replace(cpu.program, kind=compilation.kernel.ssa.kind)
    ) == ssa.render(compilation.kernel.ssa)
    lowered = _program_from_metadata(compilation.artifact.metadata["ssa"])
    assert "ssa.triton.optimize_schedule" in lowered.metadata["pass_trace"]

    if case.category == "dot":
        _scalar_dot_offsets(lowered)
        assert ir_to_dict(lowered) == compilation.artifact.metadata["ssa"]

    def cpu_inputs():
        return {
            **{
                name: value.copy() if isinstance(value, np.ndarray) else value
                for name, value in source_inputs.items()
            },
            "out": np.full_like(expected, -123 if expected.dtype.kind != "b" else True),
        }

    raw_inputs = cpu_inputs()
    cpu(**raw_inputs)
    _assert_equal(raw_inputs["out"], expected, "frontend SSA versus NumPy")
    lowered_inputs = cpu_inputs()
    interpreted = interpret_program(
        lowered,
        lowered_inputs,
        tensors=compilation.kernel.tensors,
        symbols=compilation.kernel.metadata.get("meta_defaults", {}),
    )
    _assert_equal(
        interpreted.outputs["out"], expected, "emitted target SSA versus NumPy"
    )

    device = torch.device("cuda", device_index)
    gpu_inputs = {
        name: torch.from_numpy(value.copy()).to(device)
        if isinstance(value, np.ndarray)
        else value
        for name, value in source_inputs.items()
    }
    guard = True if expected.dtype.kind == "b" else -13579
    backing = torch.from_numpy(
        np.full(expected.size + 8, guard, dtype=expected.dtype)
    ).to(device)
    gpu_inputs["out"] = backing[4:-4].reshape(expected.shape)
    launch = DEFAULT_COMPILER.materialize(compilation)
    launch(**gpu_inputs)
    torch.cuda.synchronize(device)
    actual = gpu_inputs["out"].cpu().numpy()

    if case.category == "broadcast":
        for row in range(expected.shape[0]):
            _assert_equal(
                actual[row],
                expected[row],
                f"GPU broadcast row {row}, including the final one-element tile",
            )

    _assert_equal(actual, raw_inputs["out"], "actual Triton GPU versus frontend SSA")
    _assert_equal(
        actual, interpreted.outputs["out"], "actual Triton GPU versus emitted SSA"
    )
    _assert_equal(actual, expected, "actual Triton GPU versus independent NumPy")
    guards = backing.cpu().numpy()[np.r_[0:4, expected.size + 4 : expected.size + 8]]
    np.testing.assert_array_equal(
        guards, np.full(8, guard, dtype=expected.dtype), err_msg="GPU output overrun"
    )

    for name, expected_input in source_inputs.items():
        if isinstance(expected_input, np.ndarray):
            np.testing.assert_array_equal(
                gpu_inputs[name].cpu().numpy(),
                expected_input,
                err_msg=f"Input mutated: {name}",
            )
    return {
        "name": case.name,
        "category": case.category,
        "program": application.__name__,
        "dtype": str(expected.dtype),
        "output_shape": list(expected.shape),
        "seed": SEED,
        "status": "PASS",
        "guard_lanes_unchanged": True,
        "pass_trace": list(compilation.pass_trace),
        "emitted_ssa_sha256": hashlib.sha256(ssa.render(lowered).encode()).hexdigest(),
        "max_abs_error": float(
            np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))
        ),
    }


@pytest.fixture(scope="module")
def gpu_runtime():
    try:
        return require_gpu()
    except RuntimeError as error:
        pytest.fail(str(error), pytrace=False)


@pytest.mark.parametrize("case", GPU_CASES, ids=lambda case: case.name)
def test_cpu_interpreter_matches_actual_triton_gpu(case, gpu_runtime):
    torch, _triton = gpu_runtime
    run_gpu_case(case, torch)


def _compile_case(case):
    arrangement, application, tensors, inputs, expected = case_inputs(case)

    return (
        DEFAULT_COMPILER.compile(
            CompileRequest(
                arrangement=arrangement,
                application=application,
                tensors=tensors,
                backend="triton",
                kernel_name=f"gpu_validation_{case.name}",
                num_warps=4,
                max_num_configs=1,
            )
        ),
        inputs,
        expected,
    )


@pytest.mark.parametrize(
    "case",
    tuple(
        case
        for case in GPU_CASES
        if case.category in {"broadcast", "floor_div", "remainder", "dot"}
    ),
    ids=lambda case: case.name,
)
def test_gpu_fixtures_match_oracle_before_and_after_lowering_on_cpu(case):
    """Verify selected GPU fixtures without CUDA or Triton installed."""
    arrangement, application, tensors, inputs, expected = case_inputs(case)
    compilation, _, _ = _compile_case(case)
    lowered = _program_from_metadata(compilation.artifact.metadata["ssa"])
    snapshots = {name: value.copy() for name, value in inputs.items()}

    if case.category == "dot":
        _scalar_dot_offsets(lowered)
        assert ir_to_dict(lowered) == compilation.artifact.metadata["ssa"]

    for label, program in (
        ("frontend SSA", interpret(arrangement, application, tensors).program),
        ("emitted target SSA", lowered),
    ):
        arguments = {
            **{name: value.copy() for name, value in inputs.items()},
            "out": np.full_like(expected, -123),
        }
        result = interpret_program(
            program,
            arguments,
            tensors=compilation.kernel.tensors,
            symbols=compilation.kernel.metadata.get("meta_defaults", {}),
        )
        _assert_equal(result.outputs["out"], expected, label)

        for name, snapshot in snapshots.items():
            np.testing.assert_array_equal(arguments[name], snapshot)


def _jit_function(source):
    tree = ast.parse(source)

    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(decorator, ast.Attribute)
            and isinstance(decorator.value, ast.Name)
            and decorator.value.id == "triton"
            and decorator.attr == "jit"
            for decorator in node.decorator_list
        )
    )


@pytest.mark.parametrize("case", GPU_CASES[:3], ids=lambda case: case.name)
def test_generated_stride_guards_use_binary_boolops_supported_by_triton_31(case):
    compilation, _inputs, _expected = _compile_case(case)
    kernel = _jit_function(compilation.artifact.primary_source)
    guards = [node for node in ast.walk(kernel) if isinstance(node, ast.BoolOp)]
    assert guards, "The contiguous-stride fast-path guard unexpectedly disappeared."
    offenders = [
        (node.lineno, ast.unparse(node)) for node in guards if len(node.values) > 2
    ]
    assert not offenders, f"Triton 3.1 does not support chained BoolOp: {offenders}."


def _numeric_source_expression(node, values, *, tensor_division=False):
    """Evaluate only emitted address/mask arithmetic, with no Python execution."""

    def evaluate(child):
        return _numeric_source_expression(
            child, values, tensor_division=tensor_division
        )

    binary = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.BitAnd: operator.and_,
        ast.BitOr: operator.or_,
    }
    comparison = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
    }

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        return values[node.id]

    if isinstance(node, ast.BinOp) and type(node.op) in binary:
        lhs, rhs = evaluate(node.left), evaluate(node.right)

        if tensor_division and isinstance(node.op, (ast.FloorDiv, ast.Mod)):
            # Simulate documented Triton tensor C semantics with exact int64
            # arithmetic. The int32 fixtures exclude zero/overflow division.
            quotient = (np.abs(lhs) // np.abs(rhs)) * np.where(
                (lhs < 0) != (rhs < 0), -1, 1
            )

            return (
                quotient if isinstance(node.op, ast.FloorDiv) else lhs - rhs * quotient
            )
        return binary[type(node.op)](lhs, rhs)

    if isinstance(node, ast.UnaryOp):
        function = {
            ast.USub: operator.neg,
            ast.UAdd: operator.pos,
            ast.Invert: operator.invert,
        }[type(node.op)]

        return function(evaluate(node.operand))

    if isinstance(node, ast.Compare):
        left = evaluate(node.left)
        result = True

        for operation, child in zip(node.ops, node.comparators):
            right = evaluate(child)
            result = np.logical_and(result, comparison[type(operation)](left, right))
            left = right
        return result

    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tl"
    ):
        if node.func.attr == "where":
            return np.where(*(evaluate(arg) for arg in node.args))

        if node.func.attr == "load":
            source_names = {
                child.id
                for child in ast.walk(node.args[0])
                if isinstance(child, ast.Name) and child.id in values
            }
            assert len(source_names) == 1

            return values[source_names.pop()]

    raise AssertionError(f"Unsupported generated address expression: {ast.dump(node)}.")


@pytest.mark.parametrize("backend", ("triton", "cuda"))
def test_emitted_dot_coordinates_stay_local_across_output_programs(backend):
    """Inspect emitted scalar indices; this does not execute either GPU backend."""
    case = next(case for case in GPU_CASES if case.category == "dot")
    arrangement, application, tensors, _inputs, _expected = case_inputs(case)
    compilation = DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=arrangement,
            application=application,
            tensors=tensors,
            backend=backend,
            kernel_name="dot_coordinate_validation",
            num_warps=4,
            max_num_configs=1,
        )
    )
    lowered = _program_from_metadata(compilation.artifact.metadata["ssa"])
    offsets = _scalar_dot_offsets(lowered)
    source = compilation.artifact.primary_source

    if backend == "triton":
        bindings = {
            node.targets[0].id: node.value
            for node in ast.walk(_jit_function(source))
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        }
    else:
        # Only integer coordinate assignments are interpreted. All indices in
        # this fixture are nonnegative, so C division agrees with Python //.
        bindings = {
            name: expression
            for name, expression in re.findall(
                r"\bint64_t\s+(\w+)\s*=\s*([^;\n]+);", source
            )
        }

    indices = np.arange(4 * 4 * 4, dtype=np.int64)
    values = {"index": indices}

    def resolve(name):
        if name in values:
            return values[name]

        expression = bindings[name]

        if isinstance(expression, str):
            expression = ast.parse(expression.replace("/", "//"), mode="eval").body

        dependencies = {
            child.id: resolve(child.id)
            for child in ast.walk(expression)
            if isinstance(child, ast.Name)
        }
        values[name] = _numeric_source_expression(expression, dependencies)

        return values[name]

    for dimension, value_name in offsets.items():
        # SSA %N is rendered as vN, with a scope suffix inside the K loop.
        rendered_name = "v" + value_name.removeprefix("%")
        matching = [
            name
            for name in bindings
            if name == rendered_name or name.startswith(rendered_name + "_")
        ]
        assert matching, f"No emitted coordinate binding for {value_name}."
        expected = (indices % 16) // 4 if dimension == 0 else indices % 4

        for name in matching:
            np.testing.assert_array_equal(
                resolve(name),
                expected,
                err_msg=f"{backend} dot axis {dimension} contains a program offset",
            )


@pytest.mark.parametrize(
    "case",
    tuple(case for case in GPU_CASES if case.category in {"floor_div", "remainder"}),
    ids=lambda case: case.name,
)
def test_generated_signed_division_corrects_triton_tensor_rounding(case):
    """Check emitted arithmetic under C semantics before expensive GPU runs."""
    compilation, inputs, expected = _compile_case(case)
    kernel = _jit_function(compilation.artifact.primary_source)
    stores = [
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tl"
        and node.func.attr == "store"
    ]
    assert stores
    output_names = {store.args[1].id for store in stores}
    expressions = [
        node.value
        for node in ast.walk(kernel)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in output_names
            for target in node.targets
        )
    ]
    assert expressions

    for expression in expressions:
        actual = _numeric_source_expression(
            expression,
            {name: value.astype(np.int64) for name, value in inputs.items()},
            tensor_division=True,
        )
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "case",
    tuple(case for case in GPU_CASES if case.category == "broadcast"),
    ids=lambda case: case.name,
)
@pytest.mark.parametrize("tensor_name", ("x", "bias"))
def test_generated_broadcast_input_addresses_and_masks_match_numpy_layout(
    case, tensor_name
):
    compilation, inputs, expected = _compile_case(case)
    inputs = dict(inputs, out=np.empty_like(expected))
    kernel = _jit_function(compilation.artifact.primary_source)
    loads = [
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "tl"
        and node.func.attr == "load"
        and any(
            isinstance(child, ast.Name) and child.id == tensor_name
            for child in ast.walk(node.args[0])
        )
    ]
    assert loads, f"No generated {tensor_name} load found."
    symbols = {}

    for binding in compilation.launch_abi.kernel_args:
        if binding.source not in inputs:
            continue

        value = inputs[binding.source]

        if binding.kind == "shape":
            symbols[binding.name] = value.shape[binding.dim]
        elif binding.kind == "stride":
            symbols[binding.name] = value.strides[binding.dim] // value.itemsize
        elif binding.kind == "tensor":
            symbols[binding.name] = 0  # Address relative to each allocation.

    padded_columns = 512
    symbols["index"] = np.arange(expected.shape[0] * padded_columns, dtype=np.int64)
    symbols["mask"] = np.ones(symbols["index"].shape, dtype=bool)

    if tensor_name == "x":
        rows, columns = np.divmod(symbols["index"], padded_columns)
        expected_addresses = rows * expected.shape[1] + columns
        expected_mask = columns < expected.shape[1]
        valid_lanes_per_row = expected.shape[1]
    elif case.broadcast_mode == "column":
        expected_addresses = symbols["index"] // padded_columns
        # The scalar bias is valid even on padded output lanes; output stores
        # have their own tail mask and the real-GPU test checks storage guards.
        expected_mask = np.ones(symbols["index"].shape, dtype=bool)
        valid_lanes_per_row = padded_columns
    else:
        expected_addresses = symbols["index"] % padded_columns
        expected_mask = expected_addresses < expected.shape[1]
        valid_lanes_per_row = expected.shape[1]

    for load in loads:
        address = _numeric_source_expression(load.args[0], symbols)
        mask_node = next(
            keyword.value for keyword in load.keywords if keyword.arg == "mask"
        )
        actual_mask = np.asarray(
            _numeric_source_expression(mask_node, symbols), dtype=bool
        )
        np.testing.assert_array_equal(actual_mask, expected_mask)
        np.testing.assert_array_equal(
            address[expected_mask], expected_addresses[expected_mask]
        )

        for row in range(expected.shape[0]):
            row_slice = slice(row * padded_columns, (row + 1) * padded_columns)
            assert actual_mask[row_slice].sum() == valid_lanes_per_row, (
                f"Tensor {tensor_name} lanes missing in row {row}."
            )
