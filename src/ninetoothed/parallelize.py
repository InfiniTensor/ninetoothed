import ast
import functools
import inspect
import textwrap
from dataclasses import dataclass

import ninetoothed.language as ntl
from ninetoothed.make import make
from ninetoothed.symbol import Symbol
from ninetoothed.tensor import Tensor


@dataclass(frozen=True)
class _SerialKernelSpec:
    kind: str


def parallelize(
    func=None,
    *,
    kernel_name=None,
    num_warps=None,
    num_stages=None,
    max_num_configs=None,
):
    """A decorator for lowering supported serial Python kernels.

    :param func: The function to be lowered.
    :param kernel_name: The name for the generated kernel.
    :param num_warps: The number of warps to use.
    :param num_stages: The number of pipeline stages.
    :param max_num_configs: The maximum number of auto-tuning
        configurations to use.
    :return: A handle to the compute kernel.

    .. note::

        This first-stage decorator recognizes only a small set of
        canonical serial loop nests. Today, it supports ``add`` and
        ``mm``-style kernels and raises ``NotImplementedError`` for
        unsupported functions.
    """

    def wrapper(func):
        spec = _analyze_serial_kernel(func)
        compiled = None

        @functools.wraps(func)
        def inner(*args, **kwargs):
            nonlocal compiled

            if compiled is None:
                compiled = _lower_serial_kernel(
                    spec,
                    kernel_name=kernel_name or func.__name__,
                    num_warps=num_warps,
                    num_stages=num_stages,
                    max_num_configs=max_num_configs,
                )

            return compiled(*args, **kwargs)

        inner._serial_kind = spec.kind
        inner._serial_kernel = func

        return inner

    if func is None:
        return wrapper

    return wrapper(func)


def _analyze_serial_kernel(func):
    func_def = _get_function_def(func)
    arg_names = [arg.arg for arg in func_def.args.args]

    if len(arg_names) != 3:
        raise NotImplementedError(_unsupported_message(func.__name__))

    if _is_add_kernel(func_def, arg_names):
        return _SerialKernelSpec("add")

    if _is_mm_kernel(func_def, arg_names):
        return _SerialKernelSpec("mm")

    raise NotImplementedError(_unsupported_message(func.__name__))


def _unsupported_message(func_name):
    return (
        f"{func_name!r} is not a supported serial kernel yet. "
        "The demo decorator currently recognizes canonical `add` and `mm` "
        "loop nests."
    )


def _get_function_def(func):
    source = textwrap.dedent(inspect.getsource(func))
    module = ast.parse(source)

    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == func.__name__:
            return node

    raise ValueError(f"Could not find a function definition for {func.__name__!r}.")


def _statements(func_def):
    body = list(func_def.body)

    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]

    return body


def _is_add_kernel(func_def, arg_names):
    input_name, other_name, output_name = arg_names
    body = _statements(func_def)

    if len(body) != 1 or not isinstance(body[0], ast.For):
        return False

    loop = body[0]

    if not _is_range_over_shape(loop, output_name, 0):
        return False

    if len(loop.body) != 1 or not isinstance(loop.body[0], ast.Assign):
        return False

    if len(loop.body[0].targets) != 1:
        return False

    target = loop.body[0].targets[0]
    index = loop.target.id

    if not _matches_subscript(target, output_name, index):
        return False

    value = loop.body[0].value

    if not isinstance(value, ast.BinOp) or not isinstance(value.op, ast.Add):
        return False

    direct = _matches_subscript(value.left, input_name, index) and _matches_subscript(
        value.right, other_name, index
    )
    swapped = _matches_subscript(value.left, other_name, index) and _matches_subscript(
        value.right, input_name, index
    )

    return direct or swapped


def _is_mm_kernel(func_def, arg_names):
    input_name, other_name, output_name = arg_names
    body = _statements(func_def)

    if len(body) != 1 or not isinstance(body[0], ast.For):
        return False

    loop_m = body[0]

    if not _is_range_over_shape(loop_m, output_name, 0):
        return False

    if len(loop_m.body) != 1 or not isinstance(loop_m.body[0], ast.For):
        return False

    loop_n = loop_m.body[0]

    if not _is_range_over_shape(loop_n, output_name, 1):
        return False

    if len(loop_n.body) != 3:
        return False

    init, loop_k, store = loop_n.body

    if not isinstance(init, ast.Assign) or len(init.targets) != 1:
        return False

    if not isinstance(init.targets[0], ast.Name) or not _is_zero(init.value):
        return False

    accumulator = init.targets[0].id

    if not isinstance(loop_k, ast.For) or not _is_range_over_shape(
        loop_k, input_name, 1
    ):
        return False

    if len(loop_k.body) != 1 or not isinstance(loop_k.body[0], ast.AugAssign):
        return False

    update = loop_k.body[0]

    if (
        not isinstance(update.target, ast.Name)
        or update.target.id != accumulator
        or not isinstance(update.op, ast.Add)
        or not isinstance(update.value, ast.BinOp)
        or not isinstance(update.value.op, ast.Mult)
    ):
        return False

    row_index = loop_m.target.id
    col_index = loop_n.target.id
    reduction_index = loop_k.target.id

    if not (
        _matches_subscript(update.value.left, input_name, row_index, reduction_index)
        and _matches_subscript(
            update.value.right, other_name, reduction_index, col_index
        )
    ):
        return False

    if (
        not isinstance(store, ast.Assign)
        or len(store.targets) != 1
        or not _matches_subscript(store.targets[0], output_name, row_index, col_index)
        or not isinstance(store.value, ast.Name)
        or store.value.id != accumulator
    ):
        return False

    return True


def _is_range_over_shape(node, tensor_name, dim):
    if not isinstance(node.iter, ast.Call):
        return False

    call = node.iter

    if (
        not isinstance(call.func, ast.Name)
        or call.func.id != "range"
        or len(call.args) != 1
        or call.keywords
    ):
        return False

    return _matches_shape_access(call.args[0], tensor_name, dim)


def _matches_shape_access(node, tensor_name, dim):
    if not isinstance(node, ast.Subscript):
        return False

    value = node.value

    if (
        not isinstance(value, ast.Attribute)
        or value.attr != "shape"
        or not isinstance(value.value, ast.Name)
        or value.value.id != tensor_name
    ):
        return False

    return _matches_constant(node.slice, dim)


def _matches_subscript(node, tensor_name, *indices):
    if not isinstance(node, ast.Subscript):
        return False

    if not isinstance(node.value, ast.Name) or node.value.id != tensor_name:
        return False

    actual_indices = (
        node.slice.elts if isinstance(node.slice, ast.Tuple) else (node.slice,)
    )

    if len(actual_indices) != len(indices):
        return False

    return all(
        isinstance(index, ast.Name) and index.id == expected
        for index, expected in zip(actual_indices, indices)
    )


def _matches_constant(node, value):
    return isinstance(node, ast.Constant) and node.value == value


def _is_zero(node):
    return isinstance(node, ast.Constant) and node.value in (0, 0.0)


def _lower_serial_kernel(
    spec,
    *,
    kernel_name,
    num_warps,
    num_stages,
    max_num_configs,
):
    if spec.kind == "add":
        return _lower_add_kernel(
            kernel_name=kernel_name,
            num_warps=num_warps,
            num_stages=num_stages,
            max_num_configs=max_num_configs,
        )

    if spec.kind == "mm":
        return _lower_mm_kernel(
            kernel_name=kernel_name,
            num_warps=num_warps,
            num_stages=num_stages,
            max_num_configs=max_num_configs,
        )

    raise AssertionError(f"Unsupported serial kernel kind: {spec.kind!r}")


def _lower_add_kernel(*, kernel_name, num_warps, num_stages, max_num_configs):
    block_size = Symbol("BLOCK_SIZE", meta=True)

    def arrangement(input, other, output, BLOCK_SIZE=block_size):
        return (
            input.tile((BLOCK_SIZE,)),
            other.tile((BLOCK_SIZE,)),
            output.tile((BLOCK_SIZE,)),
        )

    def application(input, other, output):
        output = input + other  # noqa: F841

    return make(
        arrangement,
        application,
        (Tensor(1), Tensor(1), Tensor(1)),
        kernel_name=kernel_name,
        num_warps=num_warps,
        num_stages=num_stages,
        max_num_configs=max_num_configs,
    )


def _lower_mm_kernel(*, kernel_name, num_warps, num_stages, max_num_configs):
    block_size_m = Symbol("BLOCK_SIZE_M", meta=True)
    block_size_n = Symbol("BLOCK_SIZE_N", meta=True)
    block_size_k = Symbol("BLOCK_SIZE_K", meta=True)

    def arrangement(
        input,
        other,
        output,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
    ):
        output_tiled = output.tile((BLOCK_SIZE_M, BLOCK_SIZE_N))

        input_tiled = (
            input.tile((BLOCK_SIZE_M, BLOCK_SIZE_K))
            .tile((1, -1))
            .expand((-1, output_tiled.shape[1]))
        )
        input_tiled.dtype = input_tiled.dtype.squeeze(0)

        other_tiled = (
            other.tile((BLOCK_SIZE_K, BLOCK_SIZE_N))
            .tile((-1, 1))
            .expand((output_tiled.shape[0], -1))
        )
        other_tiled.dtype = other_tiled.dtype.squeeze(1)

        return input_tiled, other_tiled, output_tiled

    def application(input, other, output):
        accumulator = ntl.zeros(output.shape, dtype=ntl.float32)

        for k in range(input.shape[0]):
            accumulator += ntl.dot(input[k], other[k])

        output = accumulator.to(ntl.float16)

    return make(
        arrangement,
        application,
        (Tensor(2), Tensor(2), Tensor(2)),
        kernel_name=kernel_name,
        num_warps=num_warps,
        num_stages=num_stages,
        max_num_configs=max_num_configs,
    )
