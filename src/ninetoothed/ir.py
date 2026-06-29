"""Intermediate representation objects for backend lowering.

``SSAProgramIR`` is the canonical IR consumed by backend code generation.  The
older structured ``ProgramIR`` records remain in this module as compatibility
objects for legacy tests and migration scripts; application Python source is
lowered directly to SSA by ``ninetoothed.lowering.lower``.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import sympy


@dataclass(frozen=True)
class TensorTypeIR:
    """A backend-neutral view of an application tensor parameter.

    ``shape``/``ndim`` describe the tensor view seen by the NineToothed
    application after ``arrangement`` has run.  ``attrs`` keeps source tensor
    metadata that backend emitters need when they turn view indexing back into
    public pointer indexing.
    """

    name: str
    ndim: int
    dtype: str | None = None
    shape: tuple[str, ...] = ()
    constexpr: bool = False
    jagged_dim: int | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_tensor(cls, tensor: Any) -> "TensorTypeIR":
        source = getattr(tensor, "source", tensor)
        dtype = getattr(source, "dtype", getattr(tensor, "dtype", None))
        shape = getattr(tensor, "shape", getattr(source, "shape", ()))
        source_shape = getattr(source, "shape", ())
        source_ndim = int(
            getattr(source, "ndim", getattr(tensor, "ndim", len(source_shape)))
        )

        return cls(
            name=str(getattr(source, "name", getattr(tensor, "name", "tensor"))),
            ndim=int(getattr(tensor, "ndim", getattr(source, "ndim", len(shape)))),
            dtype=None if dtype is None else str(dtype),
            shape=tuple(_shape_text(size) for size in shape),
            constexpr=bool(
                getattr(source, "constexpr", getattr(tensor, "constexpr", False))
            ),
            jagged_dim=getattr(
                source, "jagged_dim", getattr(tensor, "jagged_dim", None)
            ),
            attrs={
                "source_name": str(
                    getattr(source, "name", getattr(tensor, "name", "tensor"))
                ),
                "source_ndim": source_ndim,
                "source_shape": tuple(_shape_text(size) for size in source_shape),
                "source_dtype": None if dtype is None else str(dtype),
                "source_strides": tuple(
                    str(source.stride_string(dim))
                    for dim in range(source_ndim)
                    if hasattr(source, "stride_string")
                ),
                "target_dims": tuple(
                    None if dim is None else str(dim)
                    for dim in getattr(tensor, "target_dims", ())
                ),
            },
        )


def _shape_text(value: Any) -> str:
    try:
        return str(sympy.simplify(str(value)))
    except Exception:
        return str(value)


@dataclass(frozen=True)
class LaunchIR:
    """Runtime launch metadata shared by generated backends."""

    name: str
    args: tuple[str, ...] = ()
    grid: str | None = None


@dataclass(frozen=True)
class ElementwiseBinaryOpIR:
    """A first structured Program IR node for elementwise binary kernels."""

    operator: str
    lhs: str
    rhs: str
    output: str
    extent: str = "n"


@dataclass(frozen=True)
class ExprIR:
    """Small expression tree shared by elementwise backend lowerers."""

    kind: str
    value: Any = None
    args: tuple["ExprIR", ...] = ()


@dataclass(frozen=True)
class ElementwiseAssignOpIR:
    """A structured elementwise assignment: ``output[i] = expression(i)``."""

    output: str
    expression: ExprIR
    extent: str = "n"


@dataclass(frozen=True)
class AxisReductionAssignOpIR:
    """A row-wise reduction assignment over a 2D flattened tensor."""

    output: str
    expression: ExprIR
    rows: str = "rows"
    cols: str = "cols"
    axis: int = 1


@dataclass(frozen=True)
class RowwiseAssignOpIR:
    """A 2D row-wise assignment with optional row reduction subexpressions."""

    output: str
    expression: ExprIR
    rows: str = "rows"
    cols: str = "cols"


@dataclass(frozen=True)
class FillOpIR:
    """A structured fill kernel: ``output[i] = value``."""

    output: str
    value: float | int | bool
    extent: str = "n"


@dataclass(frozen=True)
class CopyOpIR:
    """A structured copy kernel: ``output[i] = input[i]``."""

    input: str
    output: str
    extent: str = "n"


@dataclass(frozen=True)
class ReductionOpIR:
    """A structured 1D reduction kernel."""

    operator: str
    input: str
    output: str
    extent: str = "n"
    expression: ExprIR | None = None


@dataclass(frozen=True)
class MatmulOpIR:
    """A structured 2D matrix multiplication kernel."""

    lhs: str
    rhs: str
    output: str
    m: str = "m"
    n: str = "n"
    k: str = "k"


@dataclass(frozen=True)
class FlashAttentionOpIR:
    """A structured scaled dot-product attention forward kernel.

    The semantic operation is ``output = softmax(query @ key.T * scale) @ value``.
    Backends are expected to implement it without materializing the full score
    matrix when possible. The first implementation slice uses a correctness
    oriented online-softmax loop.
    """

    query: str
    key: str
    value: str
    output: str
    q_rows: str = "q_rows"
    kv_rows: str = "kv_rows"
    head_dim: str = "head_dim"
    value_dim: str = "value_dim"
    scale: float = 1.0
    causal: bool = False


@dataclass(frozen=True)
class TransposeOpIR:
    """A structured 2D transpose kernel."""

    input: str
    output: str
    rows: str = "rows"
    cols: str = "cols"


@dataclass(frozen=True)
class ProgramIR:
    """Structured operations attached to a kernel before target rendering."""

    kind: str
    operations: tuple[
        ElementwiseBinaryOpIR
        | ElementwiseAssignOpIR
        | AxisReductionAssignOpIR
        | RowwiseAssignOpIR
        | FillOpIR
        | CopyOpIR
        | ReductionOpIR
        | MatmulOpIR
        | FlashAttentionOpIR
        | TransposeOpIR,
        ...,
    ] = ()


@dataclass(frozen=True)
class SSATypeIR:
    """A compact SSA value type.

    ``kind`` is intentionally broad in this first layer: common values are
    ``tensor``, ``scalar``, ``index``, and ``effect``. Shape and attrs keep the
    type serializable while leaving room for a richer type system later.
    """

    kind: str
    dtype: str | None = None
    shape: tuple[str, ...] = ()
    attrs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SSAValueIR:
    """A named SSA value such as ``%0`` or a public tensor argument."""

    name: str
    type: SSATypeIR


@dataclass(frozen=True)
class SSAOperationIR:
    """A single SSA operation.

    Operations are side-effect free unless their opcode explicitly models an
    effect, such as ``mem.store``. Nested regions are represented as blocks so
    loop and online-softmax style kernels can be expressed without changing the
    top-level contract.
    """

    opcode: str
    operands: tuple[str, ...] = ()
    results: tuple[SSAValueIR, ...] = ()
    attrs: Mapping[str, Any] = field(default_factory=dict)
    regions: tuple["SSABlockIR", ...] = ()


@dataclass(frozen=True)
class SSABlockIR:
    """A straight-line SSA block."""

    name: str = "entry"
    args: tuple[SSAValueIR, ...] = ()
    operations: tuple[SSAOperationIR, ...] = ()


@dataclass(frozen=True)
class SSAProgramIR:
    """Canonical SSA-like IR for backend generation.

    The IR exposes dataflow, scalar/tensor values, control-flow regions, and
    effectful memory operations.  Schedule-specific details such as tiling,
    memory scopes, and backend intrinsics can be layered on top through passes.
    """

    kind: str
    inputs: tuple[SSAValueIR, ...] = ()
    outputs: tuple[SSAValueIR, ...] = ()
    blocks: tuple[SSABlockIR, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KernelIR:
    """A kernel-level IR record consumed by backend lowerers.

    New backend generation requires ``ssa`` to be populated.  ``source`` keeps
    the original application text for provenance and reports, not as a source
    passthrough fallback.
    """

    kernel_name: str
    source: str
    source_path: str | None = None
    source_language: str = "triton"
    entrypoint: str | None = None
    launch: LaunchIR | None = None
    tensors: tuple[TensorTypeIR, ...] = ()
    compiler_options: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    program: ProgramIR | None = None
    ssa: SSAProgramIR | None = None

    @classmethod
    def from_codegen(
        cls,
        code_generator: Any,
        source_file: str | Path,
        *,
        kernel_name: str,
        source_language: str = "triton",
        compiler_options: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "KernelIR":
        path = Path(source_file)
        source = path.read_text(encoding="utf-8")

        launch_func = getattr(code_generator, "launch_func", None)
        launch_args: Sequence[str] = ()
        if launch_func is not None:
            launch_args = tuple(arg.arg for arg in launch_func.args.args)

        raw_grid = getattr(code_generator, "raw_grid", None)
        grid = None if raw_grid is None else _safe_unparse(raw_grid)

        tensors = tuple(
            TensorTypeIR.from_tensor(tensor)
            for tensor in getattr(code_generator, "tensors", ())
        )

        return cls(
            kernel_name=kernel_name,
            source=source,
            source_path=str(path),
            source_language=source_language,
            entrypoint=kernel_name,
            launch=LaunchIR(
                name=getattr(
                    code_generator, "launch_func_name", f"launch_{kernel_name}"
                ),
                args=tuple(launch_args),
                grid=grid,
            ),
            tensors=tensors,
            compiler_options=dict(compiler_options or {}),
            metadata=dict(metadata or {}),
        )

    def with_metadata(self, **metadata: Any) -> "KernelIR":
        return type(self)(
            kernel_name=self.kernel_name,
            source=self.source,
            source_path=self.source_path,
            source_language=self.source_language,
            entrypoint=self.entrypoint,
            launch=self.launch,
            tensors=self.tensors,
            compiler_options=self.compiler_options,
            metadata=dict(self.metadata) | metadata,
            program=self.program,
            ssa=self.ssa,
        )


def _safe_unparse(node: Any) -> str:
    try:
        import ast

        return ast.unparse(node)
    except Exception:
        return repr(node)


def program_to_ssa(
    program: ProgramIR,
    tensors: tuple[TensorTypeIR, ...] = (),
) -> SSAProgramIR:
    """Convert legacy structured ProgramIR fixtures to SSA.

    This helper exists for migration tests and historical scripts.  It is not
    used by the application source-to-backend lowering path.
    """

    builder = _SSABuilder(program, tensors)
    builder.lower_program()
    return builder.finish()


def ir_to_dict(value: Any) -> Any:
    """Return a JSON-serializable representation of IR dataclasses."""

    if is_dataclass(value):
        return {
            field.name: ir_to_dict(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return [ir_to_dict(item) for item in value]
    if isinstance(value, list):
        return [ir_to_dict(item) for item in value]
    if isinstance(value, MappingABC):
        return {str(key): ir_to_dict(item) for key, item in value.items()}
    return value


class _SSABuilder:
    def __init__(self, program: ProgramIR, tensors: tuple[TensorTypeIR, ...]):
        self.program = program
        self.tensor_types = {
            tensor.name: SSATypeIR(
                "tensor",
                dtype=tensor.dtype,
                shape=tensor.shape,
                attrs={
                    "ndim": tensor.ndim,
                    "constexpr": tensor.constexpr,
                    "jagged_dim": tensor.jagged_dim,
                },
            )
            for tensor in tensors
        }
        self.values: dict[str, SSAValueIR] = {}
        self.operations: list[SSAOperationIR] = []
        self.outputs: list[SSAValueIR] = []
        self.expr_cache: dict[str, SSAValueIR] = {}
        self.temp_index = 0

        for tensor in tensors:
            self._value(tensor.name, self.tensor_types[tensor.name])

    def lower_program(self) -> None:
        for op in self.program.operations:
            self._lower_operation(op)

    def finish(self) -> SSAProgramIR:
        return SSAProgramIR(
            kind=self.program.kind,
            inputs=tuple(
                self.values[name] for name in self.values if not name.startswith("%")
            ),
            outputs=tuple(self.outputs),
            blocks=(SSABlockIR(operations=tuple(self.operations)),),
            metadata={
                "source": "ProgramIR",
                "operation_count": len(self.program.operations),
                "ssa_operation_count": len(self.operations),
            },
        )

    def _lower_operation(self, op: Any) -> None:
        if isinstance(op, ElementwiseBinaryOpIR):
            lhs = self._value(op.lhs)
            rhs = self._value(op.rhs)
            result = self._emit(
                f"arith.{op.operator}",
                operands=(lhs.name, rhs.name),
                result_type=lhs.type,
            )
            self._store(result, op.output, extent=op.extent)
            return

        if isinstance(op, ElementwiseAssignOpIR):
            result = self._lower_expr(op.expression)
            self._store(result, op.output, extent=op.extent)
            return

        if isinstance(op, AxisReductionAssignOpIR):
            result = self._lower_expr(op.expression)
            self._store(result, op.output, rows=op.rows, cols=op.cols, axis=op.axis)
            return

        if isinstance(op, RowwiseAssignOpIR):
            result = self._lower_expr(op.expression)
            self._store(result, op.output, rows=op.rows, cols=op.cols)
            return

        if isinstance(op, FillOpIR):
            result = self._constant(op.value)
            self._store(result, op.output, extent=op.extent)
            return

        if isinstance(op, CopyOpIR):
            result = self._value(op.input)
            self._store(result, op.output, extent=op.extent)
            return

        if isinstance(op, ReductionOpIR):
            operand = (
                self._lower_expr(op.expression)
                if op.expression is not None
                else self._value(op.input)
            )
            result = self._emit(
                f"reduce.{op.operator}",
                operands=(operand.name,),
                attrs={"extent": op.extent},
                result_type=SSATypeIR("scalar", dtype=operand.type.dtype),
            )
            self._store(result, op.output, extent="1")
            return

        if isinstance(op, MatmulOpIR):
            lhs = self._value(op.lhs)
            rhs = self._value(op.rhs)
            result = self._emit(
                "linalg.matmul",
                operands=(lhs.name, rhs.name),
                attrs={"m": op.m, "n": op.n, "k": op.k},
                result_type=self._tensor_type(op.output),
            )
            self._store(result, op.output, m=op.m, n=op.n, k=op.k)
            return

        if isinstance(op, FlashAttentionOpIR):
            query = self._value(op.query)
            key = self._value(op.key)
            value = self._value(op.value)
            result = self._emit(
                "linalg.flash_attention",
                operands=(query.name, key.name, value.name),
                attrs={
                    "q_rows": op.q_rows,
                    "kv_rows": op.kv_rows,
                    "head_dim": op.head_dim,
                    "value_dim": op.value_dim,
                    "scale": op.scale,
                    "causal": op.causal,
                },
                result_type=self._tensor_type(op.output),
            )
            self._store(result, op.output, q_rows=op.q_rows, value_dim=op.value_dim)
            return

        if isinstance(op, TransposeOpIR):
            operand = self._value(op.input)
            result = self._emit(
                "linalg.transpose",
                operands=(operand.name,),
                attrs={"rows": op.rows, "cols": op.cols},
                result_type=self._tensor_type(op.output),
            )
            self._store(result, op.output, rows=op.rows, cols=op.cols)
            return

        raise TypeError(f"Unsupported ProgramIR operation {type(op).__name__}.")

    def _lower_expr(self, expr: ExprIR) -> SSAValueIR:
        cache_key = repr(ir_to_dict(expr))
        if cache_key in self.expr_cache:
            return self.expr_cache[cache_key]

        if expr.kind == "var":
            result = self._value(str(expr.value))
            self.expr_cache[cache_key] = result
            return result

        if expr.kind == "const":
            result = self._constant(expr.value)
            self.expr_cache[cache_key] = result
            return result

        if expr.kind == "unary":
            operand = self._lower_expr(expr.args[0])
            result = self._emit(
                f"arith.{expr.value}",
                operands=(operand.name,),
                result_type=operand.type,
            )
            self.expr_cache[cache_key] = result
            return result

        if expr.kind == "binary":
            lhs = self._lower_expr(expr.args[0])
            rhs = self._lower_expr(expr.args[1])
            result = self._emit(
                f"arith.{expr.value}",
                operands=(lhs.name, rhs.name),
                result_type=lhs.type,
            )
            self.expr_cache[cache_key] = result
            return result

        if expr.kind == "call":
            operands = tuple(self._lower_expr(arg) for arg in expr.args)
            result = self._emit(
                f"math.{expr.value}",
                operands=tuple(operand.name for operand in operands),
                result_type=operands[0].type if operands else SSATypeIR("scalar"),
            )
            self.expr_cache[cache_key] = result
            return result

        if expr.kind == "axis_reduce":
            operand = self._lower_expr(expr.args[0])
            value = expr.value if isinstance(expr.value, MappingABC) else {}
            operator = str(value.get("operator", "sum"))
            result = self._emit(
                f"reduce.{operator}",
                operands=(operand.name,),
                attrs={"axis": value.get("axis", 1)},
                result_type=SSATypeIR("tensor", dtype=operand.type.dtype),
            )
            self.expr_cache[cache_key] = result
            return result

        if expr.kind == "offset":
            value = expr.value if isinstance(expr.value, MappingABC) else {}
            result = self._emit(
                "index.offset",
                attrs={
                    "tensor": value.get("tensor"),
                    "dim": value.get("dim"),
                },
                result_type=SSATypeIR("index"),
            )
            self.expr_cache[cache_key] = result
            return result

        raise TypeError(f"Unsupported ExprIR kind {expr.kind!r}.")

    def _constant(self, value: Any) -> SSAValueIR:
        dtype = (
            "bool"
            if isinstance(value, bool)
            else "float32"
            if isinstance(value, float)
            else "int64"
        )
        return self._emit(
            "arith.constant",
            attrs={"value": value},
            result_type=SSATypeIR("scalar", dtype=dtype),
        )

    def _store(self, value: SSAValueIR, output: str, **attrs: Any) -> None:
        output_value = self._value(output, self._tensor_type(output))
        if output_value not in self.outputs:
            self.outputs.append(output_value)
        self.operations.append(
            SSAOperationIR(
                "mem.store",
                operands=(value.name, output_value.name),
                attrs={key: item for key, item in attrs.items() if item is not None},
            )
        )

    def _emit(
        self,
        opcode: str,
        *,
        operands: tuple[str, ...] = (),
        attrs: Mapping[str, Any] | None = None,
        result_type: SSATypeIR | None = None,
    ) -> SSAValueIR:
        result = self._temp(result_type or SSATypeIR("tensor"))
        self.operations.append(
            SSAOperationIR(
                opcode,
                operands=operands,
                results=(result,),
                attrs=dict(attrs or {}),
            )
        )
        return result

    def _temp(self, type_: SSATypeIR) -> SSAValueIR:
        name = f"%{self.temp_index}"
        self.temp_index += 1
        return self._value(name, type_)

    def _value(self, name: str, type_: SSATypeIR | None = None) -> SSAValueIR:
        if name not in self.values:
            self.values[name] = SSAValueIR(name, type_ or self._tensor_type(name))
        return self.values[name]

    def _tensor_type(self, name: str) -> SSATypeIR:
        return self.tensor_types.get(name, SSATypeIR("tensor"))
