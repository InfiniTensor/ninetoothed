"""Kernel, layout, and launch IR records."""

import ast
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Mapping

import sympy

from ninetoothed.ir.frozen import freeze
from ninetoothed.ir.ssa import Program


@dataclass(frozen=True, kw_only=True)
class IndexExpr:
    """Backend-neutral expression used by layout and launch mappings."""

    op: str
    operands: tuple["IndexExpr", ...] = ()
    value: Any = None

    def __post_init__(self):
        object.__setattr__(self, "operands", tuple(self.operands))

    @classmethod
    def parse(cls, value: Any) -> "IndexExpr":
        if isinstance(value, cls):
            return value

        if isinstance(value, (bool, int, float)):
            return cls(op="constant", value=value)

        node = ast.parse(str(value), mode="eval").body

        return _index_expr_from_ast(node)

    def render(self) -> str:
        if self.op == "constant":
            return repr(self.value)

        if self.op == "symbol":
            return str(self.value)

        if self.op == "attribute":
            return f"{self.operands[0].render()}.{self.value}"

        if self.op == "call":
            return f"{self.value}({', '.join(item.render() for item in self.operands)})"

        if self.op == "subscript":
            return f"{self.operands[0].render()}[{self.operands[1].render()}]"

        if self.op == "tuple":
            values = ", ".join(item.render() for item in self.operands)

            return f"({values}{',' if len(self.operands) == 1 else ''})"

        if self.op in {"neg", "pos", "invert", "not"}:
            token = {"neg": "-", "pos": "+", "invert": "~", "not": "not "}[self.op]

            return f"({token}{self.operands[0].render()})"

        if len(self.operands) == 2:
            token = _INDEX_BINARY_TOKENS.get(self.op, self.op)

            return f"({self.operands[0].render()} {token} {self.operands[1].render()})"

        raise ValueError(f"Cannot render index expression operation `{self.op}`.")


@dataclass(frozen=True, kw_only=True)
class AccessMap:
    """Map logical view coordinates to source tensor coordinates."""

    source_indices: tuple[IndexExpr, ...]
    linear_index: IndexExpr
    predicate: IndexExpr

    def __post_init__(self):
        object.__setattr__(self, "source_indices", tuple(self.source_indices))


@dataclass(frozen=True, kw_only=True)
class LayoutLevel:
    """One arranged tensor view level."""

    shape: tuple[IndexExpr, ...]
    target_dims: tuple[IndexExpr | None, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "target_dims", tuple(self.target_dims))


@dataclass(frozen=True, kw_only=True)
class TensorLayout:
    """Structured layout semantics shared by all backend emitters."""

    source_shape: tuple[IndexExpr, ...]
    source_strides: tuple[IndexExpr, ...]
    view_shape: tuple[IndexExpr, ...]
    application_shape: tuple[IndexExpr, ...]
    levels: tuple[LayoutLevel, ...] = ()
    view_access: AccessMap | None = None
    value_accesses: tuple[AccessMap, ...] = ()

    def __post_init__(self):
        for name in (
            "source_shape",
            "source_strides",
            "view_shape",
            "application_shape",
            "levels",
            "value_accesses",
        ):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True, kw_only=True)
class TensorSpec:
    """A backend-neutral view of an application tensor parameter.

    ``name`` is the application parameter and SSA binding name. Source tensor
    provenance lives in ``attrs["source_name"]`` when available.
    """

    ndim: int
    shape: tuple[str, ...] = ()
    dtype: str | None = None
    jagged_dim: int | None = None
    constexpr: bool = False
    name: str
    layout: TensorLayout | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "attrs", freeze(self.attrs))

    @classmethod
    def from_tensor(cls, tensor: Any) -> "TensorSpec":
        source = getattr(tensor, "source", tensor)
        dtype = getattr(source, "dtype", getattr(tensor, "dtype", None))
        shape = getattr(tensor, "shape", getattr(source, "shape", ()))
        source_shape = getattr(source, "shape", ())
        source_ndim = int(
            getattr(source, "ndim", getattr(tensor, "ndim", len(source_shape)))
        )

        return cls(
            ndim=int(getattr(tensor, "ndim", getattr(source, "ndim", len(shape)))),
            shape=tuple(_shape_text(size) for size in shape),
            dtype=None if dtype is None else str(dtype),
            jagged_dim=getattr(
                source, "jagged_dim", getattr(tensor, "jagged_dim", None)
            ),
            constexpr=bool(
                getattr(source, "constexpr", getattr(tensor, "constexpr", False))
            ),
            name=str(getattr(source, "name", getattr(tensor, "name", "tensor"))),
            layout=None,
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


@dataclass(frozen=True, kw_only=True)
class LaunchBinding:
    """Bind a generated kernel argument to a public runtime argument."""

    name: str
    kind: str
    source: str | None = None
    dim: int | None = None
    value: Any = None
    access: str | None = None

    def __post_init__(self):
        if self.access not in {None, "read", "write", "read_write"}:
            raise ValueError(f"Unsupported launch binding access `{self.access}`.")


@dataclass(frozen=True, kw_only=True)
class LaunchABI:
    """Backend-neutral runtime contract attached to a compiled kernel."""

    public_args: tuple[str, ...] = ()
    kernel_args: tuple[LaunchBinding, ...] = ()
    outputs: tuple[str, ...] = ()
    shape_params: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "public_args", tuple(self.public_args))
        object.__setattr__(self, "kernel_args", tuple(self.kernel_args))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "shape_params", tuple(self.shape_params))


@dataclass(frozen=True, kw_only=True)
class LaunchPlan:
    """Backend-neutral launch domain and specialization contract."""

    abi: LaunchABI
    grid: tuple[IndexExpr, ...] = ()
    block: tuple[IndexExpr, ...] = ()
    dynamic_parameters: tuple[str, ...] = ()
    specialization_key: tuple[str, ...] = ()
    tuning_candidates: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "grid", tuple(self.grid))
        object.__setattr__(self, "block", tuple(self.block))
        object.__setattr__(self, "dynamic_parameters", tuple(self.dynamic_parameters))
        object.__setattr__(self, "specialization_key", tuple(self.specialization_key))
        object.__setattr__(
            self,
            "tuning_candidates",
            tuple(freeze(candidate) for candidate in self.tuning_candidates),
        )


@dataclass(frozen=True, kw_only=True)
class Kernel:
    """A kernel-level IR record consumed by backend emitters."""

    kernel_name: str
    source: str
    source_path: str | None = None
    source_language: str = "triton"
    entrypoint: str | None = None
    launch_abi: LaunchABI | None = None
    launch_plan: LaunchPlan | None = None
    tensors: tuple[TensorSpec, ...] = ()
    compiler_options: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    ssa: Program | None = None

    def __post_init__(self):
        object.__setattr__(self, "tensors", tuple(self.tensors))
        object.__setattr__(self, "compiler_options", freeze(self.compiler_options))
        object.__setattr__(self, "metadata", freeze(self.metadata))

    def with_metadata(self, **metadata: Any) -> "Kernel":
        return type(self)(
            kernel_name=self.kernel_name,
            source=self.source,
            source_path=self.source_path,
            source_language=self.source_language,
            entrypoint=self.entrypoint,
            launch_abi=self.launch_abi,
            launch_plan=self.launch_plan,
            tensors=self.tensors,
            compiler_options=self.compiler_options,
            metadata=dict(self.metadata) | metadata,
            ssa=self.ssa,
        )


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


def _shape_text(value: Any) -> str:
    try:
        return str(sympy.simplify(str(value)))
    except Exception:
        return str(value)


_INDEX_BINARY_TOKENS = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "div": "/",
    "floordiv": "//",
    "mod": "%",
    "pow": "**",
    "and": "and",
    "or": "or",
    "bitand": "&",
    "bitor": "|",
    "bitxor": "^",
    "eq": "==",
    "ne": "!=",
    "lt": "<",
    "le": "<=",
    "gt": ">",
    "ge": ">=",
}


def _index_expr_from_ast(node: ast.AST) -> IndexExpr:
    try:
        handler = _INDEX_AST_HANDLERS[type(node)]
    except KeyError as exc:
        raise ValueError(f"Unsupported index expression: {ast.dump(node)}.") from exc
    return handler(node)


def _index_constant(node: ast.Constant) -> IndexExpr:
    return IndexExpr(op="constant", value=node.value)


def _index_name(node: ast.Name) -> IndexExpr:
    return IndexExpr(op="symbol", value=node.id)


def _index_attribute(node: ast.Attribute) -> IndexExpr:
    return IndexExpr(
        op="attribute",
        operands=(_index_expr_from_ast(node.value),),
        value=node.attr,
    )


def _index_binary(node: ast.BinOp) -> IndexExpr:
    operations = {
        ast.Add: "add",
        ast.Sub: "sub",
        ast.Mult: "mul",
        ast.Div: "div",
        ast.FloorDiv: "floordiv",
        ast.Mod: "mod",
        ast.Pow: "pow",
        ast.BitAnd: "bitand",
        ast.BitOr: "bitor",
        ast.BitXor: "bitxor",
    }

    try:
        operation = operations[type(node.op)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported index binary operator: {ast.dump(node.op)}."
        ) from exc
    return IndexExpr(
        op=operation,
        operands=(_index_expr_from_ast(node.left), _index_expr_from_ast(node.right)),
    )


def _index_unary(node: ast.UnaryOp) -> IndexExpr:
    operations = {
        ast.USub: "neg",
        ast.UAdd: "pos",
        ast.Invert: "invert",
        ast.Not: "not",
    }

    try:
        operation = operations[type(node.op)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported index unary operator: {ast.dump(node.op)}."
        ) from exc
    return IndexExpr(op=operation, operands=(_index_expr_from_ast(node.operand),))


def _index_bool(node: ast.BoolOp) -> IndexExpr:
    operation = "and" if isinstance(node.op, ast.And) else "or"
    result = _index_expr_from_ast(node.values[0])

    for value in node.values[1:]:
        result = IndexExpr(
            op=operation,
            operands=(result, _index_expr_from_ast(value)),
        )
    return result


def _index_compare(node: ast.Compare) -> IndexExpr:
    if len(node.ops) != 1 or len(node.comparators) != 1:
        raise ValueError("Index comparisons must contain exactly two operands.")

    operations = {
        ast.Eq: "eq",
        ast.NotEq: "ne",
        ast.Lt: "lt",
        ast.LtE: "le",
        ast.Gt: "gt",
        ast.GtE: "ge",
    }

    try:
        operation = operations[type(node.ops[0])]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported index comparison: {ast.dump(node.ops[0])}."
        ) from exc
    return IndexExpr(
        op=operation,
        operands=(
            _index_expr_from_ast(node.left),
            _index_expr_from_ast(node.comparators[0]),
        ),
    )


def _index_call(node: ast.Call) -> IndexExpr:
    return IndexExpr(
        op="call",
        value=ast.unparse(node.func),
        operands=tuple(_index_expr_from_ast(arg) for arg in node.args),
    )


def _index_subscript(node: ast.Subscript) -> IndexExpr:
    return IndexExpr(
        op="subscript",
        operands=(
            _index_expr_from_ast(node.value),
            _index_expr_from_ast(node.slice),
        ),
    )


def _index_sequence(node: ast.Tuple | ast.List) -> IndexExpr:
    return IndexExpr(
        op="tuple",
        operands=tuple(_index_expr_from_ast(item) for item in node.elts),
    )


_INDEX_AST_HANDLERS = {
    ast.Constant: _index_constant,
    ast.Name: _index_name,
    ast.Attribute: _index_attribute,
    ast.BinOp: _index_binary,
    ast.UnaryOp: _index_unary,
    ast.BoolOp: _index_bool,
    ast.Compare: _index_compare,
    ast.Call: _index_call,
    ast.Subscript: _index_subscript,
    ast.Tuple: _index_sequence,
    ast.List: _index_sequence,
}


__all__ = [
    "AccessMap",
    "IndexExpr",
    "Kernel",
    "LayoutLevel",
    "LaunchABI",
    "LaunchBinding",
    "LaunchPlan",
    "TensorLayout",
    "TensorSpec",
    "ir_to_dict",
]
