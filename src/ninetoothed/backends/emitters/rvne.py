"""Scalar C++ syntax for RISC-V code compiled with the RVNE toolchain."""

import ast
import math
from dataclasses import dataclass

from ninetoothed.backends.core import Target
from ninetoothed.backends.emitters import ssa as common
from ninetoothed.backends.emitters.base import EmitterTarget, ModuleRenderContext
from ninetoothed.ir import Kernel

_TYPES = {
    "bool": "bool",
    "index": "int64_t",
    "int32": "int32_t",
    "uint32": "uint32_t",
    "int64": "int64_t",
    "uint64": "uint64_t",
    "float32": "float",
}


@dataclass(frozen=True, kw_only=True)
class RvneTarget(EmitterTarget):
    """Emit a serial host ABI without dependencies on a GPU runtime."""

    backend: Target = Target.RVNE
    language: str = "c++"
    suffix: str = "cpp"
    source_route: str = "ssa-unified-rvne-emitter"
    c_style_syntax: bool = True

    def index_cast(self, value):
        return f"int64_t({value})"

    def render_index_expr(self, expression):
        return _index_expr(expression)

    def assign_scalar(self, name, value, *, mutable):
        del mutable

        return f"{name} = {value};"

    def literal(self, value):
        if isinstance(value, bool):
            return "true" if value else "false"

        if value in ("inf", "-inf"):
            return "INFINITY" if value == "inf" else "-INFINITY"

        if isinstance(value, float):
            if math.isnan(value):
                return "NAN"

            if math.isinf(value):
                return "INFINITY" if value > 0 else "-INFINITY"

            return f"{value!r}f"

        if isinstance(value, int):
            if value == -(1 << 63):
                return "(-9223372036854775807LL - 1LL)"

            if (1 << 63) <= value < (1 << 64):
                return f"{value}ULL"

            if not -(1 << 63) <= value < (1 << 64):
                raise ValueError("RVNE integer literals must fit int64 or uint64.")

            return str(value) if -(1 << 31) < value < (1 << 31) else f"{value}LL"

        raise ValueError(f"Unsupported RVNE literal: {value!r}.")

    def load(self, tensor, index, *, mask=None, other=0.0):
        del mask, other

        return f"{self.tensor_ref(tensor)}[{index}]"

    def store(self, tensor, index, value, *, mask=None):
        assignment = f"{self.tensor_ref(tensor)}[{index}] = {value};"

        return assignment if mask is None else f"if ({mask}) {{\n    {assignment}\n}}"

    def cast(self, dtype, value):
        return f"static_cast<{self.type_name(dtype)}>({value})"

    def type_name(self, dtype, kind=None):
        if kind == "index":
            return "int64_t"

        dtype = common.normalize_dtype(dtype)

        if dtype not in _TYPES:
            raise ValueError(f"Unsupported RVNE SSA dtype: {dtype!r}.")

        return _TYPES[dtype] + ("*" if kind == "pointer" else "")

    def where(self, cond, yes, no):
        return f"(({cond}) ? ({yes}) : ({no}))"

    def call(self, name, args):
        if name == "where" and len(args) == 3:
            return self.where(*args)

        if name == "load" and len(args) == 1:
            return f"*({args[0]})"

        if name in {"dot", "block_dot"} and len(args) == 2:
            return f"(({args[0]}) * ({args[1]}))"

        if name == "spike_accumulate" and len(args) == 4:
            return f"nt_spike_accumulate({', '.join(args)})"

        if name in {"maximum", "max", "minimum", "min"} and len(args) == 2:
            function = "nt_max" if name in {"maximum", "max"} else "nt_min"

            return f"{function}({args[0]}, {args[1]})"

        raise ValueError(f"Unsupported RVNE function: {name!r}.")

    def local_decl(self, type_, name, expr):
        if type_.kind == "pointer":
            return f"auto {name} = {expr};"

        return f"{self.type_name(type_.dtype, type_.kind)} {name} = {expr};"

    def loop_header(self, var, lower, upper, step):
        try:
            stride = int(step)
        except ValueError:
            raise ValueError("RVNE loops require a constant nonzero step.") from None

        if stride == 0:
            raise ValueError("RVNE loops require a constant nonzero step.")

        comparison = "<" if stride > 0 else ">"

        return f"for (int64_t {var} = {lower}; {var} {comparison} {upper}; {var} += {step}) {{"

    def reduce_update(self, operator, acc, term):
        if operator == "sum":
            return f"({acc}) + ({term})"

        if operator in {"max", "min"}:
            return f"nt_{operator}({acc}, {term})"

        raise ValueError(f"Unsupported RVNE reduction: {operator!r}.")

    def coerce_binary_args(self, operation, args, context):
        result_type = operation.results[0].type

        if result_type.kind == "pointer":
            return args

        operand_types = [context.value_types.get(name) for name in operation.operands]

        if any(
            type_ is not None and type_.kind == "pointer" for type_ in operand_types
        ):
            return args

        dtype = common.normalize_dtype(result_type.dtype)

        if operation.opcode.startswith("cmp."):
            dtypes = {
                common.normalize_dtype(type_.dtype)
                for type_ in operand_types
                if type_ is not None
            }

            if "float32" in dtypes:
                dtype = "float32"
            elif "uint64" in dtypes:
                if dtypes - {"uint64", "bool"}:
                    raise ValueError(
                        "RVNE uint64 comparisons require matching operand dtypes."
                    )

                dtype = "uint64"
            elif (
                "int64" in dtypes or "index" in dtypes or {"int32", "uint32"} <= dtypes
            ):
                dtype = "int64"
            elif "uint32" in dtypes:
                dtype = "uint32"
            elif "int32" in dtypes:
                dtype = "int32"

        return tuple(self.cast(dtype, arg) for arg in args)

    def arithmetic_expr(self, operation, args, context):
        opcode = operation.opcode
        dtype = common.normalize_dtype(operation.results[0].type.dtype)

        if opcode == "arith.invert" and dtype == "bool":
            return f"(!({args[0]}))"

        if opcode in {"arith.maximum", "arith.max", "arith.minimum", "arith.min"}:
            return self.call(
                opcode.split(".", 1)[1], tuple(self.cast(dtype, arg) for arg in args)
            )

        if opcode not in {"arith.bitwise_left_shift", "arith.bitwise_right_shift"}:
            return None

        dtype = common.normalize_dtype(context.value_types[operation.operands[0]].dtype)
        bits = {"int32": 32, "uint32": 32, "int64": 64, "uint64": 64, "index": 64}.get(
            dtype
        )
        count_op = context.operations.get(operation.operands[1])
        count = (
            count_op.attrs.get("value")
            if count_op is not None and count_op.opcode == "arith.constant"
            else None
        )

        if bits is None or type(count) is not int or not 0 <= count < bits:
            raise ValueError(
                "RVNE shifts require an integer operand and a constant count within its bit width."
            )

        value = self.cast(dtype, args[0])

        if opcode == "arith.bitwise_left_shift":
            unsigned = "uint32" if bits == 32 else "uint64"

            return self.cast(dtype, f"({self.cast(unsigned, value)} << {count})")

        return f"nt_shift_right({value}, {count})"

    def render_module(self, context: ModuleRenderContext):
        if context.block_program or context.cooperative_reduction_program:
            raise ValueError("RVNE requires a serial scalar program.")

        params = []

        for name in (*context.variables, *context.outputs):
            info = context.tensors[name]
            type_name = self.type_name(info.dtype)

            if info.ndim:
                type_name = (
                    ("const " if name in context.variables else "") + type_name + "*"
                )
            elif name in context.outputs:
                raise ValueError(
                    "RVNE output tensors must have at least one dimension."
                )

            params.append(f"{type_name} {name}")

        for name in context.shape_params:
            info = context.tensors.get(name)
            type_name = (
                self.type_name(info.dtype) if info and info.ndim == 0 else "int64_t"
            )
            params.append(f"{type_name} {name}")

        signature = ",\n".join(f"    {param}" for param in params)
        total = _index_expr(context.total)
        body = common.indent_block(context.body, "        ")
        spike_helper = ""

        if any(
            op.opcode == "call.spike_accumulate" for op in context.operations.values()
        ):
            spike_helper = _SPIKE_ACCUMULATE_HELPER

        return f"""// Generated by NineToothed's RVNE SSA backend.
// Compile signed integer arithmetic with -fwrapv.
#include <stdint.h>
#include <math.h>

static inline int64_t nt_floor_div(int64_t a, int64_t b) {{
    int64_t q = a / b;
    int64_t r = a % b;
    return q - ((r != 0) && ((r < 0) != (b < 0)));
}}

static inline int64_t nt_floor_mod(int64_t a, int64_t b) {{
    int64_t r = a % b;
    return r + (((r != 0) && ((r < 0) != (b < 0))) ? b : 0);
}}

static inline int32_t nt_shift_right(int32_t value, int count) {{
    return value >= 0 ? (value >> count) : -1 - ((-1 - value) >> count);
}}

static inline int64_t nt_shift_right(int64_t value, int count) {{
    return value >= 0 ? (value >> count) : -1 - ((-1 - value) >> count);
}}

static inline uint32_t nt_shift_right(uint32_t value, int count) {{
    return value >> count;
}}

static inline uint64_t nt_shift_right(uint64_t value, int count) {{
    return value >> count;
}}

template <typename A, typename B>
static inline auto nt_max(A a, B b) -> decltype(a + b) {{
    return a > b ? a : b;
}}

template <typename A, typename B>
static inline auto nt_min(A a, B b) -> decltype(a + b) {{
    return a < b ? a : b;
}}

static inline float nt_max(float a, float b) {{
    return isnan(a) ? a : (isnan(b) ? b : fmaxf(a, b));
}}

static inline float nt_min(float a, float b) {{
    return isnan(a) ? a : (isnan(b) ? b : fminf(a, b));
}}

{spike_helper}
extern "C" int {self.entrypoint(context.kernel.kernel_name)}(
{signature}
) {{
    for (int64_t {self.index_name} = 0; {self.index_name} < ({total}); ++{self.index_name}) {{
{body}
    }}
    return 0;
}}
"""


def _index_expr(expression):
    try:
        node = ast.parse(expression, mode="eval").body
    except SyntaxError as error:
        raise ValueError(
            f"Unsupported RVNE index expression: {expression!r}."
        ) from error

    return _render_index(node)


def _render_index(node):
    if isinstance(node, ast.Name):
        return {"true": "true", "false": "false"}.get(node.id, node.id)

    if isinstance(node, ast.Constant) and isinstance(node.value, (int, bool)):
        return TARGET.literal(node.value)

    if isinstance(node, ast.UnaryOp) and isinstance(
        node.op, (ast.UAdd, ast.USub, ast.Not, ast.Invert)
    ):
        operator = {ast.UAdd: "+", ast.USub: "-", ast.Not: "!", ast.Invert: "~"}[
            type(node.op)
        ]

        return f"({operator}{_render_index(node.operand)})"

    if isinstance(node, ast.BinOp):
        lhs, rhs = _render_index(node.left), _render_index(node.right)

        if isinstance(node.op, ast.FloorDiv):
            return f"nt_floor_div({lhs}, {rhs})"

        if isinstance(node.op, ast.Mod):
            return f"nt_floor_mod({lhs}, {rhs})"

        operators = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.BitAnd: "&",
            ast.BitOr: "|",
            ast.LShift: "<<",
            ast.RShift: ">>",
        }

        if type(node.op) in operators:
            return f"({lhs} {operators[type(node.op)]} {rhs})"

    if isinstance(node, ast.Compare):
        operators = {
            ast.Eq: "==",
            ast.NotEq: "!=",
            ast.Lt: "<",
            ast.LtE: "<=",
            ast.Gt: ">",
            ast.GtE: ">=",
        }
        values = [node.left, *node.comparators]
        comparisons = [
            f"({_render_index(lhs)} {operators[type(op)]} {_render_index(rhs)})"
            for lhs, op, rhs in zip(values, node.ops, values[1:])
        ]

        return "(" + " && ".join(comparisons) + ")"

    if isinstance(node, ast.BoolOp):
        operator = " && " if isinstance(node.op, ast.And) else " || "

        return "(" + operator.join(_render_index(value) for value in node.values) + ")"

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id

        if name == "int64_t" and len(node.args) == 1:
            return f"int64_t({_render_index(node.args[0])})"

        if name in {"nt_floor_div", "nt_floor_mod", "Mod"} and len(node.args) == 2:
            function = "nt_floor_mod" if name == "Mod" else name

            return f"{function}({', '.join(_render_index(arg) for arg in node.args)})"

        if name in {"floor", "ceil", "ceiling"} and len(node.args) == 1:
            arg = node.args[0]

            if isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Div):
                lhs, rhs = _render_index(arg.left), _render_index(arg.right)

                return (
                    f"nt_floor_div({lhs}, {rhs})"
                    if name == "floor"
                    else f"(-nt_floor_div(-({lhs}), {rhs}))"
                )

            return _render_index(arg)

    raise ValueError(f"Unsupported RVNE index expression node: {ast.dump(node)}.")


_SPIKE_ACCUMULATE_HELPER = """static inline int32_t nt_spike_accumulate(
    int32_t current, uint32_t spike, uint64_t weight_low, uint64_t weight_high
) {
    __builtin_riscv_set_ncr_32(current, 0);
    __builtin_riscv_set_svr_32(spike, 0);
    __builtin_riscv_set_wvr_64(weight_low, 0);
    __builtin_riscv_set_wvr_64(weight_high, 1);
    __builtin_riscv_calc_acc_32(0, 0, 0);
    return static_cast<int32_t>(
        static_cast<uint32_t>(__builtin_riscv_get_ncr_64(0))
    );
}
"""


TARGET = RvneTarget()


def emit(kernel: Kernel):
    from ninetoothed.backends.rvne import validate_kernel

    validate_kernel(kernel)

    return common.emit(kernel, TARGET)


__all__ = ["RvneTarget", "TARGET", "emit"]
