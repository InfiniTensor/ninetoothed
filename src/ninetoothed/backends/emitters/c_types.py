"""Computed C++ scalar types shared by native emitter strategies."""

from ninetoothed.ir import ssa


class CComputedTypesMixin:
    """Keep computed dtype witnesses and declarations in the same C++ type."""

    def load(self, tensor, index, *, mask=None, other=0.0):
        del mask, other
        rendered = f"({index})" if index.startswith("nt_idx_") else index

        return f"{self.tensor_ref(tensor)}[{rendered}]"

    def store(self, tensor, index, value, *, mask=None):
        rendered = f"({index})" if index.startswith("nt_idx_") else index
        assignment = f"{self.tensor_ref(tensor)}[{rendered}] = {value};"

        return assignment if mask is None else f"if ({mask}) {{\n    {assignment}\n}}"

    def value_dtype(self, value):
        return f"std::remove_cv<std::remove_reference<decltype({value})>::type>::type"

    def dtype_equal(self, lhs, rhs):
        return f"std::is_same<{self.value_dtype(lhs)}, {self.value_dtype(rhs)}>::value"

    def select_dtype_sample(self, condition, yes, no):
        dtype = (
            f"std::conditional<({condition}), {self.value_dtype(yes)}, "
            f"{self.value_dtype(no)}>::type"
        )

        return f"static_cast<{dtype}>(1)"

    def local_decl(self, type_: ssa.Type, name: str, expr: str) -> str:
        if type_.kind == "pointer" or (
            type_.dtype is None and type_.kind in {"scalar", "tensor"}
        ):
            return f"auto {name} = {expr};"
        return f"{self.type_name(type_.dtype, type_.kind)} {name} = {expr};"

    def check_numeric_result(self, operation, value, context):
        declared = operation.results[0].type if operation.results else None

        if (
            declared is not None
            and operation.opcode.startswith("arith.")
            and declared.dtype in {"float16", "bfloat16"}
        ):
            return self.cast(declared.dtype, value)
        return value

    def global_helper(self, name: str) -> str:
        """Resolve generated helpers in the global C++ namespace."""
        return f"::{name}"
