"""RVNE source generation and legality for the RISC-V Zne toolchain."""

from typing import TYPE_CHECKING, Any, Mapping

from ninetoothed.backends.core import Artifact, Backend, Capability, Target
from ninetoothed.compiler.passes import OptimizeSchedule
from ninetoothed.dtype import normalize_dtype
from ninetoothed.ir import Kernel, ssa

if TYPE_CHECKING:
    from ninetoothed.compiler.passes import Registry


_SUPPORTED_DTYPES = frozenset({"bool", "int32", "uint32", "int64", "uint64", "float32"})
_SUPPORTED_OPCODES = frozenset(
    {
        "arith.constant",
        "arith.add",
        "arith.sub",
        "arith.subtract",
        "arith.mul",
        "arith.multiply",
        "arith.div",
        "arith.truediv",
        "arith.and",
        "arith.or",
        "arith.bitwise_and",
        "arith.bitwise_or",
        "arith.bitwise_xor",
        "arith.bitwise_left_shift",
        "arith.bitwise_right_shift",
        "arith.neg",
        "arith.pos",
        "arith.not",
        "arith.invert",
        "arith.maximum",
        "arith.max",
        "arith.minimum",
        "arith.min",
        "cmp.eq",
        "cmp.ne",
        "cmp.lt",
        "cmp.le",
        "cmp.gt",
        "cmp.ge",
        "shape.dim",
        "tensor.stride",
        "tensor.view",
        "tensor.zeros",
        "tensor.empty",
        "tensor.full",
        "tensor.extract",
        "tensor.cast",
        "select.where",
        "index.offset",
        "symbol.attr",
        "mem.store",
        "mem.load",
        "mem.data_ptr",
        "reduce.sum",
        "reduce.max",
        "reduce.min",
        "scf.for",
        "scf.if",
        "scf.yield",
        "linalg.transpose",
        "linalg.dot",
        "linalg.matmul",
        "math.exp2",
        "call.spike_accumulate",
    }
)


class RvneBackend(Backend):
    """Emit scalar RISC-V C++ without implicit spike or weight quantization."""

    name = Target.RVNE
    supported_options = frozenset({"toolchain_root"})
    capability = Capability(
        name=name,
        emits_source=True,
        can_execute=True,
        requires_external_compiler=True,
        notes=(
            "Emits RISC-V C++ for the RVNE Zne toolchain.",
            "AOT execution uses contiguous NumPy host arrays through QEMU.",
            "JIT execution and direct hardware execution are not implemented.",
        ),
    )

    def normalize_options(self, options: Mapping[str, Any]) -> Mapping[str, Any]:
        normalized = dict(super().normalize_options(options))

        if "toolchain_root" in normalized:
            root = normalized["toolchain_root"]

            if not isinstance(root, str) or not root.strip():
                raise TypeError("RVNE `toolchain_root` must be a non-empty string.")

        return normalized

    def prepare_for_emission(self, kernel: Kernel) -> Kernel:
        from ninetoothed.compiler.specialization import specialize_schedule_tiles

        kernel = specialize_schedule_tiles(kernel)
        validate_kernel(kernel)

        return kernel

    def emit(self, kernel: Kernel) -> Artifact:
        from ninetoothed.backends.emitters.rvne import emit

        return emit(kernel)


class RvneOptimizeSchedule(OptimizeSchedule):
    """Keep the initial RVNE schedule scalar and decompose dense linear algebra."""

    name = "ssa.rvne.optimize_schedule"
    supported_backends = (Target.RVNE,)

    def optimization_policy(
        self,
        backend: Target,
        analysis: Mapping[str, Any],
        schedule: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del backend, analysis, schedule

        return {"preserve_linalg": False}


def validate_kernel(kernel: Kernel) -> None:
    """Reject unsupported operations and dtypes before emitting target source."""
    if kernel.ssa is None:
        raise ValueError("RVNE backend emission requires ssa.Program.")

    for tensor in kernel.tensors:
        _validate_dtype(tensor.dtype, f"tensor `{tensor.name}`")

        if tensor.jagged_dim is not None:
            raise ValueError("RVNE does not support jagged tensor arguments.")

    for value in (*kernel.ssa.inputs, *kernel.ssa.outputs):
        _validate_value(value)

    for block in kernel.ssa.blocks:
        _validate_block(block)

    _validate_spike_accumulators(kernel.ssa)
    _validate_float_math(kernel.ssa)


def _validate_float_math(program: ssa.Program) -> None:
    from ninetoothed.backends.emitters.analysis import program_value_types, walk_ops

    value_types = program_value_types(program)

    for block in program.blocks:
        for operation in walk_ops(block.operations):
            if operation.opcode not in {"math.exp2", "arith.div", "arith.truediv"}:
                continue

            arity = 1 if operation.opcode == "math.exp2" else 2

            if len(operation.operands) != arity or len(operation.results) != 1:
                raise ValueError(
                    f"Malformed RVNE floating operation `{operation.opcode}`."
                )

            types = tuple(value_types[name] for name in operation.operands)
            result = operation.results[0].type
            dtypes = tuple(normalize_dtype(type_.dtype) for type_ in types)
            valid = (
                result.kind in {"scalar", "tensor"}
                and normalize_dtype(result.dtype) == "float32"
                and "float32" in dtypes
            )

            for type_, dtype in zip(types, dtypes):
                valid = valid and (
                    (type_.kind in {"scalar", "tensor"} and dtype == "float32")
                    or (
                        operation.opcode != "math.exp2"
                        and type_.kind in {"scalar", "index"}
                        and dtype in {"index", "int32", "uint32", "int64", "uint64"}
                    )
                )

            if not valid:
                raise TypeError(
                    f"RVNE `{operation.opcode}` requires FP32 operands and result; "
                    "floating division also accepts an integer scalar operand."
                )


def _validate_spike_accumulators(program: ssa.Program) -> None:
    from ninetoothed.backends.emitters.analysis import program_value_types, walk_ops

    value_types = program_value_types(program)
    expected = ("int32", "uint32", "uint64", "uint64")

    for block in program.blocks:
        for operation in walk_ops(block.operations):
            if operation.opcode != "call.spike_accumulate":
                continue

            if len(operation.operands) != 4 or len(operation.results) != 1:
                raise ValueError("RVNE spike_accumulate requires four operands.")

            types = tuple(value_types[name] for name in operation.operands)

            if any(type_.kind not in {"scalar", "tensor"} for type_ in types):
                raise TypeError(
                    "RVNE spike_accumulate requires scalar or tensor values."
                )

            if tuple(normalize_dtype(type_.dtype) for type_ in types) != expected:
                raise TypeError(
                    "RVNE spike_accumulate requires int32 current, uint32 spikes, "
                    "and two uint64 packed weight words."
                )

            result = operation.results[0].type

            if (
                normalize_dtype(result.dtype) != "int32"
                or result.shape != types[0].shape
            ):
                raise TypeError(
                    "RVNE spike_accumulate must return the current's int32 shape."
                )

            if any(not _broadcasts_to(type_.shape, result.shape) for type_ in types):
                raise ValueError(
                    "RVNE spike_accumulate operands must broadcast to the current shape."
                )


def _broadcasts_to(shape, target_shape):
    if len(shape) > len(target_shape):
        return False

    return all(
        str(size) == "1" or str(size) == str(target)
        for size, target in zip(reversed(shape), reversed(target_shape))
    )


def _validate_block(block: ssa.Block) -> None:
    for value in block.args:
        _validate_value(value)

    for operation in block.operations:
        if operation.opcode not in _SUPPORTED_OPCODES:
            raise ValueError(
                f"RVNE does not support SSA operation `{operation.opcode}`."
            )

        for value in operation.results:
            _validate_value(value)

        for region in operation.regions:
            _validate_block(region)


def _validate_value(value: ssa.Value) -> None:
    if value.type.kind in {"index", "scalar", "tensor"} and value.type.dtype == "index":
        return

    if value.type.dtype is not None:
        _validate_dtype(value.type.dtype, f"SSA value `{value.name}`")


def _validate_dtype(dtype: str | None, location: str) -> None:
    if normalize_dtype(dtype) not in _SUPPORTED_DTYPES:
        supported = ", ".join(sorted(_SUPPORTED_DTYPES))

        raise TypeError(
            f"RVNE does not support dtype `{dtype}` for {location}. "
            f"Supported dtypes: {supported}."
        )


def register_ssa_passes(registry: "Registry") -> None:
    from ninetoothed.backends.registry import register_pass_bundle

    register_pass_bundle(
        registry,
        backend=Target.RVNE,
        optimize_schedule=RvneOptimizeSchedule,
    )
