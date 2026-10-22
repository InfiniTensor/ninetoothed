"""Prove AscendC SDK eligibility, packet lifetimes and physical storage budgets."""

import hashlib
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

from ninetoothed.backends.toolchain import (
    _ascend_home_for,
    _ccec_identity,
    _ccec_include_dirs,
    find_ccec,
    normalize_ascendc_arch,
    resolve_ascendc_arch,
)
from ninetoothed.compiler.passes import _merge_pass_options, _pass_options
from ninetoothed.compiler.vector_resources import Allocation, plan_vector_chunks

# CANN reports 192 KiB minus its 256-byte reserved UB region.
ASCENDC_UB_BYTES = 192 * 1024 - 256

NATIVE_DIVISION_CANDIDATE = "vector-native-division"
_DIVISION_PASS = "ssa.ascendc.optimize_schedule"
_DIVISION_STRATEGIES = frozenset({"refined_reciprocal", "native_division"})
# These exact CANN 8.2.RC1 C220 bytes produced the 216 primitive and 180
# generated strict controls in ascend-reference-first-20261014 v6/v7. This
# fingerprint is a conservative measured scope, not a universal Div theorem.
_DIVISION_HEADER_SHA256 = (
    "f79d6d19c45d86c607077261a353dd919d59f111e543338d10edd1a7b9a726b5"
)
_DIVISION_COMPILER_SHA256 = (
    "b23ce03ed55769b35cf0c4c4a210f79458d030ccf176460eaa3f49f5af2c6853"
)


def division_strategy_options(pass_options):
    """Use the master schedule precedence and one explicit Div candidate."""
    for name in ("*", _DIVISION_PASS, "ssa.optimize_schedule"):
        options = pass_options.get(name, {})

        if not isinstance(options, Mapping):
            raise ValueError(f"Options for SSA pass `{name}` must be a mapping.")

        if {"strict_division_strategy", "strict_validity_summary"} & options.keys():
            raise ValueError(
                "Removed AscendC policy switches; select the vector-native-division "
                "candidate explicitly. Validity folding is automatic."
            )

    options = _pass_options(
        SimpleNamespace(pass_options=pass_options),
        _DIVISION_PASS,
        "ssa.optimize_schedule",
    )

    if (
        options.get("candidate") == NATIVE_DIVISION_CANDIDATE
        and options.get("enabled", True) is False
    ):
        raise ValueError(
            "The native division strategy was not consumed by the target schedule pass."
        )
    return options


def requested_vector_chunk(context):
    """Read the sole target packet geometry option."""
    return context.kernel.compiler_options.get("backend_options", {}).get(
        "vector_chunk"
    )


def _verified_arch(context):
    options = context.kernel.compiler_options.get("backend_options", {})
    arch = normalize_ascendc_arch(options.get("arch", "native"))

    if arch not in {"native", "dav-c220"} or resolve_ascendc_arch(arch) != "dav-c220":
        raise ValueError("The AscendC vector resource override requires dav-c220.")


@lru_cache(maxsize=8)
def _compiler_fingerprint(path, size, mtime_ns):
    del size, mtime_ns

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_division_platform_evidence():
    """Bind the measured implementation to the actual build compiler/includes."""
    try:
        compiler = Path(find_ccec()).resolve(strict=True)
        home = _ascend_home_for(str(compiler)).resolve(strict=True)
        header = (
            home
            / "compiler/tikcpp/tikcfw/impl/dav_c220/kernel_operator_vec_binary_impl.h"
        )
        digest = hashlib.sha256(header.read_bytes()).hexdigest()
        stat = compiler.stat()
    except (OSError, RuntimeError) as error:
        raise ValueError(
            "Native division requires the validated CANN SDK header."
        ) from error

    if digest != _DIVISION_HEADER_SHA256:
        raise ValueError("Native division has no precision proof for this SDK header.")

    compiler_digest = _compiler_fingerprint(
        str(compiler), stat.st_size, stat.st_mtime_ns
    )

    if compiler_digest != _DIVISION_COMPILER_SHA256:
        raise ValueError(
            "Native division has no precision proof for this CCEC compiler."
        )
    return {
        "codegen_arch": "dav-c220",
        "SDK_header": str(header),
        "SDK_header_sha256": digest,
        "compiler": _ccec_identity(str(compiler), stat.st_size, stat.st_mtime_ns),
        "compiler_sha256": compiler_digest,
        "SDK_home": str(home),
        "include_dirs": _ccec_include_dirs(home),
        "precision_scope": "CANN 8.2.RC1 C220 Div<float>; strict CPU oracle, IEEE classes and signed zero",
        "physical_device": "not queried by source emission",
    }


def requested_division_strategy(context):
    kernel = context.kernel
    raw_options = _merge_pass_options(
        kernel.compiler_options.get("ssa_pass_options", {}),
        getattr(kernel, "metadata", {}).get("ssa_pass_options", {}),
    )
    intent = division_strategy_options(raw_options)
    metadata = kernel.ssa.metadata
    value = metadata.get("optimization", {}).get(
        "strict_division_strategy", "refined_reciprocal"
    )
    selected = metadata.get("selected_schedule_candidate") == NATIVE_DIVISION_CANDIDATE
    requested = intent.get("candidate") == NATIVE_DIVISION_CANDIDATE

    if type(value) is not str or value not in _DIVISION_STRATEGIES:
        raise ValueError("The AscendC strict division strategy is invalid.")

    if requested and (not selected or value != "native_division"):
        raise ValueError(
            "The native division strategy was not consumed by the target schedule pass."
        )

    if selected and value != "native_division":
        raise ValueError(
            "The native division candidate conflicts with the emitted division strategy."
        )

    if value == "native_division" and not selected:
        raise ValueError(
            "Native division requires the selected vector-native-division schedule candidate."
        )
    return value


def configure_basic_division_strategy(context, plan):
    """Preserve every input through sparse repair using the real allocation map."""
    if requested_division_strategy(context) != "native_division":
        return

    _verified_arch(context)
    evidence = native_division_platform_evidence()
    proofs = plan.native_division_proofs()

    if not proofs or not all(proof["operands_preserved"] for proof in proofs):
        context.scheduled_metadata["ascendc_division_strategy"] = {
            "requested": "native_division",
            "consumed": False,
            "route": "basic-vector",
            "reason": "unproved float32 vector operand lifetimes",
            "operand_lifetimes": proofs,
        }
        raise ValueError(
            "Native division requires proved float32 vector operand lifetimes."
        )

    plan.native_division = True
    context.scheduled_metadata["ascendc_division_strategy"] = {
        "requested": "native_division",
        "consumed": True,
        "route": "basic-vector",
        "platform": evidence,
        "operand_lifetimes": proofs,
        "extra_ub_bytes": 0,
        "IEEE_repair": "original packed validity, sparse repair and scalar tail",
        "scratch": "independent nt_ref0; Duplicate never overwrites a live operand",
    }


def configure_dag_division_strategy(context, plan):
    """Shared DAG physical slots never alias a live vector operand."""
    if requested_division_strategy(context) != "native_division":
        return False

    _verified_arch(context)
    evidence = native_division_platform_evidence()
    proofs = []

    for step in plan.steps:
        node = step.node
        opcode = node.operation.opcode

        if (
            node.result.scalar
            or node.result.dtype != "float32"
            or opcode not in {"arith.div", "math.reciprocal"}
        ):
            continue

        if opcode == "arith.div" and node.operands[1].scalar:
            continue

        operands = tuple(
            buffer
            for value, buffer in zip(node.operands, step.operand_buffers)
            if not value.scalar
        )
        preserved = (
            node.lowering.mode != "scalar"
            and "float32" in node.lowering.scratch_dtypes
            and node.lowering.scratch_dtypes.count("uint8") >= 2
            and step.destination is not None
            and plan.buffer_dtypes[step.destination] == "float32"
            and all(
                buffer is not None
                and buffer != step.destination
                and plan.buffer_dtypes[buffer] == "float32"
                for buffer in operands
            )
        )
        proofs.append(
            {
                "operation": opcode,
                "destination_slot": step.destination,
                "operand_slots": operands,
                "operands_preserved": preserved,
                "source_positions": "VECCALC",
                "scratch_position": "separate VECCALC",
                "lifetime": "through every original sparse repair",
            }
        )

    if not proofs or not all(p["operands_preserved"] for p in proofs):
        context.scheduled_metadata["ascendc_division_strategy"] = {
            "requested": "native_division",
            "consumed": False,
            "route": "typed-row-dag",
            "reason": "unproved float32 vector operand lifetimes",
            "operand_lifetimes": tuple(proofs),
        }
        raise ValueError(
            "Native division requires proved float32 vector operand lifetimes."
        )

    context.scheduled_metadata["ascendc_division_strategy"] = {
        "requested": "native_division",
        "consumed": True,
        "route": "typed-row-dag",
        "platform": evidence,
        "operand_lifetimes": tuple(proofs),
        "extra_ub_bytes": 0,
        "IEEE_repair": "original packed validity, sparse repair and scalar tail",
        "scratch": "independent nt_ref0; Duplicate never overwrites a live operand",
    }

    return True


def configure_validity_summary(context, *, route, maximum_elements, strict_scratch):
    """Fold existing packed masks only after their float scratch is dead.

    The strict renderer's comparisons complete before this fold, and its
    original masks stay live for exceptional repair. Its float scratch prefix
    is then dead: the integer workspace uses at most prefix / 4 bytes, while
    the scalar float tail starts at prefix * 4 bytes. No UB allocation changes.
    """
    policy = context.kernel.ssa.metadata.get("optimization", {})
    enabled = policy.get("strict_validity_summary", False)
    consumed = False
    reason = "policy-disabled-or-absent"

    if type(enabled) is not bool:
        raise ValueError("The AscendC strict validity summary option must be boolean.")

    if enabled:
        if not strict_scratch:
            reason = "no-strict-vector-repair"
        elif type(maximum_elements) is not int or maximum_elements < 2048:
            reason = "packet-cannot-reach-summary-threshold"
        else:
            try:
                _verified_arch(context)
            except (TypeError, ValueError, OSError):
                reason = "unproved-platform"
            else:
                consumed = True
                reason = "dead-float-prefix-and-preserved-packed-masks"

    context.scheduled_metadata["ascendc_validity_summary"] = {
        "enabled": enabled,
        "consumed": consumed,
        "route": route,
        "proof": reason,
        "minimum_prefix": 2048,
        "maximum_elements": maximum_elements,
        "workspace": "dead-nt_ref0-prefix",
        "extra_ub_bytes": 0,
        "integer_dtype": "int16",
        "scalar_summary_words": 4,
        "exceptional_repair": "original-packed-scan-and-scalar-tail",
    }

    return consumed


def configure_basic_validity_summary(context, plan):
    """Keep small and unproved basic packets on their exact original source."""
    maximum = (
        min(plan.chunk, plan.static_extent) if type(plan.static_extent) is int else None
    )
    plan.validity_summary = configure_validity_summary(
        context,
        route="basic-vector",
        maximum_elements=maximum,
        strict_scratch=plan.float_scratch and plan.strict_scratch,
    )


def configure_serial_queue_storage(context, plan):
    """Consume the policy only for the basic renderer's single live packet."""
    enabled = True
    reason = "policy-disabled-or-absent"

    if enabled:
        if (
            not plan.output_dtype_known
            or not plan.input_dtypes
            or any(dtype != "float32" for dtype in plan.input_dtypes)
        ):
            reason = "unproved-queue-dtypes"
        else:
            try:
                _verified_arch(context)
            except (TypeError, ValueError, OSError):
                reason = "unproved-platform"
            else:
                # The basic body allocates each input once, consumes that
                # tensor, and frees it before returning. It has no packet
                # loop, lookahead, or second AllocTensor on the queue.
                plan.input_queue_slots = 1
                reason = "one-allocation-per-input-in-one-serial-packet"

    chunk_metadata = context.scheduled_metadata.get("ascendc_vector_chunk")

    if chunk_metadata is not None:
        chunk_metadata.update(
            native_chunk=plan.chunk,
            ub_bytes=plan.ub_bytes(plan.chunk),
        )

    context.scheduled_metadata["ascendc_serial_queue_storage"] = {
        "enabled": enabled,
        "consumed": plan.input_queue_slots == 1,
        "route": "basic-vector",
        "original_input_queue_slots": 2,
        "input_queue_slots": plan.input_queue_slots,
        "proof": reason,
    }


def configure_basic_vector(context, plan):
    """Recycle dead float temporaries without aliasing a live operand."""
    requested = requested_vector_chunk(context)

    if requested is None:
        return None

    _verified_arch(context)

    if (
        not plan.output_dtype_known
        or not plan.input_dtypes
        or any(dtype != "float32" for dtype in plan.input_dtypes)
    ):
        raise ValueError("The AscendC vector resource override requires proved dtypes.")

    temporaries = plan.live_temporary_buffers()
    slots = max((slot for _, slot in temporaries), default=-1) + 1
    allocations = [
        *(
            Allocation(f"input-{index}", 4, slots=plan.input_queue_slots)
            for index in range(len(plan.inputs))
        ),
        Allocation("output", 2 if plan.output_dtype == "float16" else 4),
        *(Allocation(f"temporary-{index}", 4) for index in range(slots)),
    ]

    if plan.float_scratch:
        allocations.append(Allocation("refinement", 4))

    if plan.strict_scratch:
        allocations.extend(
            Allocation(f"mask-{index}", 1, divisor=8) for index in range(2)
        )

    if plan.extrema_scratch:
        allocations.append(Allocation("mask-extrema", 1, divisor=8))

    resources = plan_vector_chunks(
        allocations,
        budget_bytes=ASCENDC_UB_BYTES,
        candidates=(2048, 4096, 8192, 16256),
        default_chunk=plan.fallback_chunk,
        requested_chunk=requested,
        maximum_chunk=16256,
    )
    plan.resource_plan = resources
    plan.temporary_buffers = temporaries
    context.scheduled_metadata.pop("ascendc_vector_chunk", None)
    metadata = {
        **resources.metadata(),
        "route": "basic-vector",
        "fallback_chunk": plan.fallback_chunk,
        "temporary_buffers": temporaries,
        "allocations": tuple(
            {
                "name": allocation.name,
                "bytes": allocation.bytes_for(resources.selected.elements),
                "slots": allocation.slots,
                "element_bytes": allocation.element_bytes,
                "divisor": allocation.divisor,
                "alignment": allocation.alignment,
            }
            for allocation in allocations
        ),
        "usable_ub_bytes": ASCENDC_UB_BYTES,
        "arch": "dav-c220",
    }
    context.scheduled_metadata["ascendc_vector_resources"] = metadata

    return resources


def configure_flat_dag(context, plan, extent, *, default_chunk, budget_bytes):
    """Keep flat packets complete and budget their actual physical slots."""
    requested = requested_vector_chunk(context)

    if requested is None:
        return default_chunk

    _verified_arch(context)
    from ninetoothed.dtype import normalize_dtype

    for name in (
        *(item.value.name for item in plan.inputs),
        *(item.output for item in plan.stores),
    ):
        info = context.tensors.get(name)

        if (
            info is not None
            and normalize_dtype((info.attrs or {}).get("source_dtype", info.dtype))
            is None
        ):
            raise ValueError(
                "The AscendC vector resource override requires proved dtypes."
            )

    sizes = {"float32": 4, "float16": 2, "int32": 4, "uint8": 1}
    allocations = [
        Allocation(f"buffer-{index}", sizes[dtype])
        for index, dtype in enumerate(plan.buffer_dtypes)
    ]
    allocations.extend(
        Allocation(
            f"scratch-{index}", sizes[dtype], divisor=8 if dtype == "uint8" else 1
        )
        for index, dtype in enumerate(plan.scratch_dtypes)
    )
    resources = plan_vector_chunks(
        allocations,
        budget_bytes=budget_bytes,
        candidates=(2048, 4096, 8192, 16256),
        default_chunk=default_chunk,
        requested_chunk=requested,
        maximum_chunk=16256,
        extent=extent,
        require_divisible=True,
    )
    context.scheduled_metadata["ascendc_vector_resources"] = {
        **resources.metadata(),
        "route": "flat-dag",
        "buffer_dtypes": plan.buffer_dtypes,
        "scratch_dtypes": plan.scratch_dtypes,
        "usable_ub_bytes": budget_bytes,
        "proved_extent": extent,
        "arch": "dav-c220",
    }

    return resources.selected.elements
