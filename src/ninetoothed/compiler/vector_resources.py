"""Exact local-storage budgets for target-proved vector schedule candidates.

Callers supply actual target allocations and legality bounds. This module
does not infer aliases, intrinsic support, or numerical properties from size.
"""

from dataclasses import dataclass


def _positive_integer(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"The {name} value must be a positive integer.")


@dataclass(frozen=True)
class Allocation:
    """One separately aligned allocation, repeated for each queue slot."""

    name: str
    element_bytes: int
    slots: int = 1
    divisor: int = 1
    alignment: int = 32
    fixed_bytes: int = 0

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Allocation name must be a nonempty string.")

        for name in ("element_bytes", "slots", "divisor", "alignment"):
            _positive_integer(getattr(self, name), name)

        if (
            not isinstance(self.fixed_bytes, int)
            or isinstance(self.fixed_bytes, bool)
            or self.fixed_bytes < 0
        ):
            raise ValueError("The fixed_bytes value must be a nonnegative integer.")

    def bytes_for(self, elements):
        """Round each buffer before multiplying slots, including packed bits."""
        _positive_integer(elements, "elements")
        size = (
            elements * self.element_bytes + self.divisor - 1
        ) // self.divisor + self.fixed_bytes

        return (
            (size + self.alignment - 1) // self.alignment * self.alignment * self.slots
        )


@dataclass(frozen=True)
class ChunkCandidate:
    elements: int
    allocation_bytes: int


@dataclass(frozen=True)
class ChunkRejection:
    elements: int
    reason: str


@dataclass(frozen=True)
class ResourcePlan:
    selected: ChunkCandidate
    legal: tuple[ChunkCandidate, ...]
    rejected: tuple[ChunkRejection, ...]
    requested: int | None

    def metadata(self):
        return {
            "requested_chunk": self.requested,
            "selected_chunk": self.selected.elements,
            "allocation_bytes": self.selected.allocation_bytes,
            "legal_chunks": tuple(item.elements for item in self.legal),
            "rejected_chunks": tuple(
                {"elements": item.elements, "reason": item.reason}
                for item in self.rejected
            ),
        }


def plan_vector_chunks(
    allocations,
    *,
    budget_bytes,
    candidates,
    default_chunk,
    requested_chunk=None,
    chunk_alignment=128,
    maximum_chunk=None,
    extent=None,
    require_divisible=False,
):
    """Filter exact resource candidates without changing the default policy.

    An explicit request must be legal; it is never silently clamped. The
    default must also be legal. Targets retain their complete scalar fallback
    and establish dtype, architecture, effect and access-domain legality.
    """
    allocations = tuple(allocations)

    if not allocations or any(not isinstance(item, Allocation) for item in allocations):
        raise ValueError("A resource plan requires actual Allocation records.")

    if len({item.name for item in allocations}) != len(allocations):
        raise ValueError("Allocation names must be unique.")

    for name, value in (
        ("budget_bytes", budget_bytes),
        ("chunk_alignment", chunk_alignment),
        ("default_chunk", default_chunk),
    ):
        _positive_integer(value, name)

    for name, value in (
        ("requested_chunk", requested_chunk),
        ("maximum_chunk", maximum_chunk),
        ("extent", extent),
    ):
        if value is not None:
            _positive_integer(value, name)

    if not isinstance(require_divisible, bool):
        raise ValueError("The require_divisible value must be a boolean.")

    if require_divisible and extent is None:
        raise ValueError("Divisible packets require a proved static extent.")

    proposed = tuple(candidates) + (default_chunk,)

    if requested_chunk is not None:
        proposed += (requested_chunk,)

    legal = []
    rejected = []
    seen = set()

    for elements in proposed:
        _positive_integer(elements, "candidate chunk")

        if elements in seen:
            continue

        seen.add(elements)
        size = sum(item.bytes_for(elements) for item in allocations)
        reason = None

        if elements % chunk_alignment:
            reason = f"Chunk is not aligned to {chunk_alignment} elements."
        elif maximum_chunk is not None and elements > maximum_chunk:
            reason = f"Chunk exceeds the target count limit {maximum_chunk}."
        elif require_divisible and extent % elements:
            reason = f"Chunk does not divide the proved extent {extent}."
        elif size > budget_bytes:
            reason = f"Allocations require {size} bytes; budget is {budget_bytes}."

        if reason:
            rejected.append(ChunkRejection(elements, reason))
        else:
            legal.append(ChunkCandidate(elements, size))

    selected_elements = default_chunk if requested_chunk is None else requested_chunk
    selected = next(
        (item for item in legal if item.elements == selected_elements), None
    )

    if selected is None:
        reason = next(
            item.reason for item in rejected if item.elements == selected_elements
        )
        reason = reason.removesuffix(".")

        raise ValueError(f"Illegal vector chunk {selected_elements}: {reason}.")

    return ResourcePlan(selected, tuple(legal), tuple(rejected), requested_chunk)
