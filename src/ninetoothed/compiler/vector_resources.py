"""Aligned physical storage budgets for native vector execution."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VectorStorage:
    elements: int
    allocation_bytes: tuple[int, ...]

    @property
    def nbytes(self):
        return sum(self.allocation_bytes)


def plan_vector_storage(widths, *, elements, budget, alignment=128):
    """Account for every separately aligned buffer; never clamp a request."""
    widths = tuple(widths)

    if (
        not widths
        or any(
            type(n) is not int or n <= 0 for n in (*widths, elements, budget, alignment)
        )
        or elements % 128
    ):
        raise ValueError("Vector storage requires positive aligned dimensions.")

    sizes = tuple(
        (width * elements + alignment - 1) // alignment * alignment for width in widths
    )
    result = VectorStorage(elements, sizes)

    if result.nbytes > budget:
        raise ValueError(
            f"Vector storage requires {result.nbytes} bytes; budget is {budget}."
        )

    return result
