"""Sufficient numerical domains for reassociating a native float32 sum."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SumDomain:
    dtype: str = "float32"
    maximum_count: int = 8192
    absolute_sum_limit: float = 1.7014117331926443e38

    def maximum_absolute_input(self, count):
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or not 0 < count <= self.maximum_count
        ):
            raise ValueError("A sum domain requires a proved positive bounded count.")
        return self.absolute_sum_limit / count


def native_sum_domain(opcode, *, dtype):
    """Keep a factor-two margin below FLT_MAX for every partial sum.

    For at most 8192 float32 additions, the usual rounding-error factor is
    below 0.001. If each finite absolute input is at most half FLT_MAX/count,
    every partial sum stays finite under any tree. NaN/Inf and values outside
    this sufficient domain retain the caller's original reduction tree.
    This proves absence of intermediate overflow; it does not promise exact
    bit equality between different floating-point reduction orders.
    """
    return SumDomain() if opcode == "reduce.sum" and dtype == "float32" else None
