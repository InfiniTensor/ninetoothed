"""Specialization hints for NineToothed code generation.

This module defines the data structures that carry per-(tensor, dim)
specialization hints from the AOT/JIT entry point down through the
code generator. When a hint is set, the generator emits a more
compact Triton source for that case; when no hint is set, the
generator falls back to the original generic path.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class SpecializationHints:
    """Per-(source_name, source_dim) specialization hints.

    :param divisible_dims: Set of ``(source_name, source_dim)`` for which
        the innermost tile is guaranteed to evenly divide
        ``shape[source_dim]``. The generator may skip the upper-bound
        mask term for these dims.
    :param contiguous_dims: Set of ``(source_name, source_dim)`` for
        which ``stride[source_dim] == 1`` is guaranteed. The generator
        may fold ``offsets * stride`` into ``offsets`` for these dims.

    Both sets use **bare** source names (i.e., naming.remove_prefixes
    already applied), matching what AOT spec sets store.
    """

    divisible_dims: frozenset = dataclasses.field(default_factory=frozenset)
    contiguous_dims: frozenset = dataclasses.field(default_factory=frozenset)

    @staticmethod
    def empty():
        return SpecializationHints(frozenset(), frozenset())

    def is_divisible(self, source_name, source_dim):
        return (source_name, source_dim) in self.divisible_dims

    def is_contiguous(self, source_name, source_dim):
        return (source_name, source_dim) in self.contiguous_dims

    def any(self):
        return bool(self.divisible_dims) or bool(self.contiguous_dims)
