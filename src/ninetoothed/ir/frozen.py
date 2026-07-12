"""Pickle-safe immutable containers used by frozen IR records."""

from collections.abc import Iterator, Mapping
from typing import Any


class FrozenMap(Mapping):
    """A small immutable mapping with recursively frozen values."""

    __slots__ = ("_items",)

    def __init__(self, values=()):
        items = values.items() if isinstance(values, Mapping) else values
        object.__setattr__(
            self,
            "_items",
            tuple((key, freeze(value)) for key, value in items),
        )

    def __getitem__(self, key):
        for item_key, value in self._items:
            if item_key == key:
                return value

        raise KeyError(key)

    def __iter__(self) -> Iterator:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"FrozenMap({dict(self._items)!r})"

    def __reduce__(self):
        return type(self), (self._items,)


def freeze(value: Any) -> Any:
    if isinstance(value, FrozenMap):
        return value

    if isinstance(value, Mapping):
        return FrozenMap(value)

    if isinstance(value, (tuple, list)):
        return tuple(freeze(item) for item in value)

    if isinstance(value, (set, frozenset)):
        return frozenset(freeze(item) for item in value)
    return value


__all__ = ["FrozenMap", "freeze"]
