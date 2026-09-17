"""Checked NumPy storage and arranged tensor views for the SSA interpreter."""

from dataclasses import dataclass, field, replace

import numpy as np

from .expressions import evaluate, shape_value


def _identity_coordinates(shape):
    # Coordinate values are read-only. Broadcasting one vector per axis avoids
    # allocating rank copies of every logical element for an identity layout.
    return tuple(
        np.broadcast_to(coordinate, shape)
        for coordinate in np.indices(shape, sparse=True)
    )


@dataclass(frozen=True)
class Pointer:
    """An element offset into an explicitly contiguous NumPy allocation."""

    array: np.ndarray
    offset: object = 0
    observer: object | None = field(default=None, compare=False, repr=False)

    def shift(self, offset):
        return replace(self, offset=np.asarray(self.offset) + offset)

    def _indices(self, mask):
        if not self.array.flags.c_contiguous:
            raise ValueError("Raw pointer operations require C-contiguous arrays.")

        offsets, mask = np.broadcast_arrays(
            np.asarray(self.offset), np.asarray(mask, dtype=bool)
        )

        if offsets.dtype.kind not in "iu":
            raise ValueError("Pointer offsets must be integers.")

        selected = offsets[mask]

        if np.any(selected < 0) or np.any(selected >= self.array.size):
            raise IndexError("Active pointer lane is outside its allocation.")
        return offsets, mask

    def read(self, mask=True, other=0):
        offsets, mask = self._indices(mask)
        result = np.full(offsets.shape, other, dtype=self.array.dtype)
        # Do not evaluate even a NumPy index for a masked-out lane.
        result[mask] = self.array.reshape(-1)[offsets[mask]]

        if self.observer is not None:
            self.observer.access("read", self.array, (offsets,), mask, linear=True)

        return result

    def write(self, value, mask=True):
        offsets, mask = self._indices(mask)
        value = np.broadcast_to(np.asarray(value), offsets.shape)
        self.array.reshape(-1)[offsets[mask]] = value[mask]

        if self.observer is not None:
            self.observer.access("write", self.array, (offsets,), mask, linear=True)


@dataclass(frozen=True)
class TensorRef:
    """A symbolic arranged block backed by an actual NumPy tensor."""

    array: np.ndarray
    spec: object | None
    symbols: dict
    outer_index: int = 0
    level: int = 0
    extracted: tuple = ()
    observer: object | None = field(default=None, compare=False, repr=False)

    @property
    def layout(self):
        return None if self.spec is None else self.spec.layout

    @property
    def levels(self):
        return () if self.layout is None else self.layout.levels

    @property
    def shape(self):
        if self.levels:
            return shape_value(self.levels[self.level].shape, self.symbols)

        if self.layout is not None:
            return shape_value(self.layout.application_shape, self.symbols)
        return self.array.shape

    def extract(self, indices):
        if self.levels and self.level < len(self.levels) - 1:
            if len(indices) != len(self.shape):
                raise ValueError(
                    "Nested tensor extraction must index one complete layout level."
                )

            coordinates = tuple(int(index) for index in indices)

            if any(
                index < 0 or index >= size
                for index, size in zip(coordinates, self.shape)
            ):
                raise IndexError(
                    "Nested tensor extraction index is outside the logical tile."
                )
            return replace(
                self, level=self.level + 1, extracted=(*self.extracted, coordinates)
            )

        coordinates, valid = self._access()
        selected = tuple(indices)
        value = self._read_access(
            tuple(coordinate[selected] for coordinate in coordinates), valid[selected]
        )

        return value[()] if value.ndim == 0 else value

    def _access(self, extra_mask=True):
        shape = self.shape

        if self.levels and self.level < len(self.levels) - 1:
            raise ValueError(
                "Nested tensors must be indexed down to their innermost value level."
            )

        if self.layout is None:
            coords = _identity_coordinates(self.array.shape)

            return tuple(coords), np.broadcast_to(
                np.asarray(extra_mask, dtype=bool), self.array.shape
            )

        symbols = dict(self.symbols, outer_index=self.outer_index)

        for level, coordinates in enumerate(self.extracted):
            for dimension, value in enumerate(coordinates):
                symbols[f"extract_{level}_{dimension}"] = value

        if self.levels:
            if not self.layout.value_accesses:
                raise ValueError("Arranged tensor has no value access map.")

            access = self.layout.value_accesses[-1]

            for dimension, size in enumerate(shape):
                expand = (
                    (1,) * dimension + (size,) + (1,) * (len(shape) - dimension - 1)
                )
                symbols[f"value_{dimension}"] = np.arange(size, dtype=np.int64).reshape(
                    expand
                )
        else:
            access = self.layout.view_access

            if access is None:
                if self.array.shape == shape:
                    coords = _identity_coordinates(shape)

                    return tuple(coords), np.broadcast_to(
                        np.asarray(extra_mask, dtype=bool), shape
                    )

                raise ValueError("Tensor view has no access map.")

            symbols["index"] = np.arange(
                np.prod(shape, dtype=int), dtype=np.int64
            ).reshape(shape)

        mask = np.broadcast_to(
            np.asarray(evaluate(access.predicate, symbols), dtype=bool), shape
        )
        mask = mask & np.broadcast_to(np.asarray(extra_mask, dtype=bool), shape)
        coordinates = tuple(
            np.broadcast_to(np.asarray(evaluate(index, symbols)), shape)
            for index in access.source_indices
        )

        if len(coordinates) != self.array.ndim:
            raise ValueError("Layout/source rank mismatch.")

        for coordinate, size in zip(coordinates, self.array.shape):
            if coordinate.dtype.kind not in "iu":
                raise ValueError("Layout coordinates must be integers.")

            if np.any(coordinate[mask] < 0) or np.any(coordinate[mask] >= size):
                raise IndexError("Active layout lane is outside its source tensor.")
        return coordinates, mask

    def read(self, mask=True, other=None):
        coordinates, valid = self._access(mask)
        layout = self.layout

        if type(self.array) is np.ndarray and (
            layout is None or (not layout.levels and layout.view_access is None)
        ):
            return self._read_access(
                coordinates, valid, other, identity_source=self.array
            )

        return self._read_access(coordinates, valid, other)

    def _read_access(self, coordinates, valid, other=None, *, identity_source=None):
        if other is None:
            other = self.spec.attrs.get("other") if self.spec is not None else None

        result = np.full(
            valid.shape, 0 if other is None else other, dtype=self.array.dtype
        )

        if identity_source is not None:
            # The checked identity map can copy directly into fresh result storage.
            # Extraction still uses its selected coordinates, even for equal shapes.
            np.copyto(result, identity_source, where=valid)
        else:
            result[valid] = self.array[
                tuple(coordinate[valid] for coordinate in coordinates)
            ]

        if self.observer is not None:
            self.observer.access("read", self.array, coordinates, valid)

        return result

    def write(self, value, mask=True):
        coordinates, valid = self._access(mask)
        value = np.broadcast_to(np.asarray(value), valid.shape)

        if self.array.ndim == 0 and valid.ndim == 0:
            # An empty coordinate tuple selects the scalar even when masked out.
            # Boolean indexing preserves the mask and avoids array-to-scalar casts.
            self.array[valid] = value[valid]
        else:
            self.array[tuple(coordinate[valid] for coordinate in coordinates)] = value[
                valid
            ]

        if self.observer is not None:
            self.observer.access("write", self.array, coordinates, valid)


def materialize(value):
    """Load an arranged operand only when a numeric operation consumes it."""
    return value.read() if isinstance(value, TensorRef) else value
