import itertools
import re

from ninetoothed.language import call
from ninetoothed.symbol import Symbol


class Tensor:
    num_instances = 0

    def __init__(self, ndim=None, shape=None, dtype=None, strides=None, original=None):
        type(self).num_instances += 1

        self.dtype = dtype

        self.name = f"tensor_{type(self).num_instances}"

        if ndim is not None:
            self.shape = [Symbol(self.size_string(i)) for i in range(ndim)]
            self.strides = [Symbol(self.stride_string(i)) for i in range(ndim)]
        else:
            self.shape = shape

            if strides is not None:
                self.strides = strides
            else:
                self.strides = self._calculate_default_strides(shape)

        if original is not None:
            self.original = original
        else:
            self.original = self

    def tile(self, tile_shape, tile_strides=None):
        if tile_strides is None:
            tile_strides = [1 for _ in tile_shape]

        outer_shape = []
        outer_strides = []
        inner_shape = []
        inner_strides = []

        for size, stride, tile_size, tile_stride in zip(
            self.shape, self.strides, tile_shape, tile_strides
        ):
            if tile_size == -1:
                tile_size = size

            new_size = call("cdiv", size, tile_size)
            outer_shape.append(new_size)

            new_stride = stride * tile_size // tile_stride
            outer_strides.append(new_stride)

            inner_shape.append(tile_size)
            next_stride = stride * tile_stride
            inner_strides.append(next_stride)

        return type(self)(
            shape=outer_shape,
            dtype=type(self)(
                shape=inner_shape,
                dtype=self.dtype,
                strides=inner_strides,
                original=self.original,
            ),
            strides=outer_strides,
            original=self.original,
        )

    def expand(self, shape):
        # TODO: Add error handling.
        return type(self)(
            shape=[
                new_size if new_size != -1 else size
                for size, new_size in zip(self.shape, shape)
            ],
            dtype=self.dtype,
            strides=[
                stride if new_size == -1 else 0
                for new_size, stride in zip(shape, self.strides)
            ],
            original=self.original,
        )

    def names(self):
        return (
            {self.original.pointer_string()}
            | {
                name
                for value in itertools.chain(self.shape, self.strides)
                if isinstance(value, Symbol)
                for name in value.names()
            }
            | (self.dtype.names() if isinstance(self.dtype, type(self)) else set())
        )

    def offsets(self, indices=None):
        if indices is None:
            indices = self.indices()

        if not isinstance(self.dtype, type(self)):
            if len(indices) != self.ndim():
                raise IndexError("Incorrect number of indices.")

            return tuple(
                indices[idx]
                * self.stride(idx)
                * call("arange", 0, self.size(idx))[
                    tuple(slice(None) if i == idx else None for i in range(self.ndim()))
                ]
                for idx in range(self.ndim())
            )

        outer_indices = indices[: self.ndim()]
        inner_indices = indices[self.ndim() :]

        return tuple(
            index * stride for index, stride in zip(outer_indices, self.strides)
        ) + self.dtype.offsets(inner_indices)

    def indices(self, index=None):
        if index is None:
            index = call("program_id", 0)

        indices = []

        for stride in type(self)(shape=self.shape, original=self.original).strides:
            indices.append(index // stride)
            index %= stride

        curr = self.dtype
        while isinstance(curr, type(self)):
            indices.extend(
                0 if curr is not self.inmost() else 1 for _ in range(curr.ndim())
            )
            curr = curr.dtype

        return tuple(indices)

    def inmost(self):
        if not isinstance(self.dtype, type(self)):
            return self

        return self.dtype.inmost()

    def pointer_string(self):
        return f"{self.name}_pointer"

    def size_string(self, dim):
        return f"{self.name}_size_{dim}"

    def stride_string(self, dim):
        return f"{self.name}_stride_{dim}"

    def ndim(self):
        return len(self.shape)

    def size(self, dim=None):
        if dim is None:
            return self.shape

        return self.shape[dim]

    def stride(self, dim=None):
        if dim is None:
            return self.strides

        return self.strides[dim]

    @staticmethod
    def pointer_pattern():
        return re.compile(rf"({_identifier_pattern_raw_string()})_(pointer)")

    @staticmethod
    def size_pattern():
        return re.compile(rf"({_identifier_pattern_raw_string()})_(size)_(.+)")

    @staticmethod
    def stride_pattern():
        return re.compile(rf"({_identifier_pattern_raw_string()})_(stride)_(.+)")

    @staticmethod
    def _calculate_default_strides(shape):
        strides = [1]

        for size in shape[1:]:
            strides.append(size * strides[-1])

        return reversed(strides)


def _identifier_pattern_raw_string():
    return r"[a-zA-Z_][a-zA-Z0-9_]*"
