import itertools
import re

import ninetoothed.naming as naming
from ninetoothed.language import call
from ninetoothed.symbol import Symbol


class Tensor:
    num_instances = 0

    def __init__(
        self,
        ndim=None,
        shape=None,
        dtype=None,
        strides=None,
        other=None,
        name=None,
        original=None,
    ):
        self.dtype = dtype

        if name is not None:
            self.name = name
        else:
            self.name = f"_ninetoothed_tensor_{type(self).num_instances}"

        if ndim is not None:
            self.shape = (Symbol(self.size_string(i)) for i in range(ndim))
            self.strides = (Symbol(self.stride_string(i)) for i in range(ndim))
        else:
            self.shape = shape

            if strides is not None:
                self.strides = strides
            else:
                self.strides = self._calculate_default_strides(shape)

        self.other = other

        if original is not None:
            self.original = original
        else:
            self.original = self

        type(self).num_instances += 1

    def tile(self, tile_shape, strides=None, dilation=None):
        if strides is None:
            strides = [-1 for _ in tile_shape]

        if dilation is None:
            dilation = [1 for _ in tile_shape]

        outer_shape = []
        outer_strides = []
        inner_shape = []
        inner_strides = []

        for self_size, self_stride, tile_size, stride, spacing in zip(
            self.shape, self.strides, tile_shape, strides, dilation
        ):
            if tile_size == -1:
                tile_size = self_size

            if stride == -1:
                stride = tile_size

            new_size = (
                call("cdiv", self_size - spacing * (tile_size - 1) - 1, stride) + 1
                if stride != 0
                else -1
            )
            outer_shape.append(new_size)

            new_stride = self_stride * stride // spacing
            outer_strides.append(new_stride)

            inner_shape.append(tile_size)
            next_stride = self_stride * spacing
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

    def squeeze(self, dim):
        # TODO: Add error handling.
        return type(self)(
            shape=[size for i, size in enumerate(self.shape) if dim != i],
            dtype=self.dtype,
            strides=[stride for i, stride in enumerate(self.strides) if dim != i],
            original=self.original,
        )

    def names(self):
        if self.ndim == 0:
            return {self.original.name}

        return (
            {self.original.pointer_string()}
            | {
                name
                for value in itertools.chain(self.shape, self.strides)
                if isinstance(value, Symbol)
                for name in value.names()
            }
            | (self.dtype.names() if isinstance(self.dtype, type(self)) else set())
            | (self.original.names() if self.original is not self else set())
        )

    def offsets(self, indices=None):
        if indices is None:
            indices = self.indices()

        offsets = [[] for _ in range(self.original.ndim)]

        curr = self
        start = 0

        while isinstance(curr, type(self)):
            stop = start + curr.ndim
            curr_indices = indices[start:stop]

            for index, stride in zip(curr_indices, curr.strides):
                for dim in self._dims_of(stride):
                    offsets[dim].append(index * stride)

            start = stop
            curr = curr.dtype

        for dim in range(self.original.ndim):
            offsets[dim] = sum(offsets[dim])
            offsets[dim].find_and_replace(Symbol(self.original.strides[dim]), Symbol(1))

        return offsets

    def indices(self, index=None):
        if index is None:
            index = call("program_id", 0)

        indices = []

        for stride in type(self)(shape=self.shape, original=self.original).strides:
            indices.append(index // stride)
            index %= stride

        curr = self.dtype

        while isinstance(curr.dtype, type(self)):
            for _ in range(curr.ndim):
                indices.append(0)

            curr = curr.dtype

        if isinstance(curr, type(self)):
            for dim in range(curr.ndim):
                size = curr.shape[dim]

                if Symbol.is_name(size):
                    name = size.node.id
                    if not naming.is_meta(name):
                        size = naming.make_next_power_of_2(name)

                indices.append(call("arange", 0, size))

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

    def size(self, dim=None):
        if dim is None:
            return self.shape

        return self.shape[dim]

    def stride(self, dim=None):
        if dim is None:
            return self.strides

        return self.strides[dim]

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = tuple(value)

    @property
    def strides(self):
        return self._strides

    @strides.setter
    def strides(self, value):
        self._strides = tuple(value)

    @property
    def ndim(self):
        return len(self.shape)

    @staticmethod
    def pointer_pattern():
        return re.compile(rf"({_identifier_pattern_raw_string()})_(pointer)")

    @staticmethod
    def size_pattern():
        return re.compile(rf"({_identifier_pattern_raw_string()})_(size)_(.+)")

    @staticmethod
    def stride_pattern():
        return re.compile(rf"({_identifier_pattern_raw_string()})_(stride)_(.+)")

    def _dims_of(self, stride):
        dims = set()
        names = stride.names() if isinstance(stride, Symbol) else {stride}

        for dim, original_stride in enumerate(self.original.strides):
            if str(original_stride) in names:
                dims.add(dim)

        return dims

    @staticmethod
    def _calculate_default_strides(shape):
        strides = [1]

        for size in reversed(shape[1:]):
            strides.append(size * strides[-1])

        return reversed(strides)


def _identifier_pattern_raw_string():
    return r"[a-zA-Z_][a-zA-Z0-9_]*"
