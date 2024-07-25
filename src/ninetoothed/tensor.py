from ninetoothed.language import call
from ninetoothed.symbol import Symbol


class Tensor:
    num_instances = 0

    def __init__(self, ndim=None, shape=None, dtype=None, strides=None, name=None):
        type(self).num_instances += 1

        self.dtype = dtype

        if name is not None:
            self.name = name
        else:
            self.name = f"tensor_{type(self).num_instances}"

        if ndim is not None:
            self.shape = [Symbol(f"{self.name}_size_{i}") for i in range(ndim)]
            self.strides = [Symbol(f"{self.name}_stride_{i}") for i in range(ndim)]
        else:
            self.shape = shape

            if strides is not None:
                self.strides = strides
            else:
                self.strides = self._calculate_default_strides(shape)

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

            new_stride = call("cdiv", stride * size, (new_size * tile_stride))
            outer_strides.append(new_stride)

            inner_shape.append(tile_size)
            next_stride = stride * tile_stride
            inner_strides.append(next_stride)

        return type(self)(
            shape=outer_shape,
            dtype=type(self)(shape=inner_shape, strides=inner_strides, name=self.name),
            strides=outer_strides,
            name=self.name,
        )

    def names(self):
        return (
            {self._pointer()}
            | {name for size in self.shape for name in size.names()}
            | {name for stride in self.strides for name in stride.names()}
        )

    def pointers(self, offsets=None):
        if offsets is None:
            offsets = self.offsets()

        return self._pointer() + offsets

    def offsets(self, indices=None):
        if indices is None:
            indices = self.indices()

        if not isinstance(self.dtype, type(self)):
            if indices:
                raise IndexError("Incorrect number of indices.")

            return sum(
                self.stride(idx)
                * call("arange", 0, self.size(idx))[
                    tuple(slice(None) if i == idx else None for i in range(self.ndim()))
                ]
                for idx in range(self.ndim())
            )

        outer_indices = indices[: self.ndim()]
        inner_indices = indices[self.ndim() :]

        return sum(
            index * stride for index, stride in zip(outer_indices, self.strides)
        ) + self.dtype.offsets(inner_indices)

    def indices(self, index=None):
        if index is None:
            index = call("program_id", 0)

        indices = []

        for stride in type(self)(shape=self.shape, name=self.name).strides:
            indices.append(index // stride)
            index %= stride

        return tuple(indices)

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

    def _pointer(self):
        return f"{self.name}_ptr"

    @staticmethod
    def _calculate_default_strides(shape):
        strides = [1]

        for size in shape[1:]:
            strides.append(size * strides[-1])

        return reversed(strides)
