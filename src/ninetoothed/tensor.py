import itertools
import math
import re

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
        source=None,
        source_dims=None,
        target=None,
        target_dims=None,
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

        if source is not None:
            self.source = source
        else:
            self.source = self

        if source_dims is not None:
            self.source_dims = source_dims
        else:
            self.source_dims = (dim for dim in range(self.source.ndim))

        if target is not None:
            self.target = target
        else:
            self.target = self

        if target_dims is not None:
            self.target_dims = target_dims
        else:
            self.target_dims = (dim for dim in range(self.target.ndim))

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
                source=self.source,
                source_dims=self.source_dims,
                target=self.target,
                target_dims=self.target_dims,
            ),
            strides=outer_strides,
            source=self.source,
            source_dims=self.source_dims,
            target=self.target,
            target_dims=self.target_dims,
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
            source=self.source,
            source_dims=self.source_dims,
            target=self.target,
            target_dims=self.target_dims,
        )

    def squeeze(self, dim):
        # TODO: Add error handling.
        return type(self)(
            shape=[size for i, size in enumerate(self.shape) if dim != i],
            dtype=self.dtype,
            strides=[stride for i, stride in enumerate(self.strides) if dim != i],
            source=self.source,
            source_dims=[
                source_dim for i, source_dim in enumerate(self.source_dims) if dim != i
            ],
            target=self.target,
            target_dims=[
                target_dim for i, target_dim in enumerate(self.target_dims) if dim != i
            ],
        )

    def permute(self, dims):
        # TODO: Add error handling.
        new_shape = [None for _ in range(self.ndim)]
        new_strides = [None for _ in range(self.ndim)]
        new_source_dims = [None for _ in range(self.ndim)]

        for original_dim, permuted_dim in enumerate(dims):
            new_shape[original_dim] = self.shape[permuted_dim]
            new_strides[original_dim] = self.strides[permuted_dim]
            new_source_dims[original_dim] = self.source_dims[permuted_dim]

        return type(self)(
            shape=new_shape,
            dtype=self.dtype,
            strides=new_strides,
            source=self.source,
            source_dims=new_source_dims,
            target=self.target,
            target_dims=self.target_dims,
        )

    def flatten(self, start_dim=None, end_dim=None):
        # TODO: Add error handling.
        if start_dim is None:
            start_dim = 0
        if end_dim is None:
            end_dim = self.ndim

        leading_sizes = self.shape[:start_dim]
        flattening_sizes = self.shape[start_dim:end_dim]
        trailing_sizes = self.shape[end_dim:]

        new_shape = leading_sizes + (math.prod(flattening_sizes),) + trailing_sizes

        leading_strides = self.strides[:start_dim]
        flattening_strides = self.strides[start_dim:end_dim]
        trailing_strides = self.strides[end_dim:]

        new_strides = leading_strides + (flattening_strides[-1],) + trailing_strides

        leading_source_dims = self.source_dims[:start_dim]
        flattening_source_dims = self.source_dims[start_dim:end_dim]
        trailing_source_dims = self.source_dims[end_dim:]

        new_source_dims = (
            leading_source_dims + (flattening_source_dims,) + trailing_source_dims
        )

        return type(self)(
            shape=new_shape,
            dtype=self.dtype,
            strides=new_strides,
            source=self.source,
            source_dims=new_source_dims,
            target=self.target,
            target_dims=self.target_dims,
        )

    def ravel(self):
        # TODO: Add error handling.
        new_shape = []
        new_strides = []

        curr = self

        while isinstance(curr, type(self)):
            new_shape.extend(curr.shape)
            new_strides.extend(curr.strides)

            curr = curr.dtype

        return type(self)(
            shape=new_shape,
            strides=new_strides,
            other=self.source.other,
            name=self.source.name,
        )

    def names(self):
        if self.ndim == 0:
            return {self.source.name}

        return (
            {self.source.pointer_string()}
            | {
                name
                for value in itertools.chain(self.shape, self.strides)
                if isinstance(value, Symbol)
                for name in value.names()
            }
            | (self.dtype.names() if isinstance(self.dtype, type(self)) else set())
            | (self.source.names() if self.source is not self else set())
        )

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

    @property
    def source_dims(self):
        return self._source_dims

    @source_dims.setter
    def source_dims(self, value):
        self._source_dims = tuple(value)

    @property
    def target_dims(self):
        return self._target_dims

    @target_dims.setter
    def target_dims(self, value):
        self._target_dims = tuple(value)

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

        for size in reversed(shape[1:]):
            strides.append(size * strides[-1])

        return reversed(strides)


def _identifier_pattern_raw_string():
    return r"[a-zA-Z_][a-zA-Z0-9_]*"
