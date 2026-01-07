import copy
import functools
import itertools
import math
import re

import ninetoothed.naming as naming
from ninetoothed.symbol import Symbol


class Tensor:
    """A class uesed to represent a symbolic tensor.

    :param ndim: The number of dimensions of the tensor.
    :param shape: The shape of the tensor.
    :param dtype: The element type of the tensor.
    :param jagged_dim: The jagged dimension of the tensor.
    :param other: The values for out-of-bounds positions.
    :param shape_options: The options for configuring shape symbols.
    :param name: The name of the tensor.
    :param source: For internal use only.
    :param target_dims: For internal use only.
    """

    num_instances = 0

    def __init__(
        self,
        ndim=None,
        shape=None,
        dtype=None,
        jagged_dim=None,
        other=None,
        shape_options=None,
        constexpr=None,
        value=None,
        name=None,
        source=None,
        target_dims=None,
        _offsets=None,
        _outputs=None,
    ):
        self.dtype = dtype

        self.jagged_dim = jagged_dim

        if name is not None:
            self.name = name
        else:
            self.name = naming.auto_generate(f"tensor_{type(self).num_instances}")

        if ndim is not None:
            if shape_options is None:
                shape_options = tuple({} for _ in range(ndim))

            if isinstance(shape_options, dict):
                shape_options = tuple(shape_options for _ in range(ndim))

            shape_options = tuple(
                size_options if size_options is not None else {}
                for size_options in shape_options
            )

            self.shape = (
                Symbol(self.size_string(i), **size_options)
                for i, size_options in zip(range(ndim), shape_options)
            )
        else:
            self.shape = shape

        self.other = other

        if constexpr and self.ndim != 0:
            raise ValueError(
                "`constexpr` can only be set for zero-dimensional tensors."
            )

        self.constexpr = constexpr

        if self.constexpr:
            self.name = naming.make_constexpr(self.name)

        if not constexpr and value is not None:
            raise ValueError("`value` can only be set for constexpr tensors.")

        self.value = value

        if source is not None:
            self.source = source
        else:
            self.source = self

        if target_dims is not None:
            self.target_dims = target_dims
        else:
            self.target_dims = (dim for dim in range(self.ndim))

        if _offsets is not None:
            self._levels = self.source._levels

            self._offsets = _offsets

            self._outputs = _outputs
        else:
            self._levels = [[self]]

            def _offsets(indices):
                return (tuple(indices),)

            self._offsets = _offsets

            self._outputs = [[]]

        self._inputs = []

        self._history = []

        type(self).num_instances += 1

    def __getitem__(self, indices):
        """Returns a sliced or indexed tensor using the specified key.

        :param indices: The index, slice, or tuple of indices and slices.
        :return: The sliced or indexed tensor.
        """

        if not isinstance(indices, tuple):
            indices = (indices,)
        num_nones = indices.count(None)
        num_ellipses = indices.count(...)

        if num_ellipses > 1:
            raise ValueError("Only one `Ellipsis` is allowed.")
        num_effective_indices = len(indices) - num_nones - num_ellipses
        if num_effective_indices > self.ndim:
            raise IndexError(
                f"Index `{indices}` is out of bounds for tensor of shape `{self.shape}`."
            )

        expanded_indices = []
        for index in indices:
            if index is Ellipsis:
                expanded_indices.extend(
                    [slice(None)] * (self.ndim - num_effective_indices)
                )
            else:
                expanded_indices.append(index)
        indices = expanded_indices

        output = self
        curr_dim = 0
        for index in indices:
            if index is None:
                # Adds a new dimension for `None` indexing.
                output = output.unsqueeze(curr_dim)
                curr_dim += 1
            elif isinstance(index, (int)):
                output = output._slice_dim(curr_dim, index, index + 1).squeeze(curr_dim)
            elif isinstance(index, slice):
                output = output._slice_dim(
                    curr_dim, index.start, index.stop, index.step
                )
                curr_dim += 1
            else:
                raise TypeError(
                    f"Index `{index}` must be an integer, `slice`, or `None`, not `{type(index).__name__}`."
                )
        return output

    @staticmethod
    def _meta_operation(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.source is self:
                self = copy.deepcopy(self)

            result = func(self, *args, **kwargs)

            result._history.extend(self._history)
            result._history.append((func, args, kwargs))

            return result

        return wrapper

    @_meta_operation
    def tile(self, tile_shape, strides=None, dilation=None, floor_mode=False):
        """Tiles the tensor into a hierarchical tensor.

        :param tile_shape: The shape of a tile.
        :param strides: The interval at which each tile is generated.
        :param dilation: The spacing between tiles.
        :param floor_mode: If ``True``, will use floor division to
            compute the outer shape.
        :return: A hierarchical tensor.
        """

        if strides is None:
            strides = [-1 for _ in tile_shape]

        if dilation is None:
            dilation = [1 for _ in tile_shape]

        outer_shape = []
        outer_strides = []
        inner_shape = []
        inner_strides = []

        for self_size, tile_size, stride, spacing in zip(
            self.shape, tile_shape, strides, dilation
        ):
            if tile_size == -1:
                tile_size = self_size

            if stride == -1:
                stride = tile_size

            def _div(x, y, floor_mode=False):
                if floor_mode:
                    return x // y

                return (x + y - 1) // y

            new_size = (
                (
                    _div(
                        self_size - spacing * (tile_size - 1) - 1,
                        stride,
                        floor_mode=floor_mode,
                    )
                    + 1
                )
                if stride != 0
                else -1
            )

            outer_shape.append(new_size)
            outer_strides.append(stride)

            inner_shape.append(tile_size)
            inner_strides.append(spacing)

        self._inputs.extend(([], []))

        def _offsets(indices):
            return (
                tuple(index * stride for index, stride in zip(indices, inner_strides)),
            )

        dtype = type(self)(
            shape=inner_shape,
            dtype=self.dtype,
            source=self.source,
            _offsets=_offsets,
            _outputs=[self._inputs[1]],
        )

        def _offsets(indices):
            return (
                tuple(index * stride for index, stride in zip(indices, outer_strides)),
            )

        output = type(self)(
            shape=outer_shape,
            dtype=dtype,
            source=self.source,
            _offsets=_offsets,
            _outputs=[self._inputs[0]],
        )

        self._levels.append([output, dtype])

        return output

    @_meta_operation
    def expand(self, shape):
        """Expands the specified singleton dimensions of the tensor.

        :param shape: The expanded shape.
        :return: The expanded tensor.
        """

        self._inputs.append([])

        def _offsets(indices):
            return (
                tuple(
                    index if new_size == -1 else 0 * index
                    for index, new_size in zip(indices, shape)
                ),
            )

        # TODO: Add error handling.
        output = type(self)(
            shape=[
                new_size if new_size != -1 else size
                for size, new_size in zip(self.shape, shape)
            ],
            dtype=self.dtype,
            source=self.source,
            target_dims=self.target_dims,
            _offsets=_offsets,
            _outputs=[self._inputs[0]],
        )

        self._levels.append([output])

        return output

    @_meta_operation
    def unsqueeze(self, dim):
        """Inserts a singleton dimension in the tensor.

        :param dim: The dimension to be unsqueezed.
        :return: The unsqueezed tensor.
        """

        # TODO: Add error handling.
        new_shape = list(self.shape)
        new_shape.insert(dim, 1)

        new_target_dims = list(self.target_dims)
        new_target_dims.insert(dim, None)

        self._inputs.append([])

        def _offsets(indices):
            return (tuple(index for i, index in enumerate(indices) if i != dim),)

        output = type(self)(
            shape=new_shape,
            dtype=self.dtype,
            source=self.source,
            target_dims=new_target_dims,
            _offsets=_offsets,
            _outputs=[self._inputs[0]],
        )

        self._levels.append([output])

        return output

    @_meta_operation
    def squeeze(self, dim):
        """Removes the specified singleton dimensions of the tensor.

        :param dim: The dimension(s) to be squeezed.
        :return: The squeezed tensor.
        """

        if not isinstance(dim, tuple):
            dim = (dim,)

        self._inputs.append([])

        def _offsets(indices):
            return (
                (
                    lambda iter: tuple(
                        next(iter) if i not in dim else 0 for i in range(self.ndim)
                    )
                )(iter(indices)),
            )

        # TODO: Add error handling.
        output = type(self)(
            shape=[size for i, size in enumerate(self.shape) if i not in dim],
            dtype=self.dtype,
            source=self.source,
            target_dims=[
                target_dim
                for i, target_dim in enumerate(self.target_dims)
                if i not in dim
            ],
            _offsets=_offsets,
            _outputs=[self._inputs[0]],
        )

        self._levels.append([output])

        return output

    @_meta_operation
    def permute(self, dims):
        """Permutes the dimensions of the tensor.

        :param dims: The permuted ordering of the dimensions.
        :return: The permuted tensor.
        """

        # TODO: Add error handling.
        new_shape = [None for _ in range(self.ndim)]

        for original_dim, permuted_dim in enumerate(dims):
            new_shape[original_dim] = self.shape[permuted_dim]

        self._inputs.append([])

        def _offsets(indices):
            return (tuple(indices[dims.index(dim)] for dim in range(len(dims))),)

        output = type(self)(
            shape=new_shape,
            dtype=self.dtype,
            source=self.source,
            target_dims=self.target_dims,
            _offsets=_offsets,
            _outputs=[self._inputs[0]],
        )

        self._levels.append([output])

        return output

    @_meta_operation
    def flatten(self, start_dim=None, end_dim=None):
        """Flattens the specified dimensions of the tensor.

        See :func:`ravel` for the differences between :func:`flatten`
        and :func:`ravel`.

        :param start_dim: The first dimension to flatten.
        :param end_dim: The dimension after the last to flatten.
        :return: The flattened tensor.
        """

        # TODO: Add error handling.
        if start_dim is None:
            start_dim = 0
        if end_dim is None:
            end_dim = self.ndim

        leading_sizes = self.shape[:start_dim]
        flattening_sizes = self.shape[start_dim:end_dim]
        trailing_sizes = self.shape[end_dim:]

        new_shape = leading_sizes + (math.prod(flattening_sizes),) + trailing_sizes

        leading_target_dims = self.target_dims[:start_dim]
        flattening_target_dims = self.target_dims[start_dim:end_dim]
        trailing_target_dims = self.target_dims[end_dim:]

        new_target_dims = (
            leading_target_dims + (flattening_target_dims[-1],) + trailing_target_dims
        )

        self._inputs.append([])

        def _offsets(indices):
            start_dim_ = self.ndim + start_dim if start_dim < 0 else start_dim

            return (
                (
                    indices[:start_dim_]
                    + type(self)._unravel_index(indices[start_dim_], flattening_sizes)
                    + indices[start_dim_ + 1 :]
                ),
            )

        output = type(self)(
            shape=new_shape,
            dtype=self.dtype,
            source=self.source,
            target_dims=new_target_dims,
            _offsets=_offsets,
            _outputs=[self._inputs[0]],
        )

        self._levels.append([output])

        return output

    @_meta_operation
    def ravel(self):
        """Flattens the hierarchy of the tensor.

        :func:`ravel` differs from :func:`flatten`, which only flattens
        dimensions at a single level. For example, consider a tensor
        with two levels: the first level has a shape of ``(N, P, Q)``,
        and the second level has a shape of ``(C, R, S)``. After
        applying :func:`ravel`, the resulting tensor will have a single
        flattened level with a shape of ``(N, P, Q, C, R, S)``.

        :return: The raveled tensor.
        """

        # TODO: Add error handling.
        new_shape = []
        outputs = []

        curr = self

        while isinstance(curr, type(self)):
            new_shape.extend(curr.shape)
            curr._inputs.append([])
            outputs.extend(curr._inputs)

            curr = curr.dtype

        def _offsets(indices):
            outputs = []

            curr = self
            start = 0

            while isinstance(curr, type(self)):
                stop = start + curr.ndim
                curr_indices = indices[start:stop]

                outputs.append(curr_indices)

                start = stop
                curr = curr.dtype

            return tuple(outputs)

        output = type(self)(
            shape=new_shape,
            other=self.source.other,
            name=self.source.name,
            source=self.source,
            _offsets=_offsets,
            _outputs=outputs,
        )

        self._levels.append([output])

        return output

    def offsets(self):
        indices = tuple(sum(indices) for indices in zip(*self._inputs))

        outputs = self._offsets(indices)

        for index, size in zip(indices, self.shape):
            self.source._mask &= Symbol(index) < size

        for output_, output in zip(self._outputs, outputs):
            output_.clear()
            output_.extend(output)

    def names(self):
        if self.ndim == 0:
            return {Symbol(self.source.name)}

        strides = tuple(
            Symbol(self.source.stride_string(dim)) for dim in range(self.source.ndim)
        )

        return (
            {Symbol(self.source.pointer_string())}
            | {
                name
                for value in itertools.chain(self.shape, strides)
                if isinstance(value, Symbol)
                for name in value.names()
            }
            | (
                {
                    Symbol(self.source.values_string()),
                    Symbol(self.source.offsets_string()),
                    Symbol(self.source.max_seq_len_string()),
                }
                if self.jagged_dim is not None
                else set()
            )
            | (self.dtype.names() if isinstance(self.dtype, type(self)) else set())
            | (self.source.names() if self.source is not self else set())
        )

    def innermost(self):
        if not isinstance(self.dtype, type(self)):
            return self

        return self.dtype.innermost()

    def pointer_string(self):
        if self.jagged_dim is not None:
            return self.values_string()

        return naming.auto_generate(f"{self.name}_pointer")

    def size_string(self, dim):
        if dim == self.jagged_dim:
            return self.seq_len_string()

        return naming.auto_generate(f"{self.name}_size_{dim}")

    def stride_string(self, dim):
        if self.jagged_dim is not None and dim == 0:
            return 0

        return naming.auto_generate(f"{self.name}_stride_{dim}")

    def values_string(self):
        return naming.auto_generate(f"{self.name}_values")

    def offsets_string(self):
        return naming.auto_generate(f"{self.name}_offsets")

    def max_seq_len_string(self):
        return naming.auto_generate(f"{self.name}_max_seq_len")

    def seq_len_string(self):
        return naming.auto_generate(f"{self.name}_seq_len")

    def size(self, dim=None):
        if dim is None:
            return self.shape

        return self.shape[dim]

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = tuple(value)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def target_dims(self):
        return self._target_dims

    @target_dims.setter
    def target_dims(self, value):
        self._target_dims = tuple(value)

    @_meta_operation
    def _slice_dim(self, dim, start, stop=None, step=1):
        """Slices the tensor along the specified dimension.

        :param dim: The dimension to slice.
        :param start: The starting index of the slice.
        :param stop: The ending index of the slice.
        :param step: The step size of the slice.
        :return: The sliced tensor.
        """

        if step is None:
            step = 1

        if dim < 0:
            dim += self.ndim

        size = self.shape[dim]

        # Handles `None` as default values.
        if start is None:
            start = 0 if step > 0 else size - 1
        if stop is None:
            stop = size if step > 0 else -1

        # Handles negative indices by adding the dimension size.
        if isinstance(start, int) and start < 0:
            start += size
        if isinstance(stop, int) and (stop < 0 if step > 0 else stop < -1):
            stop += size

        new_shape = list(self.shape)
        new_size = (stop - start + step - (1 if step > 0 else -1)) // step
        new_shape[dim] = new_size

        self._inputs.append([])

        def _offsets(indices):
            original_indices = list(indices)
            original_indices[dim] = start + indices[dim] * step
            return (tuple(original_indices),)

        output = type(self)(
            shape=new_shape,
            source=self.source,
            dtype=self.dtype,
            target_dims=self.target_dims,
            _offsets=_offsets,
            _outputs=[self._inputs[-1]],
        )

        self._levels.append([output])

        return output

    @staticmethod
    def pointer_pattern():
        return re.compile(
            naming.auto_generate(rf"({_identifier_pattern_raw_string()})_(pointer)")
        )

    @staticmethod
    def size_pattern():
        return re.compile(
            naming.auto_generate(rf"({_identifier_pattern_raw_string()})_(size)_(.+)")
        )

    @staticmethod
    def stride_pattern():
        return re.compile(
            naming.auto_generate(rf"({_identifier_pattern_raw_string()})_(stride)_(.+)")
        )

    @staticmethod
    def values_pattern():
        return re.compile(
            naming.auto_generate(rf"({_identifier_pattern_raw_string()})_(values)")
        )

    @staticmethod
    def offsets_pattern():
        return re.compile(
            naming.auto_generate(rf"({_identifier_pattern_raw_string()})_(offsets)")
        )

    @staticmethod
    def max_seq_len_pattern():
        return re.compile(
            naming.auto_generate(rf"({_identifier_pattern_raw_string()})_(max_seq_len)")
        )

    @staticmethod
    def seq_len_pattern():
        return re.compile(
            naming.auto_generate(rf"({_identifier_pattern_raw_string()})_(seq_len)")
        )

    @staticmethod
    def _unravel_index(index, shape):
        indices = []

        for stride in Tensor._calculate_default_strides(shape):
            indices.append(index // stride)
            index %= stride

        return tuple(indices)

    @staticmethod
    def _calculate_default_strides(shape):
        strides = [1]

        for size in reversed(shape[1:]):
            strides.append(size * strides[-1])

        return reversed(strides)


def _identifier_pattern_raw_string():
    return r"[a-zA-Z_][a-zA-Z0-9_]*"
