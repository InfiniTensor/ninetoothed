"""CPU memory model for the reference interpreter.

The model mirrors the layout information the NineToothed frontend attaches to
every tensor: ``source_shape``/``source_strides`` describe the backing buffer,
``dtype_shapes`` describes the tile hierarchy, and ``access_templates`` maps
``(outer_index, extract_*, value_*)`` coordinates to a linear buffer offset plus
a bounds predicate.

Two invariants are enforced here.  A masked-out access never touches the backing
buffer.  An unmasked access that falls outside the buffer is an error, not a
silent read or write of the wrong element.
"""

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .dtypes import resolve_dtype
from .errors import (
    MissingSymbolError,
    UnsupportedAccessError,
    UnsupportedDTypeError,
)
from .expr import parse_expression

#: Subscript element kinds.
NEW_AXIS = "new_axis"
SLICE = "slice"
INTEGER = "integer"


def _is_balanced(text) -> bool:
    """Return whether every bracket in ``text`` is closed in order."""
    depth = 0

    for character in text:
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1

            if depth < 0:
                return False

    return depth == 0


def _split_top_level(text) -> tuple:
    """Split ``text`` on commas that are not nested inside brackets."""
    parts = []
    depth = 0
    current = []

    for character in text:
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth -= 1

        if character == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue

        current.append(character)

    tail = "".join(current).strip()

    if tail or parts:
        parts.append(tail)

    return tuple(part for part in parts if part)


def parse_subscript(text) -> tuple:
    """Parse an SSA ``subscript`` attribute into element kinds.

    The frontend records the unparsed slice node, so the text is a
    comma-separated list instead of a valid expression.  For example
    ``x[:, None]`` is recorded as ``"(:, None)"`` and ``x[0, i]`` as
    ``"(0, i)"``.  This matches the backend emitters' convention.

    :param text: The subscript text, for example ``"(0, i)"`` or ``"(:, None)"``.
    :return: A tuple whose entries are :data:`NEW_AXIS`, :data:`SLICE`, or
        :data:`INTEGER`.
    """
    if text is None:
        return ()

    text = str(text).strip()

    if not text:
        return ()

    if text.startswith("(") and text.endswith(")") and _is_balanced(text[1:-1]):
        text = text[1:-1]

    kinds = []

    for part in _split_top_level(text):
        if part == "None":
            kinds.append(NEW_AXIS)
        elif ":" in part:
            kinds.append(SLICE)
        else:
            kinds.append(INTEGER)

    return tuple(kinds)


def _as_int(value, namespace, what):
    """Resolve a possibly symbolic integer expression to a Python ``int``."""
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, float):
        if value != int(value):
            raise MissingSymbolError(
                f"Expected `{what}` to be an integer; got {value!r}."
            )
        return int(value)

    if not isinstance(value, str):
        raise MissingSymbolError(f"Cannot resolve `{what}` from {value!r}.")

    text = value.strip()

    if not text:
        raise MissingSymbolError(f"Cannot resolve an empty `{what}`.")

    try:
        return int(text)
    except ValueError:
        pass

    try:
        result = parse_expression(text).evaluate(namespace)
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise MissingSymbolError(
            f"Cannot resolve `{what}` from `{text}`: {exc}."
        ) from exc

    if isinstance(result, np.ndarray):
        if result.size != 1:
            raise MissingSymbolError(
                f"Cannot resolve `{what}` from `{text}`: got {result!r}."
            )

        result = result.reshape(()).item()

    return int(result)


@dataclass(frozen=True)
class AccessTemplate:
    """A compiled ``access_templates`` entry."""

    level: int
    shape: tuple
    offsets: tuple
    linear_offset: Any
    mask: Any

    @staticmethod
    def from_attrs(template: Mapping[str, Any]) -> "AccessTemplate":
        return AccessTemplate(
            level=int(template.get("level", 0)),
            shape=tuple(str(dim) for dim in template.get("shape", ())),
            offsets=tuple(
                parse_expression(str(offset)) for offset in template.get("offsets", ())
            ),
            linear_offset=parse_expression(str(template.get("linear_offset", 0))),
            mask=parse_expression(str(template.get("mask", True))),
        )


@dataclass
class TensorRuntime:
    """A CPU buffer together with the layout needed to address it."""

    name: str
    buffer: np.ndarray
    dtype: str
    levels: tuple
    source_shape: tuple
    source_strides: tuple
    view_shape: tuple
    template: AccessTemplate | None
    view_offsets: Any = None
    view_mask: Any = None
    other: Any = 0
    writable: bool = True

    @property
    def numpy_dtype(self):
        """Return the NumPy dtype of the backing buffer."""
        return self.buffer.dtype

    @property
    def addressable_level(self):
        """Return the dtype level that can be mapped to the backing buffer.

        A tiled tensor resolves through its innermost ``access_templates`` entry;
        a tensor that was only sliced resolves through its view-level map, which
        lives on dtype level ``0``.
        """
        if self.template is None:
            return 0

        return len(self.levels) - 1

    @property
    def is_tiled(self):
        """Return whether the tensor carries a tile-hierarchy access template."""
        return self.template is not None

    @property
    def launch_extent(self):
        """Return the number of program instances implied by the view shape."""
        if not self.is_tiled:
            return 1

        return int(math.prod(self.view_shape)) if self.view_shape else 1

    def root_view(self, level=0):
        """Return the root (unindexed) view of this tensor."""
        return View(
            tensor=self,
            level=level,
            assigned={},
            positions=tuple(range(len(self.levels[level]))),
            shape=tuple(int(dim) for dim in self.levels[level]),
        )


@dataclass(frozen=True)
class View:
    """A lazily addressed tile view of a :class:`TensorRuntime`."""

    tensor: TensorRuntime
    level: int
    assigned: Mapping = field(default_factory=dict)
    positions: tuple = ()
    shape: tuple = ()

    def free_dims(self):
        """Return the dims of the current level that are not yet fixed."""
        fixed = self.assigned.get(self.level, {})

        return tuple(
            dim
            for dim in range(len(self.tensor.levels[self.level]))
            if dim not in fixed
        )

    def natural_shape(self):
        """Return the extents of the free dims of the current level."""
        level_shape = self.tensor.levels[self.level]

        return tuple(int(level_shape[dim]) for dim in self.free_dims())

    def index(self, kinds, coords, *, opcode=None, location=None):
        """Apply a subscript to this view.

        :param kinds: Element kinds from :func:`parse_subscript`.
        :param coords: The integer index values, in subscript order.
        :return: The resulting :class:`View`.
        """
        level_shape = self.tensor.levels[self.level]
        assigned = {level: dict(dims) for level, dims in self.assigned.items()}
        current = assigned.setdefault(self.level, {})
        positions = []
        shape = []
        cursor = 0
        coord_iter = iter(coords)

        for kind in kinds:
            if kind is NEW_AXIS:
                positions.append(None)
                shape.append(1)
                continue

            if cursor >= len(self.positions):
                raise UnsupportedAccessError(
                    f"Subscript for tensor `{self.tensor.name}` has more elements "
                    f"than the view has dimensions.",
                    opcode=opcode,
                    location=location,
                )

            natural = self.positions[cursor]
            cursor += 1

            if kind is SLICE:
                if natural is None:
                    raise UnsupportedAccessError(
                        f"Slice subscript for tensor `{self.tensor.name}` targets an "
                        "inserted dimension.",
                        opcode=opcode,
                        location=location,
                    )

                positions.append(natural)
                shape.append(int(level_shape[natural]))
                continue

            if natural is None:
                raise UnsupportedAccessError(
                    f"Integer subscript for tensor `{self.tensor.name}` targets an "
                    "inserted dimension.",
                    opcode=opcode,
                    location=location,
                )

            if natural in current:
                raise UnsupportedAccessError(
                    f"Dimension {natural} of tensor `{self.tensor.name}` is indexed "
                    "more than once.",
                    opcode=opcode,
                    location=location,
                )

            try:
                current[natural] = int(next(coord_iter))
            except StopIteration as exc:
                raise UnsupportedAccessError(
                    f"Subscript for tensor `{self.tensor.name}` is missing index "
                    "values.",
                    opcode=opcode,
                    location=location,
                ) from exc

        for natural in self.positions[cursor:]:
            positions.append(natural)

            if natural is not None:
                shape.append(int(level_shape[natural]))

        if not shape and self.level + 1 < len(self.tensor.levels):
            next_level = self.level + 1

            return View(
                tensor=self.tensor,
                level=next_level,
                assigned=assigned,
                positions=tuple(range(len(self.tensor.levels[next_level]))),
                shape=tuple(int(dim) for dim in self.tensor.levels[next_level]),
            )

        return View(
            tensor=self.tensor,
            level=self.level,
            assigned=assigned,
            positions=tuple(positions),
            shape=tuple(shape),
        )


class AccessContext:
    """Binds the symbolic layout expressions to one program instance.

    :param symbols: The resolved symbol values (shapes, strides, constexprs).
    :param program_id: The linear id of the program instance being executed.
    """

    __slots__ = ("program_id", "symbols", "total")

    def __init__(self, symbols, program_id, total=None):
        self.symbols = dict(symbols)
        self.program_id = int(program_id)
        self.total = total

    def namespace(self, view):
        """Return the evaluation namespace for a view of one tensor."""
        tensor = view.tensor

        if view.level != tensor.addressable_level:
            raise UnsupportedAccessError(
                f"Tensor `{tensor.name}` is read at dtype level {view.level}, but "
                f"only level {tensor.addressable_level} (the innermost) has an "
                "access map. Index the outer levels with `tensor.extract` "
                "(for example `x[0, i]`) before reading elements."
            )

        if not tensor.is_tiled:
            return self._view_namespace(view)

        template = tensor.template
        namespace = dict(self.symbols)
        namespace["outer_index"] = self.program_id
        namespace["index"] = self.program_id
        namespace["inner_index"] = 0

        if self.total is not None:
            namespace["total"] = int(self.total)

        for level in range(view.level):
            coords = view.assigned.get(level, {})

            for dim in range(len(tensor.levels[level])):
                if dim not in coords:
                    raise UnsupportedAccessError(
                        f"Tensor `{tensor.name}` is read at dtype level "
                        f"{view.level} without fixing dimension {dim} of level "
                        f"{level}."
                    )

                namespace[f"extract_{level}_{dim}"] = int(coords[dim])

        free = view.free_dims()
        extents = [int(tensor.levels[view.level][dim]) for dim in free]
        grids = (
            np.meshgrid(*[np.arange(extent) for extent in extents], indexing="ij")
            if extents
            else []
        )
        fixed = view.assigned.get(view.level, {})

        for dim in range(len(tensor.levels[view.level])):
            if dim in fixed:
                namespace[f"value_{dim}"] = int(fixed[dim])
            else:
                namespace[f"value_{dim}"] = grids[free.index(dim)]

        del template

        return namespace

    def _view_namespace(self, view):
        """Namespace for a tensor addressed through its view-level index map."""
        namespace = dict(self.symbols)
        linear = self._view_linear_index(view)
        namespace["index"] = linear
        namespace["outer_index"] = linear
        namespace["inner_index"] = 0

        if self.total is not None:
            namespace["total"] = int(self.total)

        return namespace

    def _view_linear_index(self, view):
        """Return the linear view-domain index for every element of a view."""
        tensor = view.tensor
        level_shape = tuple(int(dim) for dim in tensor.levels[0])
        free = view.free_dims()
        extents = [level_shape[dim] for dim in free]
        shape = tuple(extents)
        grids = (
            np.meshgrid(*[np.arange(extent) for extent in extents], indexing="ij")
            if extents
            else []
        )
        fixed = view.assigned.get(0, {})
        strides = _default_strides(level_shape)
        linear = np.zeros(shape, dtype=np.int64)

        for dim in range(len(level_shape)):
            if dim in fixed:
                linear = linear + int(fixed[dim]) * strides[dim]
            else:
                linear = linear + grids[free.index(dim)] * strides[dim]

        return linear

    def evaluate_access(self, view):
        """Return ``(offsets, mask)`` for every element of a view."""
        tensor = view.tensor
        namespace = self.namespace(view)

        if tensor.is_tiled:
            offsets = tensor.template.linear_offset.evaluate(namespace)
            mask = tensor.template.mask.evaluate(namespace)
        else:
            offsets = tensor.view_offsets.evaluate(namespace)
            mask = tensor.view_mask.evaluate(namespace)

        return np.asarray(offsets, dtype=np.int64), np.asarray(mask, dtype=bool)

    def read(self, view):
        """Gather the elements of a view, honouring the bounds predicate."""
        tensor = view.tensor
        offsets, mask = self.evaluate_access(view)
        natural = view.natural_shape()
        offsets = np.broadcast_to(offsets, natural)
        mask = np.broadcast_to(mask, natural)
        size = tensor.buffer.size
        masked_in = mask & ((offsets < 0) | (offsets >= size))

        if masked_in.any():
            raise UnsupportedAccessError(
                f"Tensor `{tensor.name}` resolves an in-bounds-expected access to "
                f"offset {int(np.max(offsets[masked_in]))} outside the buffer of "
                f"{size} element(s)."
            )

        safe = np.where(mask, offsets, 0)
        gathered = (
            tensor.buffer[safe] if size else np.zeros(natural, dtype=tensor.numpy_dtype)
        )
        other = tensor.other if tensor.other is not None else 0
        result = np.where(mask, gathered, other).astype(tensor.numpy_dtype, copy=False)

        return result.reshape(view.shape)

    def write(self, view, values, *, atomic=False):
        """Scatter ``values`` through a view, honouring the bounds predicate."""
        tensor = view.tensor

        if not tensor.writable:
            raise UnsupportedAccessError(
                f"Tensor `{tensor.name}` is not writable in this interpretation."
            )

        offsets, mask = self.evaluate_access(view)
        natural = view.natural_shape()
        offsets = np.broadcast_to(offsets, natural)
        mask = np.broadcast_to(mask, natural)
        values = np.asarray(values)

        if values.shape != natural:
            try:
                values = np.broadcast_to(values, natural)
            except ValueError as exc:
                raise UnsupportedAccessError(
                    f"Cannot store a value of shape {values.shape} into tensor "
                    f"`{tensor.name}` through a view of shape {natural}."
                ) from exc

        values = values.astype(tensor.numpy_dtype, copy=False)
        size = tensor.buffer.size
        masked_in = mask & ((offsets < 0) | (offsets >= size))

        if masked_in.any():
            raise UnsupportedAccessError(
                f"Tensor `{tensor.name}` resolves an in-bounds-expected store to "
                f"offset {int(np.max(offsets[masked_in]))} outside the buffer of "
                f"{size} element(s)."
            )

        if not mask.any():
            return

        if atomic:
            np.add.at(tensor.buffer, offsets[mask], values[mask])
        else:
            tensor.buffer[offsets[mask]] = values[mask]


def tile_access_map(tensor, context):
    """Return the access map of one program instance's whole tile hierarchy.

    A tensor with several dtype levels is only addressable once every outer
    level has been indexed.  This function enumerates those outer coordinates so
    the returned arrays cover the full tile, with shape
    ``concat(levels[0..n])`` where ``n`` is the addressable level.

    :param tensor: A :class:`TensorRuntime`.
    :param context: An :class:`AccessContext` for one program instance.
    :return: ``(offsets, mask)`` as NumPy arrays.
    """
    level = tensor.addressable_level
    outer_shapes = [
        tuple(int(dim) for dim in tensor.levels[index]) for index in range(level)
    ]
    level_shape = tuple(int(dim) for dim in tensor.levels[level])
    extents = [range(dim) for shape in outer_shapes for dim in shape]
    offset_blocks = []
    mask_blocks = []

    for combination in itertools.product(*extents):
        assigned = {}
        cursor = 0

        for index, shape in enumerate(outer_shapes):
            coordinates = {}

            for dim in range(len(shape)):
                coordinates[dim] = combination[cursor]
                cursor += 1

            assigned[index] = coordinates

        view = View(
            tensor=tensor,
            level=level,
            assigned=assigned,
            positions=tuple(range(len(level_shape))),
            shape=level_shape,
        )
        offsets, mask = context.evaluate_access(view)
        offset_blocks.append(np.asarray(offsets))
        mask_blocks.append(np.asarray(mask))

    outer_extents = tuple(dim for shape in outer_shapes for dim in shape)
    offsets = np.stack(offset_blocks).reshape(outer_extents + level_shape)
    mask = np.stack(mask_blocks).reshape(outer_extents + level_shape)

    return offsets, mask


class CPUMemory:
    """The set of CPU buffers visible to one interpretation."""

    def __init__(self, symbols, total=None):
        self.symbols = dict(symbols)
        self.total = total
        self.tensors: dict[str, TensorRuntime] = {}

    def add(self, tensor: TensorRuntime) -> None:
        """Register a tensor runtime."""
        self.tensors[tensor.name] = tensor

    def get(self, name) -> TensorRuntime:
        """Return a registered tensor runtime."""
        try:
            return self.tensors[name]
        except KeyError as exc:
            raise UnsupportedAccessError(f"Unknown tensor `{name}`.") from exc

    def context(self, program_id):
        """Return an :class:`AccessContext` for one program instance."""
        return AccessContext(self.symbols, program_id, total=self.total)

    def allocate(self, name, dtype, shape, *, fill=0):
        """Allocate a fresh contiguous buffer and register it as a tensor."""
        numpy_dtype = resolve_dtype(dtype)
        buffer = np.full(int(math.prod(shape)) if shape else 1, fill, dtype=numpy_dtype)
        tensor = TensorRuntime(
            name=name,
            buffer=buffer,
            dtype=str(dtype),
            levels=(tuple(int(dim) for dim in shape),),
            source_shape=tuple(int(dim) for dim in shape),
            source_strides=_default_strides(shape),
            view_shape=(1,) if shape else (),
            template=None,
            other=0,
        )
        self.add(tensor)

        return tensor


def _default_strides(shape):
    strides = [1]

    for size in reversed(tuple(shape)[1:]):
        strides.append(size * strides[-1])

    return tuple(reversed(strides))


def build_tensor_runtime(name, type_, array, symbols):
    """Build a :class:`TensorRuntime` from an SSA type and a runtime array.

    :param name: The SSA value name of the tensor.
    :param type_: The specialized :class:`ninetoothed.ir.ssa.Type`.
    :param array: The runtime ``numpy.ndarray`` backing the tensor.
    :param symbols: The resolved symbol values.
    :return: A :class:`TensorRuntime`.
    """
    attrs = dict(type_.attrs)
    dtype = type_.dtype

    if dtype is None:
        dtype = str(array.dtype)

    try:
        numpy_dtype = resolve_dtype(dtype)
    except UnsupportedDTypeError as exc:
        reason = exc.message.rstrip(".")

        raise UnsupportedDTypeError(f"Tensor `{name}`: {reason}.") from exc

    buffer = np.ascontiguousarray(array).reshape(-1)

    if buffer.dtype != numpy_dtype:
        buffer = buffer.astype(numpy_dtype)

    source_shape = tuple(
        _as_int(dim, symbols, f"{name}.source_shape")
        for dim in attrs.get("source_shape", ())
    )

    if not source_shape:
        source_shape = tuple(int(dim) for dim in array.shape)

    source_strides = tuple(
        _as_int(dim, symbols, f"{name}.source_strides")
        for dim in attrs.get("source_strides", ())
    )

    if not source_strides:
        source_strides = _default_strides(source_shape)

    levels = tuple(
        tuple(_as_int(dim, symbols, f"{name}.dtype_shapes") for dim in shape)
        for shape in attrs.get("dtype_shapes", ())
    )

    view_shape = tuple(
        _as_int(dim, symbols, f"{name}.view_shape")
        for dim in attrs.get("view_shape", ())
    )

    if not view_shape:
        view_shape = tuple(int(dim) for dim in type_.shape)

    if not levels:
        levels = (view_shape,)

    templates = attrs.get("access_templates", ())
    template = AccessTemplate.from_attrs(templates[-1]) if templates else None
    view_offsets = attrs.get("view_linear_offset")
    view_mask = attrs.get("view_mask")

    if template is None and (view_offsets is None or view_mask is None):
        raise UnsupportedAccessError(
            f"Tensor `{name}` has neither `access_templates` nor a view-level "
            "index map; the interpreter cannot resolve its memory access pattern."
        )

    return TensorRuntime(
        name=name,
        buffer=buffer,
        dtype=str(dtype),
        levels=levels,
        source_shape=source_shape,
        source_strides=source_strides,
        view_shape=view_shape,
        template=template,
        view_offsets=None
        if view_offsets is None
        else parse_expression(str(view_offsets)),
        view_mask=None if view_mask is None else parse_expression(str(view_mask)),
        other=attrs.get("other"),
    )


__all__ = [
    "INTEGER",
    "NEW_AXIS",
    "SLICE",
    "AccessContext",
    "AccessTemplate",
    "CPUMemory",
    "TensorRuntime",
    "View",
    "build_tensor_runtime",
    "parse_subscript",
    "tile_access_map",
]
