import itertools
import math
import textwrap

import torch

from ninetoothed.eval import _generate_target_tensor_shape
from ninetoothed.generation import cache_source
from ninetoothed.jit import import_from_path
from ninetoothed.make import make
from ninetoothed.tensor import Tensor


def simulate_arrangement(arrangement, tensors, device=None):
    """Simulate the arrangement of the tensors.

    :param arrangement: The arrangement of the tensors.
    :param tensors: The tensors.
    :param device: The device on which the tensors are.
    :return: A tuple of source tensors and a tuple of target tensors,
        where each tuple is sorted according to the parameter order
        specified in ``arrangement``, and each element in each tensor
        stores the index of that element in the source tensor.
    """

    if device is None:
        device = "cuda"

    def _arrangement(*tensors):
        return tensors

    source_tensors = []
    target_tensors = []

    for tensor, arranged in zip(tensors, arrangement(*tensors)):
        if tensor.ndim == 0:
            source_tensors.append(torch.tensor(0, device=device))
            target_tensors.append(torch.tensor(0, device=device))

            continue

        num_programs = math.prod(tensor.shape)

        shape = _generate_target_tensor_shape(arranged)

        source_tensor = torch.arange(num_programs, device=device).view(tensor.shape)
        target_tensor = torch.empty(shape, dtype=source_tensor.dtype, device=device)

        source_tensors.append(source_tensor)
        target_tensors.append(target_tensor)

    tensors = arrangement(
        *(
            Tensor(tensor.ndim, other=-1, shape_options={"constexpr": True})
            for tensor in tensors
        )
    )
    debug_tensors = tuple(_generate_debug_tensor(tensor) for tensor in tensors)

    application_source = _generate_debug_application_source(tensors, debug_tensors)

    source_file = str(cache_source(application_source))

    module = import_from_path(source_file, source_file)
    module_vars = vars(module)

    application = module_vars[_APPLICATION_NAME]

    kernel = make(_arrangement, application, tensors + debug_tensors)

    kernel(*source_tensors, *target_tensors)

    return tuple(source_tensors), tuple(target_tensors)


_INDENT = "    "

_APPLICATION_NAME = "application"


def _generate_debug_application_source(tensors, debug_tensors):
    params = ", ".join(
        tensor.source.name for tensor in itertools.chain(tensors, debug_tensors)
    )

    body_lines = []

    for tensor, debug_tensor in zip(tensors, debug_tensors):
        assignment_lines = _generate_debug_assignment_lines(tensor, debug_tensor)

        body_lines.extend(assignment_lines)

    body = textwrap.indent("\n".join(body_lines), _INDENT)

    return f"def {_APPLICATION_NAME}({params}):\n{body}"


def _generate_debug_assignment_lines(tensor, debug_tensor):
    if tensor.ndim == 0:
        return []

    num_indices = 0

    curr = debug_tensor.dtype

    while isinstance(curr.dtype, type(debug_tensor)):
        num_indices += curr.ndim

        curr = curr.dtype

    lines = []

    indices = []

    for i in range(num_indices):
        index = chr(ord("i") + i)
        line = textwrap.indent(
            f"for {index} in range({debug_tensor.source.name}.shape[{i}]):", _INDENT * i
        )

        lines.append(line)
        indices.append(index)

    if indices:
        joined_indices = ", ".join(indices)

        lines.append(
            textwrap.indent(
                f"{debug_tensor.source.name}[{joined_indices}] = {tensor.source.name}[{joined_indices}]",
                _INDENT * num_indices,
            )
        )
    else:
        lines.append(
            textwrap.indent(
                f"{debug_tensor.source.name} = {tensor.source.name}",
                _INDENT * num_indices,
            )
        )

    return lines


def _generate_debug_tensor(tensor):
    if tensor.ndim == 0:
        return Tensor(tensor.ndim)

    levels = []

    curr = tensor

    while isinstance(curr, type(tensor)):
        levels.append(curr)

        curr = curr.dtype

    ndim = sum(level.ndim for level in levels[1:]) + 1

    tile_shapes = []

    start = 0

    for level in reversed(levels[1:]):
        stop = start + level.ndim

        tile_shape = tuple(-1 if start <= i < stop else 1 for i in range(ndim))
        tile_shape = tuple(reversed(tile_shape))

        tile_shapes.append(tile_shape)

        start = stop

    debug_tensor = Tensor(ndim, shape_options={"constexpr": True})

    for tile_shape in tile_shapes:
        singleton_dims = tuple(i for i in range(ndim) if tile_shape[i] == 1)

        debug_tensor = debug_tensor.tile(tile_shape)
        debug_tensor.dtype = debug_tensor.dtype.squeeze(singleton_dims)

    return debug_tensor
