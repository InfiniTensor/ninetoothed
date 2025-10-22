import copy
import math
import re

import numpy as np

import ninetoothed.language
from ninetoothed.generation import CodeGenerator
from ninetoothed.symbol import Symbol
from ninetoothed.tensor import Tensor

_NUMPY = "np"


def _eval(tensor, subs=None):
    """Evaluate the symbolic tensor into a numeric tensor.

    :param tensor: The symbolic tensor.
    :param subs: The substitutions for symbolic variables.
    :return: A numeric tensor as a ``numpy.ndarray``.

    .. versionchanged:: 0.22.0

        The dimensions of the outermost level are preserved.
        Previously, the evaluation would flatten the dimensions of the
        outermost level.
    """

    def _generate_replacements(subs):
        subs = copy.deepcopy(subs)

        replacements = {
            ninetoothed.language.LANGUAGE: _NUMPY,
            "slice(None, None, None)": ":",
        }

        for old, new in subs.items():
            if isinstance(old, Tensor):
                if "strides" not in new:
                    new["strides"] = Tensor._calculate_default_strides(new["shape"])

                for dim, size in enumerate(new["shape"]):
                    replacements[old.size_string(dim)] = str(size)

                for dim, stride in enumerate(new["strides"]):
                    replacements[old.stride_string(dim)] = str(stride)
            elif isinstance(old, Symbol):
                replacements[str(old)] = str(new)

        return replacements

    def _replace(string, replacements):
        for old, new in sorted(
            replacements.items(), key=lambda key: len(key), reverse=True
        ):
            string = re.sub(rf"\b{re.escape(old)}\b", new, string)

        return string

    if tensor.source.ndim == 0:
        return np.array(0, dtype=np.intp)

    if subs is None:
        subs = {tensor.source: {"shape": tensor.source.shape}}

    replacements = _generate_replacements(subs)

    shape = _generate_target_tensor_shape(tensor)
    shape = tuple(eval(_replace(str(size), replacements)) for size in shape)

    if not isinstance(tensor.dtype, type(tensor)):
        return np.arange(math.prod(shape), dtype=np.intp).reshape(shape)

    result = np.empty(shape, dtype=np.intp)

    for index in np.ndindex(shape[: -tensor.innermost().ndim]):
        overall_offsets, mask = CodeGenerator._generate_overall_offsets_and_mask(
            tensor,
            tuple(Symbol(index) for index in index)
            + CodeGenerator._generate_innermost_indices(tensor),
        )

        overall_offsets = _replace(str(overall_offsets), replacements)
        mask = _replace(str(mask), replacements)

        result[index + (...,)] = eval(
            f"np.where({mask}, {overall_offsets}, -1)", {_NUMPY: np}
        )

    return result


def _generate_target_tensor_shape(tensor, flatten_outermost=False):
    if flatten_outermost:
        shape = [math.prod(tensor.shape)]

        curr = tensor.dtype
    else:
        shape = []

        curr = tensor

    while isinstance(curr, type(tensor)):
        shape.extend(curr.shape)

        curr = curr.dtype

    return shape


Tensor.eval = _eval
