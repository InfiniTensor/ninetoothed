import functools
import inspect
import itertools

import ninetoothed.jit
import ninetoothed.naming as naming
from ninetoothed.generation import cache_source
from ninetoothed.jit import import_from_path
from ninetoothed.make import make
from ninetoothed.symbol import Symbol
from ninetoothed.tensor import Tensor


class Node:
    def __init__(self, kernel, args=None, kwargs=None):
        if args is None:
            args = ()

        if kwargs is None:
            kwargs = {}

        self.kernel = kernel

        self.args = args

        self.kwargs = kwargs


def fuse(graph_module, _example_inputs):
    graph = graph_module.graph

    ninetoothed_nodes = []
    past_args = set()

    def _is_hoistable(node):
        if hasattr(node.target, "__name__") and node.target.__name__ in (
            "empty",
            "empty_like",
        ):
            return True

        for arg in _iterate_recursively(
            itertools.chain(node.args, node.kwargs.values())
        ):
            if arg in past_args:
                return False

        return True

    def _fuse_nodes(nodes):
        if len(nodes) == 1:
            return Node(nodes[0].kernel, args=nodes[0].args, kwargs=nodes[0].kwargs)

        return functools.reduce(_fuse_node_pair, nodes)

    for node in graph.nodes:
        if isinstance(node.target, ninetoothed.jit.__globals__["_Handle"]):
            ninetoothed_node = Node(node.target, args=node.args, kwargs=node.kwargs)
            ninetoothed_nodes.append(ninetoothed_node)
            past_args.update(
                _iterate_recursively(itertools.chain(node.args, node.kwargs.values()))
            )
            graph.erase_node(node)

            continue

        if not _is_hoistable(node):
            if ninetoothed_nodes:
                with graph.inserting_before(node):
                    ninetoothed_node = _fuse_nodes(ninetoothed_nodes)
                    graph.call_function(
                        ninetoothed_node.kernel,
                        args=ninetoothed_node.args,
                        kwargs=ninetoothed_node.kwargs,
                    )

                ninetoothed_nodes = []
                past_args = set()

            continue

    return graph_module.forward


def _fuse_node_pair(input_node, other_node):
    if input_node.kwargs or other_node.kwargs:
        return None

    input_kernel = input_node.kernel
    other_kernel = other_node.kernel

    mapping = {}

    for other_position, arg in enumerate(other_node.args):
        if arg not in input_node.args:
            continue

        mapping[other_position] = input_node.args.index(arg)

    fused_kernel = _fuse_kernel_pair(input_kernel, other_kernel, mapping)

    if fused_kernel is None:
        return None

    fused_args = input_node.args + other_node.args
    fused_kwargs = input_node.kwargs | other_node.kwargs

    fused_node = Node(fused_kernel, args=fused_args, kwargs=fused_kwargs)

    return fused_node


def _fuse_kernel_pair(input_kernel, other_kernel, mapping):
    arrangement, tensors = _fuse_arrangement_pair(input_kernel, other_kernel, mapping)

    if arrangement is None:
        return None

    application = _fuse_application_pair(input_kernel, other_kernel)

    if application is None:
        return None

    input_num_warps = (
        input_kernel.num_warps
        if not isinstance(input_kernel.num_warps, int)
        else (input_kernel.num_warps,)
    )
    other_num_warps = (
        other_kernel.num_warps
        if not isinstance(other_kernel.num_warps, int)
        else (other_kernel.num_warps,)
    )

    num_warps = tuple(set(input_num_warps) | set(other_num_warps))

    input_num_stages = (
        input_kernel.num_stages
        if not isinstance(input_kernel.num_stages, int)
        else (input_kernel.num_stages,)
    )
    other_num_stages = (
        other_kernel.num_stages
        if not isinstance(other_kernel.num_stages, int)
        else (other_kernel.num_stages,)
    )

    num_stages = tuple(set(input_num_stages) | set(other_num_stages))

    if input_kernel.max_num_configs is None or other_kernel.max_num_configs is None:
        max_num_configs = None
    else:
        max_num_configs = max(
            input_kernel.max_num_configs, other_kernel.max_num_configs
        )

    return make(
        arrangement,
        application,
        tensors,
        num_warps=num_warps,
        num_stages=num_stages,
        max_num_configs=max_num_configs,
    )


def _fuse_arrangement_pair(input_kernel, other_kernel, mapping):
    input_arrangement = input_kernel.arrangement
    other_arrangement = other_kernel.arrangement

    def rename_tensor(tensor, name):
        return Tensor(
            tensor.ndim,
            other=tensor.source.other,
            shape_options=tensor.shape_options,
            name=name,
        )

    input_tensors = tuple(
        rename_tensor(tensor, f"{tensor.name}_0") for tensor in input_kernel.tensors
    )
    other_tensors = tuple(
        rename_tensor(tensor, f"{tensor.name}_1") for tensor in other_kernel.tensors
    )

    input_tensors_arranged = input_arrangement(*input_tensors)
    other_tensors_arranged = other_arrangement(*other_tensors)

    input_tensor_positions = tuple(mapping.values())
    other_tensor_positions = tuple(mapping.keys())

    block_size_mapping = {}

    for input_tensor_position, other_tensor_position in zip(
        input_tensor_positions, other_tensor_positions
    ):
        for input_block_size, other_block_size in zip(
            input_tensors_arranged[input_tensor_position].innermost().shape,
            other_tensors_arranged[other_tensor_position].innermost().shape,
        ):
            for block_size in (input_block_size, other_block_size):
                if not (
                    Symbol.is_name(block_size) and naming.is_meta(block_size.node.id)
                ):
                    return None, None

            new_lower_bound = max(
                input_block_size.lower_bound, other_block_size.lower_bound
            )
            new_upper_bound = min(
                input_block_size.upper_bound, other_block_size.upper_bound
            )

            new_block_size = ninetoothed.block_size(
                lower_bound=new_lower_bound, upper_bound=new_upper_bound
            )

            block_size_mapping[input_block_size] = new_block_size
            block_size_mapping[other_block_size] = new_block_size

    for tensor in itertools.chain(input_tensors_arranged, other_tensors_arranged):
        _replace_history(tensor, block_size_mapping)

    (input_prefix, input_suffix), (other_prefix, other_suffix) = (
        _get_fusion_prefix_and_suffix(
            input_tensors_arranged[input_tensor_positions[0]],
            other_tensors_arranged[other_tensor_positions[0]],
        )
    )

    if input_prefix is None:
        return None, None

    for input_tensor_position, other_tensor_position in zip(
        input_tensor_positions[1:], other_tensor_positions[1:]
    ):
        (input_prefix_, input_suffix_), (other_prefix_, other_suffix_) = (
            _get_fusion_prefix_and_suffix(
                input_tensors_arranged[input_tensor_position],
                other_tensors_arranged[other_tensor_position],
            )
        )

        if (
            input_prefix_ != input_prefix
            or input_suffix_ != input_suffix
            or other_prefix_ != other_prefix
            or other_suffix_ != other_suffix
        ):
            return None, None

    records_on_tensors = []
    tensors = []

    def _get_records_on_tensor(tensor):
        records = []

        curr = tensor

        while isinstance(curr, type(tensor)):
            records.append(list(curr._history))

            curr = curr.dtype

        return records

    for input_tensor_position, (input_tensor, input_tensor_arranged) in enumerate(
        zip(input_tensors, input_tensors_arranged)
    ):
        records_on_tensor = _get_records_on_tensor(input_tensor_arranged)

        records_on_tensor[0] = (
            type(records_on_tensor[0])(input_prefix)
            + records_on_tensor[0]
            + type(records_on_tensor[0])(input_suffix)
        )

        records_on_tensors.append(records_on_tensor)
        tensors.append(input_tensor)

    for other_tensor_position, (other_tensor, other_tensor_arranged) in enumerate(
        zip(other_tensors, other_tensors_arranged)
    ):
        records_on_tensor = _get_records_on_tensor(other_tensor_arranged)

        records_on_tensor[0] = (
            type(records_on_tensor[0])(other_prefix)
            + records_on_tensor[0]
            + type(records_on_tensor[0])(other_suffix)
        )

        records_on_tensors.append(records_on_tensor)
        tensors.append(other_tensor)

    def arrangement(*tensors):
        tensors_arranged = []

        for records_on_tensor, tensor in zip(records_on_tensors, tensors):
            records_on_level_iter = iter(records_on_tensor)

            prev = None
            curr = tensor

            while isinstance(curr, type(tensor)):
                records_on_level = next(records_on_level_iter)

                for func, args, kwargs in records_on_level:
                    curr = func(curr, *args, **kwargs)

                if prev is not None:
                    prev.dtype = curr
                else:
                    tensors_arranged.append(curr)

                prev = curr
                curr = curr.dtype

        return tuple(tensors_arranged)

    return arrangement, tuple(tensors)


def _fuse_application_pair(input_kernel, other_kernel):
    input_application = input_kernel.application
    other_application = other_kernel.application

    input_params = inspect.signature(input_application).parameters
    other_params = inspect.signature(other_application).parameters

    count = 0

    def _make_param():
        nonlocal count

        param = naming.auto_generate(f"parameter_{count}")
        count += 1

        return param

    input_param_names = ", ".join(_make_param() for _ in input_params)
    other_param_names = ", ".join(_make_param() for _ in other_params)

    param_names = f"{input_param_names}, {other_param_names}"

    input_module = inspect.getmodule(input_application)
    other_module = inspect.getmodule(other_application)

    _APPLICATION_NAME = "application"

    application_source = f"""import {input_module.__name__}
import {other_module.__name__}


def {_APPLICATION_NAME}({param_names}):
    {input_module.__name__}.{input_application.__name__}({input_param_names})
    {other_module.__name__}.{other_application.__name__}({other_param_names})
"""

    source_file = cache_source(application_source)

    module = import_from_path(source_file.stem, source_file)
    module_vars = vars(module)

    application = module_vars[_APPLICATION_NAME]

    return application


def _get_fusion_prefix_and_suffix(input, other):
    if (fusion_position := _get_fusion_position(input, other)) is not None:
        prefix = tuple(input._history[:-fusion_position])
        suffix = tuple(other._history[fusion_position:])

        return ((), suffix), (prefix, ())

    if (fusion_position := _get_fusion_position(other, input)) is not None:
        prefix = tuple(other._history[:-fusion_position])
        suffix = tuple(input._history[fusion_position:])

        return (prefix, ()), ((), suffix)

    return (None, None), (None, None)


def _get_fusion_position(input, other):
    for k in range(1, len(input._history) + 1):
        if tuple(input._history)[-k:] == tuple(other._history)[:k]:
            return k

    return None


def _replace_history(tensor, mapping):
    curr = tensor

    while isinstance(curr, type(tensor)):
        history = []

        for record in curr._history:
            record = _replace_record(record, mapping)

            history.append(record)

        curr._history = tuple(history)

        curr = curr.dtype


def _replace_record(record, mapping):
    return (record[0], _replace(record[1], mapping), _replace(record[2], mapping))


def _replace(object, mapping):
    if isinstance(object, (list, tuple, set)):
        return type(object)(_replace(item, mapping) for item in object)

    if isinstance(object, dict):
        return {
            _replace(key, mapping): _replace(value, mapping)
            for key, value in object.items()
        }

    if object in mapping:
        return mapping[object]

    if isinstance(object, Symbol):
        for old, new in mapping.items():
            object = object.find_and_replace(old, new)

        return object

    return object


def _iterate_recursively(object):
    if isinstance(object, (str, bytes)):
        yield object

        return

    if isinstance(object, dict):
        for value in object.values():
            yield from _iterate_recursively(value)

        return

    try:
        for item in object:
            yield from _iterate_recursively(item)

        return
    except TypeError:
        yield object
