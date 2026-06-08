"""Public entry point: ninetoothed.make(), with content-sensitive handle cache.

The handle cache (L1, in-process, FIFO) is keyed by a content hash of the
arrangement + application source code, tensor structural signatures, and
compilation parameters. Editing the user-facing functions invalidates the
cache; editing unrelated code does not.
"""

import inspect

from ninetoothed._cache import Cache, hash_function_source, hash_tensor_signature
from ninetoothed.aot import aot
from ninetoothed.jit import jit
from ninetoothed.tensor import Tensor


def _build_cache_key(
    arrangement,
    application,
    tensors,
    caller,
    kernel_name,
    num_warps,
    num_stages,
    max_num_configs,
):
    def _hash_one(t):
        # Tensor instances get content-sensitive structural hashing.
        if isinstance(t, Tensor):
            return hash_tensor_signature(t)
        # Non-Tensor elements (slices, ints, lists, etc. used as
        # arrangement() kwargs) are hashed via repr() so they
        # correctly participate in the cache key.
        return ("__raw__", repr(t))

    return (
        hash_function_source(arrangement),
        hash_function_source(application),
        tuple(_hash_one(t) for t in tensors),
        caller,
        kernel_name,
        num_warps,
        num_stages,
        max_num_configs,
    )


# Per-process L1 cache for JIT handles. Not shared across processes
# (handles are not serializable). 256-entry FIFO matches prior behavior.
_HANDLE_CACHE = Cache(max_memory=256)


def make(
    arrangement,
    application,
    tensors,
    caller="torch",
    kernel_name=None,
    output_dir=None,
    num_warps=None,
    num_stages=None,
    max_num_configs=None,
):
    """Integrate the arrangement and the application of the tensors.

    :param arrangement: The arrangement of the tensors.
    :param application: The application of the tensors.
    :param tensors: The tensors.
    :param caller: Who will call the compute kernel.
    :param kernel_name: The name for the generated kernel.
    :param output_dir: The directory to store the generated files.
    :param num_warps: The number of warps to use.
    :param num_stages: The number of stages to use.
    :param max_num_configs: The maximum number of auto-tuning
        configurations to use.
    :return: A handle to the compute kernel.
    """

    # Cache only the JIT ("torch") path. The AOT path produces on-disk
    # build artifacts (.so, .csv, .fingerprint) that are managed by
    # build.py's own cache.
    if caller == "torch":
        key = _build_cache_key(
            arrangement,
            application,
            tensors,
            caller,
            kernel_name,
            num_warps,
            num_stages,
            max_num_configs,
        )
        cached = _HANDLE_CACHE.get(key)
        if cached is not None:
            return cached

    params = inspect.signature(application).parameters
    types = arrangement(*tensors)
    types = types if isinstance(types, tuple) else (types,)
    annotations = {param: type for param, type in zip(params, types)}
    application.__annotations__ = annotations

    if caller == "torch":
        handle = jit(
            application,
            caller=caller,
            kernel_name=kernel_name,
            num_warps=num_warps,
            num_stages=num_stages,
            max_num_configs=max_num_configs,
        )
        _HANDLE_CACHE.put(key, handle)
        return handle

    return aot(
        application,
        caller=caller,
        kernel_name=kernel_name,
        output_dir=output_dir,
        num_warps=num_warps,
        num_stages=num_stages,
    )
