"""Generated kernels and cache patching for the experimental full-replacement stage."""

import contextlib
import functools

import torch

import ninetoothed
from ninetoothed import Tensor


def arrangement(x, out):
    return x.tile((1, 1, 1, -1)), out.tile((1, 1, 1, -1))


def application(x, out):
    out = x  # noqa: F841


@functools.cache
def _kernel(dtype_name, ndim):
    dtype = getattr(torch, dtype_name)
    return ninetoothed.make(
        arrangement,
        application,
        (Tensor(ndim, dtype=dtype), Tensor(ndim, dtype=dtype)),
        kernel_name=f"e2e_cache_copy_{dtype_name}_{ndim}",
        max_num_configs=1,
    )


def copy_token_values(source, destination):
    """Copy a [batch, heads, tokens, width] source into a cache slice."""
    if source.ndim != 4 or source.shape != destination.shape:
        raise ValueError(
            "Cache copy source and destination views must have equal rank-four shapes"
        )
    _kernel(str(source.dtype).removeprefix("torch."), source.ndim)(source, destination)


@contextlib.contextmanager
def patched(adapter, *, max_length=512):
    """Replace DynamicLayer's native concatenation with generated cache copies."""
    from transformers.cache_utils import DynamicLayer

    original_update = DynamicLayer.update

    def update(self, key_states, value_states, cache_kwargs=None):
        del cache_kwargs

        if not self.is_initialized:
            self.dtype, self.device = key_states.dtype, key_states.device
            shape = (1, key_states.shape[1], max_length, key_states.shape[-1])
            self._ninetoothed_key_storage = torch.empty(
                shape, dtype=self.dtype, device=self.device
            )
            self._ninetoothed_value_storage = torch.empty(
                shape, dtype=self.dtype, device=self.device
            )
            self.keys = self._ninetoothed_key_storage[:, :, :0, :]
            self.values = self._ninetoothed_value_storage[:, :, :0, :]
            self.is_initialized = True

        old_length = self.keys.shape[-2]
        new_length = old_length + key_states.shape[-2]
        if new_length > max_length:
            raise ValueError(
                f"Experimental preallocated KV cache overflow: {new_length} > {max_length}"
            )

        if adapter.collect:
            adapter.active_operation = "kv_cache_copy"
            adapter.calls["kv_cache_copy"] += 2

        copy_token_values(
            key_states,
            self._ninetoothed_key_storage[:, :, old_length:new_length, :],
        )
        copy_token_values(
            value_states,
            self._ninetoothed_value_storage[:, :, old_length:new_length, :],
        )
        self.keys = self._ninetoothed_key_storage[:, :, :new_length, :]
        self.values = self._ninetoothed_value_storage[:, :, :new_length, :]
        return self.keys, self.values

    DynamicLayer.update = update
    try:
        yield
    finally:
        DynamicLayer.update = original_update
