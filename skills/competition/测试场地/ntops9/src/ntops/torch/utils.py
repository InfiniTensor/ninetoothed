import functools

import ninetoothed
import torch

import ntops


class _CachedMakeDefaultConfig:
    def __init__(self, num_warps=None, num_stages=None, max_num_configs=None):
        self.num_warps = num_warps

        self.num_stages = num_stages

        self.max_num_configs = max_num_configs


_cached_make_default_config = _CachedMakeDefaultConfig()


def get_default_num_warps():
    return _cached_make_default_config.num_warps


def set_default_num_warps(num_warps):
    _cached_make_default_config.num_warps = num_warps


def get_default_num_stages():
    return _cached_make_default_config.num_stages


def set_default_num_stages(num_stages):
    _cached_make_default_config.num_stages = num_stages


def get_default_max_num_configs():
    return _cached_make_default_config.max_num_configs


def set_default_max_num_configs(max_num_configs):
    _cached_make_default_config.max_num_configs = max_num_configs


@functools.cache
def _cached_make(
    premake, *args, num_warps=None, num_stages=None, max_num_configs=None, **keywords
):
    if num_warps is None:
        num_warps = _cached_make_default_config.num_warps

    if num_stages is None:
        num_stages = _cached_make_default_config.num_stages

    if max_num_configs is None:
        max_num_configs = _cached_make_default_config.max_num_configs

    return ninetoothed.make(
        *premake(*args, **keywords),
        num_warps=num_warps,
        num_stages=num_stages,
        max_num_configs=max_num_configs,
    )


def _get_matmul_input_precision():
    if torch.get_float32_matmul_precision() == "highest":
        return ntops.kernels.mm.InputPrecisionVariant.IEEE

    return ntops.kernels.mm.InputPrecisionVariant.TF32
