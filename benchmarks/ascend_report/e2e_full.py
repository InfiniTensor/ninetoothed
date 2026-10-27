"""Host adapters for the additional stage-seven inference replacements."""

import contextlib
import importlib


class UnsupportedFullReplacement(ValueError):
    """The requested stage-seven operation is outside the pinned model contract."""


def _load(name):
    return importlib.import_module("benchmarks.ascend_report." + name)


@contextlib.contextmanager
def patched(adapter, stage):
    if stage < 7 or adapter.backend != "ntops":
        yield
        return

    rope = _load("e2e_rope")
    llama = importlib.import_module("transformers.models.llama.modeling_llama")
    original_rotary = llama.apply_rotary_pos_emb

    def apply_rotary(query, key, cos, sin, position_ids=None, unsqueeze_dim=1):
        del position_ids

        if adapter.collect:
            adapter.active_operation = "rotary_embedding"
            adapter.calls["rotary_embedding"] += 2

        half = query.shape[-1] // 2
        cos = cos[..., :half].unsqueeze(unsqueeze_dim)
        sin = sin[..., :half].unsqueeze(unsqueeze_dim)
        return rope.rotary_embedding(query, cos, sin), rope.rotary_embedding(
            key, cos, sin
        )

    llama.apply_rotary_pos_emb = apply_rotary

    if __package__:
        from .e2e_cache import patched as cache_patched
    else:
        from e2e_cache import patched as cache_patched

    try:
        with cache_patched(adapter, max_length=512):
            yield {"rotary": rope}
    finally:
        llama.apply_rotary_pos_emb = original_rotary
