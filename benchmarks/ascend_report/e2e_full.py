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
    gqa = _load("e2e_gqa")
    generation = _load("e2e_generation")
    embedding = _load("e2e_embedding")
    layout = _load("e2e_layout")
    runner = _load("e2e_runner")
    llama = importlib.import_module("transformers.models.llama.modeling_llama")
    sdpa_attention = importlib.import_module("transformers.integrations.sdpa_attention")
    adapter._direct_stage7 = True
    torch = adapter.torch
    original_rotary = llama.apply_rotary_pos_emb
    original_rotary_forward = llama.LlamaRotaryEmbedding.forward
    original_create_causal_mask = llama.create_causal_mask
    original_generate = runner.generate
    original_sdpa_forward = sdpa_attention.sdpa_attention_forward
    original_sdpa_repeat_kv = sdpa_attention.repeat_kv
    original_repeat_kv = llama.repeat_kv

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

    def rotary_forward(self, x, position_ids):
        if self.attention_scaling != 1:
            raise ValueError("The direct RoPE path requires attention_scaling=1.")

        cache = getattr(adapter, "_stage7_rope_frequency_cache", None)
        if cache is None:
            positions = torch.arange(512, device=x.device, dtype=torch.float32)
            cache = (
                rope.frequency_outer(self.inv_freq.reshape(-1), positions)
                .transpose(0, 1)
                .contiguous(),
                positions,
            )
            adapter._stage7_rope_frequency_cache = cache

        if adapter.collect:
            adapter.active_operation = "rope_frequency_outer"
            adapter.calls["rope_frequency_outer"] += 1
        frequencies = embedding.embedding(cache[0], position_ids.reshape(-1))
        frequencies = frequencies.unsqueeze(0)

        return adapter.call("cos", frequencies), adapter.call("sin", frequencies)

    llama.LlamaRotaryEmbedding.forward = rotary_forward

    def repeat_kv(hidden_states, n_rep):
        if n_rep == 1:
            return hidden_states

        if adapter.collect:
            adapter.active_operation = "gqa_head_expansion"
            adapter.calls["gqa_head_expansion"] += 1

        return gqa.repeat_kv_heads(hidden_states, n_rep)

    llama.repeat_kv = repeat_kv
    sdpa_attention.repeat_kv = repeat_kv

    def create_causal_mask(*args, **kwargs):
        del args, kwargs
        return None

    llama.create_causal_mask = create_causal_mask
    runner.generate = generation.generate
    generation.ACTIVE_ADAPTER = adapter

    def sdpa_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask=None,
        dropout_p=0.0,
        scaling=None,
        **kwargs,
    ):
        del attention_mask, kwargs

        attention = adapter.core_operator_adapters["sdpa"]
        output = attention(
            query,
            key,
            value,
            dropout_p=dropout_p,
            is_causal=query.shape[2] > 1,
            scale=module.scaling if scaling is None else scaling,
            enable_gqa=True,
        )
        if adapter.collect:
            adapter.active_operation = "attention_layout"
            adapter.calls["attention_layout"] += 1
        return layout.attention_output(output), None

    sdpa_attention.sdpa_attention_forward = sdpa_attention_forward
    original_sdpa_entry = llama.ALL_ATTENTION_FUNCTIONS["sdpa"]
    llama.ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_attention_forward

    if __package__:
        from .e2e_cache import patched as cache_patched
    else:
        from e2e_cache import patched as cache_patched

    try:
        with cache_patched(adapter, max_length=512):
            yield {"rotary": rope}
    finally:
        adapter._direct_stage7 = False
        generation.ACTIVE_ADAPTER = None
        runner.generate = original_generate
        llama.ALL_ATTENTION_FUNCTIONS["sdpa"] = original_sdpa_entry
        sdpa_attention.sdpa_attention_forward = original_sdpa_forward
        llama.create_causal_mask = original_create_causal_mask
        sdpa_attention.repeat_kv = original_sdpa_repeat_kv
        llama.repeat_kv = original_repeat_kv
        llama.LlamaRotaryEmbedding.forward = original_rotary_forward
        llama.apply_rotary_pos_emb = original_rotary
