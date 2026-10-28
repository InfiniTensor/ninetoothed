"""Minimal stage-seven greedy generation loop for the pinned Llama workload."""

import torch

if __package__:
    from .e2e_layout import copy_flat, iota
else:
    from e2e_layout import copy_flat, iota

ACTIVE_ADAPTER = None


def _record(name):
    adapter = ACTIVE_ADAPTER
    if adapter is not None and adapter.collect:
        adapter.active_operation = name
        adapter.calls[name] += 1


def generate(model, inputs, tokenizer, settings):
    del tokenizer

    initial_ids = inputs["input_ids"]
    initial_length = initial_ids.shape[-1]
    output_ids = torch.empty(
        (1, initial_length + settings["output_tokens"]),
        dtype=initial_ids.dtype,
        device=initial_ids.device,
    )

    _record("position_iota")
    cache_positions = iota(output_ids.shape[-1])

    _record("token_store")
    copy_flat(
        initial_ids.reshape(-1),
        output_ids[:, :initial_length].reshape(-1),
    )

    forward = model(
        input_ids=initial_ids,
        attention_mask=None,
        cache_position=cache_positions[:initial_length],
        use_cache=True,
    )
    past_key_values = forward.past_key_values

    for index in range(settings["output_tokens"]):
        token_tensor = torch.argmax(forward.logits[:, -1:, :], dim=-1)
        destination = output_ids[
            :, initial_length + index : initial_length + index + 1
        ].reshape(-1)
        _record("token_store")
        copy_flat(token_tensor.reshape(-1), destination)

        if index + 1 == settings["output_tokens"]:
            break

        current_position = cache_positions[
            initial_length + index : initial_length + index + 1
        ]
        forward = model(
            input_ids=output_ids[
                :, initial_length + index : initial_length + index + 1
            ],
            attention_mask=None,
            cache_position=current_position,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = forward.past_key_values

    return output_ids
