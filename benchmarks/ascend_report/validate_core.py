"""Validate the model's attention and Linear shapes before generation timing."""

import argparse
import json
import math
from pathlib import Path

if __package__:
    from . import e2e_runner as runner
    from .e2e_attention import Adapter as AttentionAdapter
    from .e2e_linear import Adapter as LinearAdapter
else:
    import e2e_runner as runner
    from e2e_attention import Adapter as AttentionAdapter
    from e2e_linear import Adapter as LinearAdapter


def validate(config, backend):
    torch, cache = runner.configure_runtime(config)
    recorder = runner.Adapters(backend, torch, cache)
    attention = AttentionAdapter(backend, torch, recorder=recorder)
    linear = LinearAdapter(
        backend,
        torch,
        recorder=recorder,
        tuning=config["e2e"]["linear_tuning"],
        block_sizes=config["e2e"]["linear_block_sizes"],
        weight_layout=config["e2e"]["linear_weight_layout"],
    )
    model = json.loads((Path(config["paths"]["model"]) / "config.json").read_text())
    hidden = model["hidden_size"]
    heads = model["num_attention_heads"]
    kv_heads = model.get("num_key_value_heads", heads)
    head_dim = hidden // heads
    input_tokens = config["e2e"]["input_tokens"]
    final_kv = input_tokens + config["e2e"]["output_tokens"] - 1
    results = []

    def check(name, actual, expected, atol):
        torch.npu.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0.025, atol=atol)
        item = {
            "case": name,
            "passed": True,
            "shape": list(actual.shape),
            "rtol": 0.025,
            "atol": atol,
            "max_abs_error": (actual.float() - expected.float()).abs().max().item(),
        }
        results.append(item)
        print(f"PASS {backend} {name}", flush=True)

    with torch.inference_mode():
        reference_sdpa = torch.nn.functional.scaled_dot_product_attention
        for query_length, kv_length, causal in (
            (input_tokens, input_tokens, True),
            (1, input_tokens + 1, False),
            (1, final_kv, False),
        ):
            q = torch.randn(
                (1, query_length, heads, head_dim), device="npu", dtype=torch.bfloat16
            ).transpose(1, 2)
            k = torch.randn(
                (1, kv_length, kv_heads, head_dim), device="npu", dtype=torch.bfloat16
            ).transpose(1, 2)
            v = torch.randn_like(k)
            expanded_k = k.repeat_interleave(heads // kv_heads, dim=1)
            expanded_v = v.repeat_interleave(heads // kv_heads, dim=1)
            expected = reference_sdpa(q, expanded_k, expanded_v, is_causal=causal)
            with attention.patched():
                actual = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    is_causal=causal,
                    enable_gqa=heads != kv_heads,
                )
            check(
                f"sdpa_q{query_length}_kv{kv_length}_causal{causal}",
                actual,
                expected,
                0.015,
            )

        q = torch.randn(
            (1, heads, input_tokens, head_dim), device="npu", dtype=torch.bfloat16
        )
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        mask = torch.ones(
            (input_tokens, input_tokens), device="npu", dtype=torch.bool
        ).tril()
        expected = reference_sdpa(q, k, v, attn_mask=mask)
        with attention.patched():
            actual = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask
            )
        check("sdpa_explicit_causal_mask", actual, expected, 0.015)

        shapes = {
            "q_proj": (hidden, hidden),
            "k_proj": (kv_heads * head_dim, hidden),
            "gate_proj": (model["intermediate_size"], hidden),
            "down_proj": (hidden, model["intermediate_size"]),
            "lm_head": (model["vocab_size"], hidden),
        }
        if not any(stage >= 6 for stage in config["e2e"]["stages"]):
            shapes = {}
        for name, (columns, reduction) in shapes.items():
            weight = torch.randn(
                (columns, reduction), device="npu", dtype=torch.bfloat16
            ) / math.sqrt(reduction)
            if config["e2e"]["linear_weight_layout"] == "packed":
                weight = weight.transpose(0, 1).contiguous().transpose(0, 1)
            for rows in (1, input_tokens):
                x = torch.randn(
                    (1, rows, reduction), device="npu", dtype=torch.bfloat16
                )
                expected = torch.nn.functional.linear(x, weight)
                actual = linear.linear(x, weight, module_name=name)
                check(
                    f"linear_{name}_m{rows}_k{reduction}_n{columns}",
                    actual,
                    expected,
                    0.08,
                )
            del weight, x, actual, expected

    return {
        "backend": backend,
        "passed": True,
        "checks": results,
        "attention": attention.provenance(),
        "linear": linear.provenance(),
        "policy": "BF16 tolerances match the CSV; references are native NPU model operations. Logits are not an acceptance gate.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--backend", required=True, choices=("triton", "ntops"))
    parser.add_argument("--device", required=True, type=int)
    args = parser.parse_args()
    config = runner.read_config(args.config)
    config["device"] = args.device
    result = validate(config, args.backend)
    runner.write_json(
        Path(config["output_dir"]) / f"core-validation-{args.backend}.json", result
    )


if __name__ == "__main__":
    main()
