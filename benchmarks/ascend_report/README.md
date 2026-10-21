# Ascend report measurements

This suite measures the CSV-aligned operator cases and configured stages of DeepSeek generation, then writes the data needed for the Word report. It uses the configured source checkouts and records their actual file hashes. Operator and model phases run sequentially to avoid competing with synchronized generation timing.

Use an Ascend Linux environment with CANN, PyTorch, torch_npu, Triton-Ascend, NumPy, SymPy and Transformers already installed. Load the CANN environment and activate that Python environment before running. The baseline checkout must contain `benchmarks/cases.py` and `benchmarks/run.py`; the validated baseline is `zebra-uestc/triton-ascend-ops` at `c35f31b027489517e9c745b2338e01832f718489`, with `ssen0829/ntops.lab` at `56f2df800e859f378b6d2f8f2bd57cc4660fb77e`.

Copy `config.example.json`, fill in source/model paths, a new result directory and a cache directory on a volume with enough free space. Set `devices` to the physical cards available to this run; the example uses four cards. Relative paths are resolved from the configuration file. Keep results outside source checkouts. The optional `case_csv` verifies case IDs, operators, dtypes, labels, layouts, input shapes, strides and parameters against the supplied CSV; previous CSV timing values are never used as new measurements. Model weights stay local and are fingerprinted before generation.

Run the complete measurement and statistics workflow:

```bash
python benchmarks/ascend_report/run_report.py --config /path/to/config.json
```

The operator phase measures both implementations on the same physical card for each case, with 50 warmups, 15 device samples and 60 public-API samples by default. Generated-kernel graph replay and public calls retain the original case reference checks and tolerances. Existing host layout/parameter adapters, the masked-load `Tensor.to` compatibility shim and explicit NPU launcher-name shim are recorded in every result. Unsupported source contracts stay `UNSUPPORTED`; no numerical fallback is substituted.

Operator groups are distributed across the configured cards. The model phase assigns stages to cards and runs one lane per card: a reference process and that card's stages, with both backends and both execution orders kept on the same physical card. CPU affinity is split into nonoverlapping groups by default and recorded; `cpu_affinity` can provide explicit per-card CPU lists. `operators.devices` or `e2e.devices` can override the global card list.

`e2e.stages` selects stages by number. The default and example run all six stages on four cards (four reference processes plus24 timed processes). Set `[1, 2, 3, 4]` explicitly to measure the earlier limited scope. Every selected stage runs both backend orders. The stages add replacements cumulatively:

1. SiLU and RMSNorm.
2. MLP gating Mul.
3. Floating TensorAdd, including RoPE and residual paths.
4. Neg, Sin and Cos.
5. SDPA for causal prefill and cached decode, preserving scale and grouped query attention.
6. All model `nn.Linear` projections and the language model head.

Stages5–6 call the pinned numeric kernels directly. Both backends preserve operand strides and avoid per-token full-weight copies; unsupported masks or biases raise instead of silently using native kernels. The NineToothed matrix path bypasses the repository's handwritten NPU fallback and retains its original FP32-to-FP16 intermediate rounding before BF16 storage. The framework still provides embedding, cache management, generation control and unpatched primitives, including RoPE frequency construction. These stages do not claim that every model operation has been replaced. Recorded counters distinguish native matrix operations from verified custom-kernel launches and verify that every patched Linear module executed.

Before timing stages5–6, the entry point validates representative model attention/Linear shapes against native NPU operations with the existing BF16 operator tolerances. This includes transposed prefill layouts, grouped KV heads, cached decode and the vocabulary projection. The validation processes finish before model timing begins. Raw validation outcomes are written beside the measurement results.

The model protocol defaults to BF16, batch1, 19 input tokens, 128 fixed greedy output tokens, two warmups and ten synchronized `generate()` samples. Loading, tokenization, text decoding, result checking and launch counting remain outside timing. Each sample's output text and token IDs are saved. Logits are optional diagnostics and never determine acceptance. Multi-card wall-clock results are labeled as concurrent measurements: compiler, CPU and memory resources are shared, so total runtime is not guaranteed to scale linearly with the card count.

Use `--resume` only with unchanged configuration and sources. It verifies hashes before reusing completed results and preserves failed model attempts. For a smoke test, use a separate output directory and `--phase operators --ops add`. `--phase e2e` runs model measurements only; `--phase summary` recomputes statistics from existing raw records.

Main outputs:

- `operator-comparison.csv`: all selected cases, statuses, timings and ratios.
- `operator-summary.json` and `.md`: coverage, sample audits, case geometric mean, operator-equal geometric mean, arithmetic mean and ratio of summed times. Incomplete full41/465 coverage leaves formal metrics unset.
- `operators/`: immutable per-path records, environment, source/script hashes, generated sources and logs.
- `e2e/report-data.json`: both-order model medians, relative throughput, output text checks and operator call counts.
- `e2e/results/`: raw samples, complete generated texts/tokens and runtime provenance.
- `report-data.json`: combined operator/model report inputs.
- `pipeline.json`: phase completion and script identity.

Relative operator performance is `100 × baseline_device_median / ninetoothed_device_median`; model relative throughput is `100 × baseline_generate_median / ninetoothed_generate_median`. Values above100 mean NineToothed is faster. These metrics are not interchangeable with ratios of aggregated times or arithmetic averages.

Normal coherent text is the report's end-to-end acceptance criterion. The suite checks mechanical validity and preserves the text for review; token or logits differences alone do not fail acceptance. After reviewing each timed output, write `e2e/semantic-review.json` with records keyed by the raw result's relative path:

```json
{"records":{"results/triton_then_ntops_stage1_ntops.json":{"raw_sha256":"<exact raw JSON SHA-256>","passed":true,"reviewer":"<reviewer>"}}}
```

Then run `run_report.py --config /path/to/config.json --phase summary`. Reviews are bound to raw hashes and cannot silently approve changed results. Performance measurements must be rerun after compiler, operator, model, scheduling or measurement changes; correctness-only results must not be substituted for timings.
