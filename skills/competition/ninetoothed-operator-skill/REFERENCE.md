# References and Disclosure

## Official materials
- NineToothed repository: https://github.com/InfiniTensor/ninetoothed
- NineToothed examples: https://github.com/InfiniTensor/ninetoothed-examples
- NineToothed CONTRIBUTING.md
- Competition rule document: 九齿 .skill 创新挑战赛道赛题与规则 T3-1-1

## External references
- Triton language documentation: https://triton-lang.org/
- PyTorch operator reference implementations used for correctness tests

## Third-party code
- Operator patterns (arrangement/application structure) adapted from
  InfiniTensor/ninetoothed-examples (Apache 2.0 License).

Specific reference files and usage:

| Source | Referenced content | Usage in this submission |
|---|---|---|
| `ninetoothed-examples/ops/ninetoothed/kernels/element_wise.py` | Generic elementwise arrangement style | ReLU arrangement guidance and tests |
| `ninetoothed-examples/ops/ninetoothed/kernels/softmax.py` | Row-wise reduction / online softmax pattern | Softmax self-test operator and benchmark case |
| `ninetoothed-examples/ops/ninetoothed/kernels/rms_norm.py` | Row-wise normalization pattern | RMSNorm self-test and failure diagnosis |
| `ninetoothed-examples/ops/ninetoothed/kernels/reduction.py` | Generic reduction arrangement concepts | SKILL.md reduction guidance |
| `ninetoothed-examples/tests/` | PyTorch reference comparison style | Correctness test structure and tolerances |
| `ninetoothed-examples/bench/` | Benchmark organization | CUDA-event benchmark records and README guidance |

Modifications and original work:
- Reorganized the patterns into a reusable AI-agent workflow in `SKILL.md`.
- Split detailed templates and diagnostics into `references/operator-patterns.md`,
  `references/v0.26-known-failures.md`, and
  `references/benchmarking-and-diagnostics.md` to keep the main skill concise.
- Added self-test wrappers, failure diagnosis notes, and Colab validation records.
- Added explicit limitations for fixed RMSNorm `eps`, non-contiguous fallback,
  generated-source dump scope, unvalidated AOT workflow, and unsupported GEMM scope.

## Generative AI assistance
AI tools used:
- Codex (OpenAI): assisted with skill structure design, debugging NineToothed
  kernel errors, writing documentation, and benchmark analysis

AI assistance scope:
- Diagnosed and fixed kernel compilation errors (Tensor(0) pointer type mismatch,
  negative index constexpr evaluation, Symbol closure capture issues)
- Drafted SKILL.md, references, and examples documentation
- Helped analyze benchmark results and write self-test records

All generated content was reviewed and validated by the participant before submission.
Operator code was verified by running pytest in Google Colab (T4 GPU).
