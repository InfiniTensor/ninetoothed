# References and Disclosure

## Official Materials

- NineToothed repository: https://github.com/InfiniTensor/ninetoothed
- NineToothed CONTRIBUTING.md
- NineToothed Examples: https://github.com/InfiniTensor/ninetoothed-examples
- Competition rule document: 九齿 .skill 创新挑战赛道赛题与规则 v0.8
- NineToothed Documentation: https://ninetoothed.org/

## External References

- PyTorch documentation: https://pytorch.org/docs/stable/ — reference implementations for softmax, GELU, and elementwise ops
- Triton documentation: https://triton-lang.org/ — understanding generated IR and GPU kernel optimisation
- GELU activation paper: Hendrycks & Gimpel (2016), "Gaussian Error Linear Units (GELUs)" — tanh approximation formula
- Online softmax algorithm: Milakov & Gimelshein (2018), "Online Normalizer Calculation for Softmax" — numerical stability reference

## Third-Party Code

- None. All operator implementations are original code following NineToothed's documented patterns. No third-party kernel code was reused.

## Generative AI Assistance

AI tools used:

- Claude Code (Anthropic Claude): assisted throughout the project

AI assistance scope:

- Drafted documentation structure (SKILL.md, README.md, references/index.md)
- Suggested self-test task designs and test case parameterisation
- Generated code drafts for operator implementations, test files, and benchmark scripts — all reviewed and modified by the participant
- Helped structure the final competition report and compliance checklist
- Assisted with organising the failure diagnosis protocol and common pitfalls reference

All generated content was reviewed by the participant before submission. Code correctness was verified against PyTorch reference implementations.
