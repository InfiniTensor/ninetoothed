# References and External Sources

## NineToothed Framework
- Repository: https://github.com/InfiniTensor/ninetoothed
- License: Apache-2.0
- Baseline version: master branch (commit tracked at time of development)

## Triton
- Triton Language Reference: https://triton-lang.org/main/index.html
- Used for: GPU kernel compilation and execution
- Version: as specified by NineToothed requirements

## Academic References

1. Tillet, P., Kung, H. T., & Cox, D. (2019). "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations." MAPS@PLDI 2019.
   - Referenced for: Understanding tiling abstractions and code generation patterns

2. NineToothed: A Tensor-Oriented Meta-Programming DSL Based on Triton
   - InfiniTensor Open Source Community
   - Referenced for: Understanding the arrange-and-apply paradigm, tensor hierarchy, and AOT compilation

## AI-Assisted Development

- **Claude Code (Anthropic, Claude Opus 4.8)**: Used for codebase exploration, weakness identification, implementation design, and test/benchmark scaffolding.
  - Usage scope: All modified files disclosed in PR
  - Human review: All AI-suggested code was reviewed for correctness and merged manually

## Third-Party Tools

- Python standard library (`ast`, `dataclasses`, `pathlib`, `re`, `json`, `time`)
- PyTorch (for tensor operations and GPU utilities)
- Triton (for GPU kernel compilation)
- pytest (for test execution)
- sympy (for symbolic inequality reasoning in auto-tuning)

## No External Dependencies Introduced

Our modifications use only standard library `dataclasses` (which is built into Python 3.7+) and existing NineToothed dependencies. No new packages are required.
