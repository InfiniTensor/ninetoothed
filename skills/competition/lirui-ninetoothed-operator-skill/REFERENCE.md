# References and AI Assistance Disclosure

## Competition materials

- InfiniTensor 2026 春季赛官方页面: https://www.infinitensor.com/contest/spring2026#九齿-skill-创新挑战赛道
- `九齿 .skill 创新挑战赛道赛题与规则.pdf`: local competition rule document provided by the participant.
- `NineToothed_skill_创新挑战赛操作手册.docx`: local operation manual provided by the participant. The manual states that the final work should be submitted in a NineToothed PR under `skills/competition/<skill-name>/`.

## NineToothed repository

- InfiniTensor/ninetoothed main repository: https://github.com/InfiniTensor/ninetoothed
- Repository clone URL from operation manual: https://github.com/InfiniTensor/ninetoothed.git
- `README.md`: NineToothed overview and arrange-and-apply example.
- `CONTRIBUTING.md`: branch naming, commit message, PR title, pytest output, Ruff, and style checker requirements.
- `docs/`: NineToothed basics, Python API, visualization, and AOT build documentation.
- `docs/source/basics.rst`: symbolic tensors, tensor-oriented metaprogramming, meta-operations, and `ninetoothed.make` examples.
- `docs/source/build.rst`: `ninetoothed.build`, `premake`, `configs`, `meta_parameters`, AOT artifacts, `.so`, and auto-tuning CSV behavior.
- `tests/`: local operator patterns and correctness tests.
- `tests/test_addmm.py`: scalar and matrix arguments with PyTorch reference.
- `tests/test_softmax.py`: reduction-style pattern using `ntl.max`, `ntl.sum`, and `Tensor(..., other=float("-inf"))`.
- `tests/test_aot.py`: AOT examples using `caller`, `kernel_name`, and `output_dir`.
- `tests/test_aot_auto_tuning.py`: `ninetoothed.build`, `premake`, `configs`, and `meta_parameters`.
- `src/ninetoothed/aot.py`: AOT generated C++ dispatcher, launch wrapper, and compilation flow.
- `src/ninetoothed/build.py`: ahead-of-time multi-variant build, auto-tuning, CSV cache, and fingerprint logic.
- `src/ninetoothed/generation.py`: generated source cache and code-generation entry points.

## AI tools used

Codex / ChatGPT was used to assist with:

- Reading and summarizing the NineToothed repository structure.
- Extracting development patterns from `README.md`, `CONTRIBUTING.md`, `docs/`, `tests/`, and `src/ninetoothed/`.
- Drafting and strengthening this `.skill` package structure and documentation.
- Drafting workflow checklists, task templates, validation plan, honor code, and reference disclosure.

AI assistance did not include:

- Hidden evaluation answer generation.
- Hidden task-name hard-coding.
- API key usage.
- Credential usage.
- Test result fabrication.
- Benchmark result fabrication.
