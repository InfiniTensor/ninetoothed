# References and Disclosure

## Official materials

- NineToothed repository: <https://github.com/InfiniTensor/ninetoothed>
- NineToothed CONTRIBUTING.md (in-repo).
- NineToothed documentation: <https://ninetoothed.org/>
- NineToothed operators (ntops): <https://github.com/InfiniTensor/ntops>
- NineToothed examples: <https://github.com/InfiniTensor/ninetoothed-examples>
- Competition rule document: 九齿 .skill 创新挑战赛道赛题与规则 v0.8.

## In-repo sources studied

The skill's references and workflow were derived by reading the actual repository
(version 0.26.0), specifically:

- `README.md`, `docs/source/basics.rst`, `docs/source/build.rst`
- `tests/test_add.py`, `tests/test_softmax.py`, `tests/test_max_pool2d.py`,
  `tests/test_matmul.py`, `tests/test_getitem.py`, `tests/utils.py`, `tests/conftest.py`
- `src/ninetoothed/__init__.py`, `make.py`, `tensor.py`, `language.py`,
  `debugging.py`, `build.py`
- `CONTRIBUTING.md`, `scripts/check_contributing_style.py`

## Third-party code

- None reused verbatim. The example operators follow the structure of the repo's
  own test operators (Apache-2.0) as permitted for repo contributions.

## Generative AI assistance

AI tools used:

- An AI coding assistant was used to study the repository, draft the skill
  documents, author the example operators and tests, and organize the reports.

AI assistance scope:

- Drafted the SKILL.md workflow, references, and scripts.
- Authored the four self-test operators and tests, following repo patterns.
- Helped summarize logs and structure records.

All generated content was reviewed, and all correctness results were executed on
real hardware before being reported.

## Environment for reported results

- OS: WSL2 Ubuntu on Windows 11.
- GPU: NVIDIA GeForce RTX 5060 (8 GB).
- CUDA: 13.0 (driver 582.05).
- Python: 3.14.
- torch 2.13.0+cu130, triton 3.7.1, ninetoothed 0.26.0 (editable install).
