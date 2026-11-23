# Repository Reference Index

## Repo map

- `README.md`: introduces NineToothed as a Triton-based DSL using tensor-oriented meta-programming and the arrange-and-apply paradigm.
- `CONTRIBUTING.md`: contribution workflow, branch naming, commit message, PR title, pytest output, Ruff, and project-specific style checks.
- `docs/source/basics.rst`: symbols, symbolic tensors, tensor meta-operations, `arrangement`, `application`, and `ninetoothed.make`.
- `docs/source/build.rst`: ahead-of-time build workflow with `ninetoothed.build`, `premake`, `configs`, `meta_parameters`, generated files, `.so`, and auto-tuning CSV.
- `src/ninetoothed/`: core DSL and compiler implementation.
- `tests/`: operator-like examples and correctness tests.

There is no root-level examples directory in the repository snapshot inspected for this skill. Existing operator examples are primarily in `tests/` and `docs/`.

## Key DSL files

- `src/ninetoothed/tensor.py`: symbolic `Tensor`, shape symbols, and meta-operations such as `tile`, `expand`, `unsqueeze`, `squeeze`, `permute`, `flatten`, `ravel`, and `pad`.
- `src/ninetoothed/make.py`: connects `arrangement`, `application`, and `tensors`; dispatches to JIT or AOT depending on `caller`.
- `src/ninetoothed/generation.py`: generated Triton source construction and cache source handling.
- `src/ninetoothed/jit.py`: JIT integration.
- `src/ninetoothed/aot.py`: AOT code generation, C++ dispatcher, compilation, and launch wrapper.
- `src/ninetoothed/build.py`: multi-config AOT build, auto-tuning, generated dispatcher, CSV cache, and fingerprint handling.
- `src/ninetoothed/auto_tuner.py`: benchmark timing with `triton.testing.do_bench`.
- `src/ninetoothed/debugging.py`: `simulate_arrangement` for debugging tensor arrangements.

## Test examples

- `tests/test_matmul.py`: tiled matmul pattern with `tile`, `expand`, `squeeze`, `ntl.dot`, and float32 accumulator.
- `tests/test_conv2d.py`: layout transformation pattern that lowers convolution-like access into a matmul arrangement using `pad`, `tile`, `squeeze`, `ravel`, `flatten`, and `permute`.
- `tests/test_addmm.py`: scalar and matrix arguments with PyTorch reference.
- `tests/test_attention.py`: block attention-like pattern.
- `tests/test_aot.py`: AOT calls with `caller`, `kernel_name`, and `output_dir`.
- `tests/test_aot_auto_tuning.py`: `ninetoothed.build`, `premake`, `configs`, `meta_parameters`, and auto-tuning.
- `tests/test_generation.py`: generated source behavior and basic kernel generation cases.
- `tests/test_jagged.py`: jagged tensor patterns.
- `tests/utils.py`: device discovery for CUDA and MLU.
- `tests/conftest.py`: deterministic seeding per module and test.

## AOT, build, and generated source references

- Use `ninetoothed.make(..., caller="cuda", kernel_name=..., output_dir=...)` for AOT-style generated artifacts.
- Use `ninetoothed.build(premake, configs, meta_parameters=..., caller=..., kernel_name=..., output_dir=...)` for ahead-of-time multi-variant builds.
- `output_dir` must already exist for `ninetoothed.build`.
- Generated artifacts can include `.cpp`, `.h`, `.so`, `.csv`, and `.fingerprint`.
- Auto-tuning benchmarks meta variants and records winners in a CSV next to the `.so`.
- `src/ninetoothed/generation.py` defines the default generated source cache as `~/.ninetoothed`.

## CONTRIBUTING checklist

- Branch name: kebab-case, lowercase letters, numbers, and hyphens only; maximum 50 characters.
- Commit message and PR title: first letter capitalized, imperative mood, no trailing punctuation.
- PR description: must include a non-empty `pytest` output code block.
- Run:

```bash
ruff format
ruff check
python scripts/check_contributing_style.py
pytest
```

- Full local CI sequence recommended by `CONTRIBUTING.md`:

```bash
python scripts/check_contributing_style.py --fix
ruff format
ruff check
python scripts/check_contributing_style.py
pytest
```

- For this competition package, keep changes under `skills/competition/lirui-ninetoothed-operator-skill/`.


