# REFERENCE.md

Upstream citations, version pins, licenses, and AI-assistance disclosure for the
**runtime** skill package. Competition studies, workspace gate logs, and local
path pins live in the external evidence bundle linked in the PR description —
not in this runtime tree.

## Upstream projects

| Project | URL | Notes |
|---------|-----|-------|
| NineToothed | https://github.com/InfiniTensor/ninetoothed | Target `--repo-root` (contains `src/ninetoothed/`) |
| NineToothed docs | https://ninetoothed.org/ | Concepts, AOT / build |
| ninetoothed-examples | https://github.com/InfiniTensor/ninetoothed-examples | Optional `--examples-root` |
| ntops | https://github.com/InfiniTensor/ntops | Related ops library |

## Version pins (reproducibility)

Pin the fork you develop against; re-verify with `env_check.py --repo-root .`.

| Component | Suggested pin |
|-----------|----------------|
| Python | 3.10+ |
| PyTorch | CUDA build matching the GPU (e.g. cu128 nightly for sm_120) |
| pytest | recent 8.x / 9.x |
| ninetoothed | editable install of the fork at a known commit |
| ninetoothed-examples | optional; known commit if used |

```bash
cd <repo-root>
python -c "import torch, ninetoothed; print(torch.__version__, ninetoothed.__version__)"
git rev-parse HEAD
python /path/to/ninetoothed-op-dev-skill/scripts/env_check.py --repo-root .
```

## Licenses

| Component | License | Notes |
|-----------|---------|-------|
| This skill package | See competition / repo license | No hidden answers or credentials |
| NineToothed (`<repo-root>`) | Apache-2.0 | Prefer patches outside compiler core |
| ninetoothed-examples (`--examples-root`) | Apache-2.0 | Optional reference / bench only |

## AI assistance disclosure

Drafting of `SKILL.md`, `references/`, helper scripts, reports, and packaging
checklists used generative AI assistants (**Cursor** and **Codex / GPT-family**
tools) under human direction. The submitter remains responsible for factual
accuracy, reproducible commands, and honest benchmark claims. Packaged
`examples/` are workflow demos, not evaluation answer keys.

## Agent skill conventions

- Cursor Agent Skills / Open Agent Skills layout (`SKILL.md` + `references/` + `scripts/`)
