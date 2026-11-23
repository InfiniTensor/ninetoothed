# ninetoothed-operator-skill

Guide AI agents to develop, test, benchmark, debug, and integrate NineToothed operators.

**赛题编号：** T3-1-1
**参赛者：** 钱泓林 (qhl18)

---

## Environment

**Windows 11 + NVIDIA RTX 5060 Laptop GPU (8GB) + PyTorch 2.12.0.dev20260408+cu128 + ninetoothed 0.26.0**

All pytest tests and benchmarks have been executed on the above GPU environment. No fabricated results.

| Component | Status |
|-----------|--------|
| pytest | All tests PASSED on RTX 5060 + CUDA 12.8 |
| benchmark | Results collected on RTX 5060 |
| Operator code | Compiled and numerically verified |

---

## Scope

- Elementwise/Broadcast operators (e.g. add, relu)
- Reduction/Block operators (e.g. softmax, reduce)
- Layout-Sensitive operators (e.g. non-contiguous, stride, offset)
- Benchmark/Debug tasks (e.g. benchmark, generated source, AOT build, failing test diagnosis)

## Out of Scope

- Dynamic shape (requires shape guard)
- bfloat16 / float64 (not fully validated)
- Modifying NineToothed compiler core
- Multi-GPU / distributed
- Hidden evaluation answers or hardcoded outputs

## Package Structure

```text
ninetoothed-operator-skill/
├── SKILL.md              # Core workflow for AI agents
├── README.md             # This file
├── HONOR_CODE.md         # Honor code declaration
├── REFERENCE.md          # Reference disclosure
├── .gitignore
├── examples/             # T1-T4 self-test tasks
│   ├── task-01/          # Elementwise/Broadcast add
│   ├── task-02/          # Reduction softmax
│   ├── task-03/          # Layout transpose_add
│   └── task-04/          # Benchmark/Debug
├── references/           # Repository reference and guides
│   ├── index.md          # Repo structure index
│   ├── testing.md        # Test patterns
│   ├── benchmarking.md   # Benchmark and AOT guide
│   └── debugging.md      # Failure diagnosis guide
├── scripts/              # Self-test and log collection scripts
├── tests/                # Skill verification
└── reports/              # Competition report
```

## Self-test Tasks

| ID | Type | Task | Correctness | Benchmark |
|----|------|------|-------------|-----------|
| T1 | Elementwise/Broadcast | add with broadcast | PASSED | Optional |
| T2 | Reduction/Block | softmax | PASSED | Yes |
| T3 | Layout-Sensitive | transpose_add | PASSED | No |
| T4 | Benchmark/Debug | generated source check | PASSED | Yes |

## Installation and Usage

### 1. Install NineToothed

```bash
git clone https://github.com/InfiniTensor/ninetoothed.git
cd ninetoothed
pip install -e .
```

### 2. Load skill

In Cursor / Trae / Codex or other AI tools, ask the AI to read:

```text
skills/competition/ninetoothed-operator-skill/SKILL.md
```

### 3. Run self-tests

```bash
cd skills/competition/ninetoothed-operator-skill
bash scripts/run_self_tests.sh
python scripts/collect_logs.py
```

### 4. Pre-submission checklist

- `HONOR_CODE.md`: Signed honor code
- `REFERENCE.md`: References and AI assistance disclosed
- `reports/`: Competition report complete

## Skill vs No-Skill Comparison

| Dimension | Without skill | With skill |
|-----------|---------------|------------|
| Testing | Often missing or non-standard | Fixed PyTorch reference + GPU verified |
| Layout | Often only tests contiguous | Forces non-contiguous/stride/offset |
| Benchmark | Often missing | Fixed warmup/sync/baseline/ratio |
| Diagnosis | Often large changes, no records | Classified debugging + documented loop |
| Compliance | Easy to miss disclosures | REFERENCE + HONOR_CODE templates |
| AOT/Source | Often ignored | SKILL.md references benchmarking.md |

## Language

Documentation is primarily in Chinese with key technical terms in English.
