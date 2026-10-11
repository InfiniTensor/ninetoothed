# NineToothed Operator Skill

A reusable `.skill` package that guides AI coding agents (Trae, Cursor, Codex, etc.) to reliably develop, test, debug, benchmark, and integrate NineToothed GPU operators.

## Competition

- **Event**: 2026 Spring AI Contest
- **Track**: NineToothed .skill Innovation Challenge
- **Problem ID**: T3-1-1
- **Participant**: Zhao Junkai

## Scope

This skill covers the following operator development scenarios:

| Category | Examples | Key Concerns |
|---|---|---|
| Elementwise / Broadcast | Add, ReLU, GELU, masked add | dtype, broadcast rules, mask constraints, boundary conditions |
| Reduction / Block | Softmax, reduce sum, block max | tile/block design, local reduction, cross-block merge, numerical stability |
| Layout-Sensitive | Non-contiguous inputs, transpose, stride, offset | stride handling, contiguous vs non-contiguous correctness and performance |
| Benchmark / Debug | Generated source inspection, AOT build, failing test diagnosis | benchmark methodology, failure root cause analysis, minimal fix, verification closure |

## Out of Scope

- Network-dependent operations or online API calls
- Any task requiring API keys or private credentials
- Hardcoded solutions for specific hidden evaluation tasks
- Operators outside the NineToothed/Triton GPU kernel domain
- CPU execution (NineToothed targets GPU via Triton)
- Dynamic shapes, empty tensors, float64, multi-GPU
- See SKILL.md §"Supported and Unsupported Scenarios" for full details

## Package Structure

```
skills/competition/ninetoothed-operator-skill/
├── SKILL.md                 # Core workflow — read by AI agents
├── README.md                # This file — read by humans
├── references/
│   └── index.md             # Repository map and development lookup index
├── scripts/
│   ├── env_check.py         # Environment dependency checker
│   └── run_selftests.py     # Automated self-test runner
├── examples/                # Self-test tasks (4 tasks)
│   ├── task1_elementwise_broadcast/
│   │   ├── operator_impl.py
│   │   ├── test_correctness.py
│   │   ├── benchmark.py
│   │   └── task_description.md
│   ├── task2_reduction_block/
│   │   ├── operator_impl.py
│   │   ├── test_correctness.py
│   │   ├── benchmark.py
│   │   └── task_description.md
│   ├── task3_layout_sensitive/
│   │   ├── operator_impl.py
│   │   ├── test_correctness.py
│   │   ├── test_layout_compare.py
│   │   ├── benchmark.py
│   │   └── task_description.md
│   └── task4_benchmark_debug/
│       ├── benchmark_multisize.py
│       ├── buggy_operator.py
│       ├── diagnosis_record.md
│       └── task_description.md
├── tests/
│   └── test_skill_structure.py   # Structural validation tests
├── reports/
│   └── T3-1-1_赛题报告.md        # Final competition report
├── HONOR_CODE.md
└── REFERENCE.md
```

## Current Status

> **✅ 状态**: T1/T2/T3 已于 2026-07-11 在 Tesla T4 (CUDA 12.8, PyTorch 2.11.0, NineToothed 0.26.0) 通过全部测试。T4 benchmark 脚本已完成，4A/4C 子任务代码已就位。

| ID | Type | Task | Code | Test | Benchmark | GPU Verified |
|---|---|---|---|---|---|---|
| T1 | Elementwise / Broadcast | Masked Add with Broadcast | ✅ | ✅ | ✅ | ✅ 12/12 PASSED |
| T2 | Reduction / Block | Tiled Softmax | ✅ | ✅ | ✅ | ✅ 13/13 PASSED |
| T3 | Layout-Sensitive | Non-Contiguous GELU | ✅ | ✅ | ✅ | ✅ 12/12 PASSED |
| T4 | Benchmark / Debug | Source + Regression + Bug | ✅ | ✅ | ✅ | ⬜ 脚本就位，待执行 |

## How to Use

1. Open the NineToothed repository in an AI coding tool.
2. Ask the AI agent to read `SKILL.md` before starting any operator task.
3. Follow the phased workflow: Understand → Search → Implement → Test → Benchmark → Diagnose → Summarise.

## How to Run Self-Tests

```bash
# 1. Check environment
python skills/competition/ninetoothed-operator-skill/scripts/env_check.py

# 2. Run structural validation (no GPU needed)
pytest skills/competition/ninetoothed-operator-skill/tests/test_skill_structure.py -v

# 3. Run correctness tests (requires GPU + NineToothed)
python skills/competition/ninetoothed-operator-skill/scripts/run_selftests.py

# Or run individually:
pytest skills/competition/ninetoothed-operator-skill/examples/task1_elementwise_broadcast/test_correctness.py -v
```

## Constraints (Red Lines)

- No external network dependencies at runtime
- No API keys, private credentials, or account secrets
- No hardcoded hidden evaluation task data
- No modification to existing repository files outside `skills/competition/`
- No falsified test or benchmark outputs
