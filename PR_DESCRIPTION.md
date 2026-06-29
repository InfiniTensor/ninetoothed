## 1. Skill 名称、赛题编号和小组名称

- **Skill 名称**：ntops-forge（主）+ ntops-copilot（辅）
- **赛题编号**：T3-1-1
- **小组名称**：于鸿伟

**独立 skill 仓库**：https://github.com/hongwei-2026/qiyuanbisai  
**Commit**：https://github.com/hongwei-2026/qiyuanbisai/commit/750db19

## 2. 适用任务范围与不适用范围

**适用**：ntops elementwise 算子（silu/add/gelu/relu/mul）；2026-spring PR 规范；GPU + editable ntops。

**不适用**：Triton `@triton.jit`；norm/attention 需先读 reference；无 CUDA。

## 3. 安装与使用方式

```bash
source /root/miniconda3/bin/activate base
pip install pytest && pip install -e /path/to/ntops
python skills/competition/ntops-forge/scripts/forge.py gate --ntops-root /path/to/ntops
```

Cursor：将 `skills/competition/ntops-forge/` 链到 `.cursor/skills/ntops-forge/`。

## 4. 自测运行记录

| 记录 | 路径 |
|------|------|
| GPU 测试报告 | `skills/competition/ntops-forge/docs/GPU_Test_Report.md` |
| A/B 报告 | `skills/competition/ntops-forge/docs/AB_Report.md` |
| 四类自测案例 | `skills/competition/ntops-forge/docs/selftests/`（ST1–ST4） |
| ST2/ST3 结果 | `skills/competition/ntops-forge/docs/st2_st3_gpu_results.md` |
| 审计日志 | 独立仓库 `docs/forge_runs.jsonl`、`docs/ab_runs.csv` |

**环境**：AutoDL · RTX 4080 · `pip install -e /root/work/ntops` · 工作目录 `/root/work/skill`

**ST1–ST4 摘要**：

| 编号 | 类型 | 状态 |
|------|------|------|
| ST1 | 逐元素/广播（silu/add/gelu/relu/mul） | ✅ `forge gate` 五算子全通过 |
| ST2 | 归约/分块（softmax） | ✅ `test_softmax.py` 8 passed |
| ST3 | 布局 stride（max_pool2d） | ✅ `test_max_pool2d.py` 62 passed, 54 skipped |
| ST4 | 性能/诊断 | ✅ A/B + silu benchmark + fix_cards |

## 5. 自测结果（有 skill vs 无 skill）

见 `skills/competition/ntops-forge/docs/AB_Report.md`：

| 指标 | Baseline（无 skill） | Treatment（有 skill） |
|------|---------------------|----------------------|
| preflight 通过率 | 0% | 100% |
| pytest 通过率 | 未跑通 | 100% |
| 平均步骤 | 6 | 1 |
| 人工介入 | 4 次 | 0 次 |
| 平均耗时 | ~1200s | ~7s/算子 |

**GPU forge gate（五算子）**：

| 算子 | pytest | 耗时 |
|------|--------|------|
| silu | 8 passed | 6.7s |
| add | 8 passed | 6.9s |
| gelu | 8 passed, 8 skipped | 7.0s |
| relu | 16 passed | 10.9s |
| mul | 8 passed | 12.5s |

**GATE OK: all operators passed**

## 6. HONOR_CODE 与 REFERENCE

- 本 PR 根目录 [`HONOR_CODE.md`](HONOR_CODE.md)
- 本 PR 根目录 [`REFERENCE.md`](REFERENCE.md)

## 7. Proposal 与报告

| 材料 | 位置 |
|------|------|
| Proposal | 独立仓库 `docs/Proposal.md` |
| 自测计划 | 独立仓库 `docs/SelfTestPlan.md` |
| 中期报告 | 独立仓库 `docs/MidTermReport.md`（PDF：`于鸿伟_九齿skill创新挑战_中期报告.pdf`） |
| 初赛 zip | `submission-initial-20260608.zip` |

---

## pytest output（CONTRIBUTING 要求）

### ninetoothed 仓库 pytest

本 PR **仅新增** `skills/competition/`，**未修改** `src/ninetoothed/`。上游 `master` 已在 NVIDIA self-hosted runner 通过 pytest（[#158](https://github.com/InfiniTensor/ninetoothed/pull/158)）。

### ntops skill 验收（GPU · RTX 4080 · 2026-06-08）

```text
$ python skills/competition/ntops-forge/scripts/forge.py gate --ntops-root /root/work/ntops
GATE OK: all operators passed (silu, add, gelu, relu, mul)

$ cd /root/work/ntops && pytest tests/test_softmax.py -q
8 passed

$ cd /root/work/ntops && pytest tests/test_max_pool2d.py -q
62 passed, 54 skipped in ~175s
# 54 skipped: upstream Invalid padding（非失败）
```
