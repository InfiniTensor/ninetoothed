# PR 描述（提交到 InfiniTensor/ninetoothed 时使用）

> **赛题**：T3-1-1 · **选手**：于鸿伟（hongwei-2026）  
> **独立 skill 仓库**：https://github.com/hongwei-2026/qiyuanbisai  
> **Commit**：https://github.com/hongwei-2026/qiyuanbisai/commit/e7b32bb

---

## 1. Skill 名称、赛题编号和小组名称

- **Skill 名称**：ntops-forge（主）+ ntops-copilot（辅）
- **赛题编号**：T3-1-1
- **小组名称**：于鸿伟

## 2. 适用任务范围与不适用范围

**适用**：ntops elementwise 算子（silu/add/gelu/relu/mul）；2026-spring PR 规范；GPU + editable ntops。

**不适用**：Triton `@triton.jit`；norm/attention 需先读 reference；无 CUDA。

## 3. 安装与使用方式

```bash
source /root/miniconda3/bin/activate base
pip install pytest && pip install -e /path/to/ntops
# 使用本 PR 内路径
python skills/competition/ntops-forge/scripts/forge.py gate --ntops-root /path/to/ntops
```

Cursor：将 `skills/competition/ntops-forge/` 链到 `.cursor/skills/ntops-forge/`。

## 4. 自测运行记录

- `docs/GPU_Test_Report.md`（本 PR 内 `skills/competition/ntops-forge/docs/`）
- `docs/forge_runs.jsonl`（独立仓库）
- 四类自测：`docs/selftests/ST1–ST4`

## 5. 自测结果（有 skill vs 无 skill）

见 `docs/AB_Report.md`：preflight 0%→100%，步骤 6→1，人工介入 4→0，A/B 5v5。

GPU 验收：五算子 gate OK；softmax 8 passed；max_pool2d 62 passed。

## 6. HONOR_CODE 与 REFERENCE

- 本 PR 根目录 `HONOR_CODE.md`、`REFERENCE.md`

## 7. Proposal 与报告

- Proposal / 中期报告：见独立仓库 `docs/` 与 zip 附件

---

## pytest output（CONTRIBUTING 要求）

### ninetoothed 仓库 pytest（本 PR 未改 src/，应全绿）

```
（在 fork 的 ninetoothed 根目录执行 pytest 后粘贴）
```

### ntops skill 验收（补充）

```
forge gate: GATE OK (silu/add/gelu/relu/mul)
test_softmax.py: 8 passed
test_max_pool2d.py: 62 passed, 54 skipped
```
