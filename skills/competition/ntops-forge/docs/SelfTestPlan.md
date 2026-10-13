# 自测计划（九齿 .skill 创新挑战 · 初赛）

| 项目 | 内容 |
|------|------|
| 组名/选手 | 于鸿伟 |
| GitHub | hongwei-2026 |
| 仓库 | https://github.com/hongwei-2026/qiyuanbisai |
| 日期 | 2026-06-08 |

---

## 1. 自测目标

验证 skill 能否让 Agent **更少犯错、更快完成** ntops 九齿算子开发，并形成可复现、可量化的 A/B 证据。

---

## 2. 自测环境

| 环境 | 用途 | 配置 |
|------|------|------|
| **GPU 主环境** | 正确性结论 | AutoDL，RTX 4080，conda `base` |
| **CPU 对照** | 安装/结构检查 | 华为 CPU 机（pytest 因无 CUDA 跳过） |
| **本地 Windows** | 脚本/preflight | 无 triton，不跑 ntops pytest |

工作目录（GPU）：`/root/work/skill`，`/root/work/ntops`

---

## 3. 自测任务集（对照赛题 4.2 四类）

### 3.1 逐元素 / 广播类 ✅ 初赛已完成

| 算子 | 类型 | pytest 文件 | 状态 |
|------|------|-------------|------|
| silu | unary | `tests/test_silu.py` | ✅ 8 passed |
| add | binary | `tests/test_add.py` | ✅ 8 passed |
| gelu | unary | `tests/test_gelu.py` | ✅ 8 passed |
| relu | unary | `tests/test_relu.py` | ✅ 16 passed |
| mul | binary | `tests/test_mul.py` | ✅ 8 passed |

示例：`skills/ntops-forge/examples/silu_walkthrough.md`

### 3.2 归约 / 分块类 📋 决赛计划

| 算子 | 类型 | pytest | spec | 状态 |
|------|------|--------|------|------|
| softmax | reduction | `tests/test_softmax.py` | `specs/softmax.yaml` | ✅ ST2 **8 passed** GPU |

策略：taxonomy 对 `reduction` family 禁止公式硬注入，先读 `src/ntops/kernels/softmax.py`。

### 3.3 布局敏感（stride / padding）📋 决赛计划

| 算子 | 场景 | pytest | spec | 状态 |
|------|------|--------|------|------|
| max_pool2d | stride=(None,1,(2,3)) | `tests/test_max_pool2d.py` | `specs/max_pool2d.yaml` | ✅ ST3 **62 passed** GPU |

### 3.4 性能 / 诊断类

| 任务 | 初赛 | 决赛 | 文档 |
|------|------|------|------|
| 流水线耗时 A/B | ✅ 5v5 完成 | 维持 | `docs/AB_Report.md` |
| silu kernel 计时 | ✅ 0.052/0.072 ms | 扩展多 shape | `bench_silu.json`、图17 |
| 失败诊断 | ✅ fix_cards | 扩展 | `skills/ntops-forge/fix_cards.md` |

### 3.5 任务卡 / forge spec

- `skills/ntops-copilot/tasks/task_*.yaml`（5 张）
- `skills/ntops-forge/specs/*.yaml`（7 张：5 已完成 + 2 决赛规划）
- 命令：`forge.py run <op>` 或 `run_task.py --finish`

决赛路线图：`docs/FinalsRoadmap.md`

---

## 4. 自测流程（forge 主路径）

```bash
source /root/miniconda3/bin/activate base
cd /root/work/skill
python scripts/doctor.py
bash scripts/run_ab_manual.sh /root/work/ntops
```

五段流水线：**PLAN → CODEGEN → GUARD → PROVE → SHIP**

GUARD：`preflight --strict` + `compare_ref matches reference`

---

## 5. 量化指标

| 指标 | Baseline（无 skill） | Treatment（有 skill） | 记录方式 |
|------|---------------------|----------------------|----------|
| preflight 通过率 | **0%**（5 算子） | **100%**（5 算子） | `ab_runs.csv` |
| pytest 通过率 | 未跑通 | **100%** | GPU 日志 |
| 人工介入次数 | **4 次** | **0 次** | `ab_runs.csv` |
| 端到端耗时 | **~1200s** | **~7s/算子** | `forge_runs.jsonl` |
| 结构错误拦截 | Triton/缺 premake | preflight **100% 拒绝** | 脚本实测 |

```bash
python scripts/eval_ab.py --input docs/ab_runs.csv --output docs/AB_Report.md
```

---

## 6. 通过标准（初赛）

- [x] `doctor` 全 OK（GPU）
- [x] `forge gate`：silu / add / gelu / relu / mul 全 `GATE OK`
- [x] 每个算子 GUARD：`matches reference`
- [x] 每个算子 PROVE：pytest 全通过
- [x] 审计日志 `docs/forge_runs.jsonl` 可复现
- [x] A/B 干净 5 baseline / 5 treatment（见 `docs/AB_Report.md`）
- [x] skill 包含 `examples/`、`references/`、`tests/`（见 `skills/ntops-forge/`）

---

## 7. 风险与规避

| 风险 | 规避 |
|------|------|
| 跑全量 `pytest tests/` 误失败 conv2d | 只用 `tests/test_<op>.py` |
| 在 `$HOME` 跑脚本找不到文件 | 固定 `cd /root/work/skill` |
| gelu 参考无 `application()` | `compare_ref` 支持 `default_application` |
| 复杂算子硬写公式 | taxonomy 路由 → 只读 reference |

---

## 8. 四类自测案例（赛题 4.2）

| 案例 | 文件 |
|------|------|
| ST1 逐元素/广播 | `docs/selftests/ST1_elementwise.md` |
| ST2 归约/分块 | `docs/selftests/ST2_softmax_reduce.md` |
| ST3 布局 stride | `docs/selftests/ST3_max_pool2d_layout.md` |
| ST4 性能/诊断 | `docs/selftests/ST4_perf_diagnosis.md` |

评分对照：`docs/ScoringAlignment.md`

## 9. 附件

- `docs/GPU_Test_Report.md`、`docs/FinalsRoadmap.md`、`docs/BenchmarkPlan.md`
- `docs/screenshots/`（17 张）、`docs/forge_runs.jsonl`、`docs/bench_silu.json`
