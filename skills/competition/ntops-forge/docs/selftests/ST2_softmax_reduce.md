# ST2：归约 / 分块类算子（softmax）

**赛题 4.2 类型**：归约或分块类（softmax 等）  
**状态**：✅ PLAN + reference + **GPU pytest 8/8 通过**

## 任务说明

softmax 沿 dim 归约，需两遍循环（max 稳定 + 归一化），**禁止公式硬注入**，必须先读官方 reference。

## Agent 执行记录摘要

```bash
python scripts/forge.py spec softmax
python scripts/forge.py plan softmax
```

PLAN 对 `reduction` family 输出 STOP，强制读 reference（见 `taxonomy.md`）。

## GPU Correctness 验证 ✅

```bash
cd /root/work/ntops
pytest tests/test_softmax.py -v
```

| 结果 | 详情 |
|------|------|
| **8 passed** | shape0–shape7 × dtype × cuda |
| 0 failed | RTX 4080 · 2026-06-08 |

截图：`docs/screenshots/18-st2-st3-pytest-start.png`

## Reference 阅读要点

官方 `src/ntops/kernels/softmax.py`：reduction.arrangement、两遍循环、float16 稳定 _exp。

## 证据

- spec：`skills/ntops-forge/specs/softmax.yaml`
- 实测：`docs/st2_st3_gpu_results.md`
