# GPU 环境测试报告

**日期**：2026-06-08（更新）  
**选手**：于鸿伟（hongwei-2026）  
**Skill 版本**：ntops-forge v1.0 / ntops-copilot v0.5  
**环境**：AutoDL · `autodl-container` · RTX 4080

## 硬件与 CUDA

```
GPU: NVIDIA GeForce RTX 4080
torch.cuda.is_available(): True
conda: /root/miniconda3/envs/base
```

截图：`docs/screenshots/01-doctor-gpu-ok.png`

## 工作目录

```bash
source /root/miniconda3/bin/activate base
pip install pytest
pip install -e /root/work/ntops
cd /root/work/skill
python scripts/doctor.py
```

## v1.0 forge gate（最终验收 · 五算子）

```bash
python scripts/forge.py gate --ntops-root /root/work/ntops
```

| 算子 | GUARD compare_ref | pytest | 单算子耗时 | 结果 |
|------|-------------------|--------|------------|------|
| silu | matches reference | 8 passed | 6.7s | OK |
| add | matches reference | 8 passed | 6.9s | OK |
| gelu | matches reference | 8 passed, 8 skipped | 7.0s | OK |
| relu | matches reference | 16 passed | 10.9s | OK |
| mul | matches reference | 8 passed | 12.5s | OK |

**GATE OK: all operators passed**（五算子合计约 44s）

截图：
- `docs/screenshots/02-forge-gate-summary.png`
- `docs/screenshots/13-forge-gate-gelu-pipeline.png`
- `docs/screenshots/07-forge-audit-jsonl.png`（jsonl 审计）

## 护栏与对照

| 检查 | 结果 | 截图 |
|------|------|------|
| Triton 稿 preflight | 5 项 FAIL | `05-preflight-triton-vs-forge.png` |
| forge 稿 preflight | OK | 同上 |
| compare_ref silu/gelu | matches reference | `15-compare-ref-silu-gelu.png` |
| spec 公式注入 | formula → application 一致 | `04-spec-formula-injection.png` |

## A/B 对照（最终 · 5v5）

| 指标 | Baseline | Treatment |
|------|----------|-----------|
| preflight | 0% | 100% |
| pytest | 未跑通 | 100% |
| 平均步骤 | 6 | 1 |
| 人工介入 | 4 | 0 |
| 平均耗时 | 1200s | 7s/算子 |

截图：`docs/screenshots/16-gate-ab-5v5-final.png`（gate + A/B 同屏）  
数据：`docs/ab_runs.csv`、`docs/AB_Report.md`

## silu benchmark（RTX 4080 · 2026-06-08）

| 指标 | PyTorch | ntops | 比值 |
|------|---------|-------|------|
| 4096×4096 fp16 | 0.0523 ms | 0.0715 ms | 1.37× |

截图：`docs/screenshots/17-bench-silu-gpu.png` · 数据：`docs/bench_silu.json`

## ST2/ST3 归约与布局 pytest（2026-06-08）

| 测试 | 结果 | 截图 |
|------|------|------|
| test_softmax.py | **8 passed** | `18-st2-st3-pytest-start.png` |
| test_max_pool2d.py | **62 passed**, 54 skipped | `19-st3-maxpool2d-pytest-summary.png` |

> 54 skipped 为 ntops 官方对非法 padding 的 skip，非失败。详见 `docs/st2_st3_gpu_results.md`。

## 注意事项

- **不要**运行 `pytest tests/` 全量：上游 `conv2d` 可能失败，与 skill 无关。
- 新机器需 `pip install pytest`，否则 PROVE 阶段报 `No module named pytest`（见 fix_cards FC-012）。
- 工作目录必须是 `/root/work/skill`。

## 结论

ntops-forge + ntops-copilot 在 RTX 4080 GPU 环境完成五算子工厂流水线验收，A/B 干净 5v5 数据已记录，可作为初赛「可实现 + 可量化」证据。
