# ntops-forge 有效性验证

本目录说明如何验证 **skill 本身**（非 ntops 上游全量测试）。

## 快速验收

```bash
cd /root/work/skill
python scripts/doctor.py
python scripts/forge.py gate --ntops-root /root/work/ntops
```

通过标准：`GATE OK: all operators passed`（silu / add / gelu / relu / mul）。

## A/B 证据

```bash
bash scripts/run_ab_manual.sh /root/work/ntops
cat docs/AB_Report.md   # 预期 5 baseline / 5 treatment
```

## 护栏单测

| 检查 | 命令 | 预期 |
|------|------|------|
| Triton 拒绝 | `python scripts/preflight.py <triton_sample.py> --strict` | FAIL |
| forge 通过 | `python scripts/preflight.py /tmp/silu_forge_kernel.py --strict` | OK |
| 语义对照 | `python scripts/compare_ref.py --gen ... --ref ...` | matches reference |

## 决赛前扩展验证

见 `docs/FinalsRoadmap.md`：softmax（归约）、max_pool2d（stride 布局）、benchmark 案例。
