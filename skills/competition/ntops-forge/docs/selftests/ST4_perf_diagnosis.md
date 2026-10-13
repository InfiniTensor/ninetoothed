# ST4：性能验证与失败诊断

**赛题 4.2 类型**：benchmark、性能回退分析、失败诊断  
**状态**：✅ 初赛完成（2 项 benchmark + 3 项诊断案例）

## 任务 A：开发流水线耗时 benchmark ✅

| 指标 | Baseline | Treatment |
|------|----------|-----------|
| 平均步骤 | 6 | 1 |
| 人工介入 | 4 | 0 |
| 平均耗时 | ~1200s | ~7s/算子 |

```bash
bash scripts/run_ab_manual.sh /root/work/ntops
```

证据：`docs/AB_Report.md`、图16

## 任务 B：silu 算子级 GPU benchmark ✅

| 字段 | 值 |
|------|-----|
| shape | 4096×4096 fp16 |
| PyTorch | 0.0523 ms |
| ntops | 0.0715 ms |
| ratio | 1.37× |

```bash
python scripts/bench_op.py --op silu --ntops-root /root/work/ntops
```

证据：`docs/bench_silu.json`、图17

**结论**：同量级无数量级回退；skill 主优化在开发效率，非内核极限调优。

## 失败诊断案例

| 案例 | 现象 | 修复 | 证据 |
|------|------|------|------|
| FC-001 | Triton @triton.jit | 改用 premake/application | 图8 |
| FC-004 | 脚本路径错误 | cd /root/work/skill | fix_cards |
| FC-012 | No module named pytest | pip install pytest | GPU 日志 |
| gelu 特例 | 无 application() | compare_ref default_application | 图7 |

```bash
python scripts/forge.py diagnose --log docs/forge_runs.jsonl
```

## 性能回退诊断流程

1. jsonl 定位失败阶段（PLAN/CODEGEN/GUARD/PROVE/SHIP）
2. compare_ref 排除语义偏差
3. bench_op.py 对比 PyTorch 基线
4. fix_cards 给出最小修复动作

## 证据

- `docs/BenchmarkPlan.md`
- `skills/ntops-forge/fix_cards.md`
- 截图：图8（Triton FAIL）、图9（diagnose）、图17（benchmark）
