# ST3：布局敏感算子（max_pool2d · stride）

**赛题 4.2 类型**：非连续输入、步长或偏移量场景  
**状态**：✅ spec + **GPU pytest 62 passed**（54 skipped 为上游 Invalid padding）

## 任务说明

max_pool2d 涉及 kernel_size、stride、padding、dilation、ceil_mode。

## GPU Correctness 验证 ✅

```bash
cd /root/work/ntops
pytest tests/test_max_pool2d.py -v
```

| 结果 | 详情 |
|------|------|
| **62 passed** | 多 stride/kernel/padding/dilation 组合 |
| **54 skipped** | 上游标记 `Invalid padding`（非 skill 失败） |
| 耗时 | 175.45s |

截图：`docs/screenshots/19-st3-maxpool2d-pytest-summary.png`

## 布局场景（已验证）

| 场景 | 状态 |
|------|------|
| stride=None / 1 / (2,3) | ✅ 部分组合 passed |
| padding=0 + stride=2 | ✅ passed |
| 非法 padding 组合 | skipped（官方预期） |

## Agent PLAN

```bash
python scripts/forge.py plan max_pool2d
# pooling ∈ COMPLEX_FAMILIES → 先读 reference
```

## 证据

- spec：`skills/ntops-forge/specs/max_pool2d.yaml`
- 实测：`docs/st2_st3_gpu_results.md`
