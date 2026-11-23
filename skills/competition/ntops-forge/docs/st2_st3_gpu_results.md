# ST2 / ST3 GPU pytest 实测（2026-06-08）

**环境**：AutoDL · RTX 4080 · `/root/work/ntops`

## 命令

```bash
source /root/miniconda3/bin/activate base
cd /root/work/ntops
pytest tests/test_softmax.py tests/test_max_pool2d.py -v
```

## 结果

| 测试文件 | passed | skipped | 耗时 | 说明 |
|----------|--------|---------|------|------|
| `test_softmax.py` | **8** | 0 | ~6s | 多 shape × dtype × cuda |
| `test_max_pool2d.py` | **62** | 54 | ~175s | 54 skipped 为上游 `Invalid padding`（非失败） |
| **合计** | **70** | 54 | ~2m55s | 116 collected |

## 截图

- `docs/screenshots/18-st2-st3-pytest-start.png` — softmax 8 passed + max_pool2d 进行中
- `docs/screenshots/19-st3-maxpool2d-pytest-summary.png` — `62 passed, 54 skipped in 175.45s`

## 结论

- **ST2 归约类**：官方 softmax 内核在 GPU 上 correctness 全通过  
- **ST3 布局类**：max_pool2d 覆盖 stride/kernel/padding/dilation 组合，62 组通过；skip 为 ntops 官方对非法 padding 的跳过逻辑
