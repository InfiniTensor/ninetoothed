# ntops-copilot 有效性验证

```bash
python scripts/run_task.py --task silu --ntops-root /root/work/ntops --finish
python scripts/run_task.py --task add --ntops-root /root/work/ntops --finish
```

通过标准：preflight OK + compare_ref matches + pytest 8 passed。

完整五算子回归请用 **ntops-forge**：`python scripts/forge.py gate`。
