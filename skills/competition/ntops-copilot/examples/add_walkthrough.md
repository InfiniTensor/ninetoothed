# 完整示例：ntops-copilot 完成 add 算子

## 任务

二元广播算子 `add`，验收 `tests/test_add.py`。

## 一键执行

```bash
source /root/miniconda3/bin/activate base
cd /root/work/skill
python scripts/run_task.py --task add --ntops-root /root/work/ntops --finish
```

## 六步工作流

1. 读 `tasks/task_add.yaml`
2. scaffold 生成内核
3. preflight 结构检查
4. compare_ref 语义对照
5. pytest `tests/test_add.py`
6. 记录 A/B + PR 提示

## 与 forge 的关系

复杂算子用 **forge**；单算子快速路径用 **copilot --finish**。见 `docs/DualSkillGuide.md`。
