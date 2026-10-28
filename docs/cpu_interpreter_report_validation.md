# GPU 报告保护：补充验收

2026-09-16 晚间，在 `788208c` 基础上修复 `verify_interpreter_gpu.py` 覆盖既有成功报告的问题，并添加九项 CPU 报告协议回归；解释器、SSA、编译器和数值用例未改动。

| 验证 | 结果 | 范围 |
|---|---|---|
| 无 Torch/Triton 的本地 CPU 回归 | 502 passed、15 GPU deselected，43.91 秒 | 包含新增九项报告测试 |
| 最终本地报告测试 | 9 passed，0.37 秒 | 格式整理后的文件；包含在上述范围 |
| RTX 4090 D 实际 Triton GPU 差分 | 15/15 PASS | 最终脚本、9 个程序、10 类功能 |
| RTX 4090 D 环境的最终报告协议测试 | 9 passed，0.35 秒 | 测试替身检查报告行为，不是另九个 GPU 数值用例 |
| 静态检查 | Ruff、format、项目贡献风格通过 | 本轮源码 |

报告输出路径必须不存在。普通文件、目录及普通/悬空符号链接在 GPU 初始化前被拒绝；正常中断记录已完成用例，标为 INTERRUPTED 并退出 130。环境不可用退出 2，数值失败退出 1，完整通过才退出 0。进度刷新到文件流，但不承诺断电、SIGKILL 或写盘错误后的文件完整性。

首次运行后只整理脚本空行、测试错误消息标点；报告脚本 AST 不变，最终两个文件再次实机复验。源码指纹、两轮原始输出及本地 CPU JUnit 均在[固定证据](https://github.com/a962695448-rgb/ninetoothed/tree/c6d9dd6bfaa3d7fe1b6dd06aedde427549c9f556/docs/validation/report-lifecycle-20260916)中。最终记录为 `nine-final-*`，首次记录为 `nine-gpu.*` / `nine-report-tests.*`，不重复累计 GPU 用例数。

环境：RTX 4090 D 24GB、sm89、驱动 570.124.06、NGC Torch 2.6.0a0+ecf3bae40a.nv25.01、CUDA 12.8、Triton 3.1.0、Python 3.12.3、NumPy 1.26.4、SymPy 1.13.1。

复现：

```bash
python scripts/verify_interpreter_gpu.py --report /tmp/new-gpu-report.json
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests/test_interpreter_gpu_report.py
```

本轮没有重复运行未修改的完整卷积集合。之前的 [902 项集合验收](cpu_interpreter_rtx4090_validation.md)仍对应 `1b68040`；历史 A100 记录保留原版本范围，不能将旧统计冒充本次新增测试后的全仓库通过数。此次是正确性和验收记录保护改进，不宣称计算性能提升。
