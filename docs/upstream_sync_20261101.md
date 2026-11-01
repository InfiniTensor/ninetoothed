# 2026-11-01 上游兼容性修复

## 范围

将上游 `f64a6de566aa1c7321aed92edb9285690cbf3c66` 合入 CPU 解释器分支，解决五处合并冲突。本轮是截止后的兼容性维护，不新增性能优化；截止前提交与历次硬件验收仍保留各自的源码版本范围。

## 修复

- CPU 准备过程复用编译器的冻结前端和合法 meta 默认值，保留显式参数优先级，避免重复调用 arrangement。
- 支持新版 `dtype.constant/of/select`、cast/构造器 dtype 操作数；保留旧 dtype 引用。类型查询只读元数据，跟踪快照可序列化，失败包可重放。
- 把 Triton 导入移到实际设备访问函数，恢复无 Torch/Triton 的 CPU 导入路径。
- 类型规范化保留未改操作的身份，并明确记录实际改写/删除的来源；更新 pass 分析信息后仍保留差分定位。整数 matmul 分解使用类型合法的零初值。
- SSA emitter 同时保留上游动态提取索引和原广播地址/mask 映射。旧失败回放用例使用合法但尚无 CPU 实现的数学操作，继续检验原异常、定位及回放，不放宽 SSA 校验。

## 本轮验证

| 范围 | 实际结果 |
|---|---|
| 无 Torch/Triton 的完整 CPU 选择 | 970 passed, 17 deselected in 36.49s |
| 前端/dtype/named layout/嵌套形状边界 | 46 passed, 17 deselected in 5.51s |
| Torch CPU 适配 | 9 passed, 1 deselected, 1 warning in 1.06s |
| 前端专项格式调整后复验 | 10 passed in 0.72s |
| 静态与文档检查 | Ruff、格式、贡献风格及 Sphinx `-W` 通过 |
| 包装与独立运行 | 92 个 wheel Python 源码逐字节一致；全新 CPU 环境的 `python -I` 导入、debug 示例、独立 replay 通过 |

CPU 主选择的 17 项排除是 15 个真实 GPU 差分以及动态整数归约、动态表提取两个 GPU 用例。边界选择排除真正的 GPU 执行函数；Torch 选择排除实际 CUDA 张量拒绝检查，保留原 sparse 提示。各选择范围有重叠，不相加。主测试后仅给一个新测试补空行，该测试文件的 10 项已重跑，执行源码未变。

修复前合并候选发生无 Triton 导入错误；修复导入后记录为 117 failed / 823 passed / 17 deselected。新前端用例在独立进程加载旧准备逻辑的对照为 5 failed / 5 passed，当前为 10 passed。上述失败是诊断记录，不计入通过数量。

复现主 CPU 选择（全新无 Torch/Triton 的环境）：

```sh
python -m pip install -r requirements-cpu.txt
python scripts/run_cpu_tests.py --junitxml cpu-results.xml
```

包安装与隔离运行步骤见 `.github/workflows/cpu-interpreter.yml`，使用普通 wheel、`--no-deps` 和源码目录外的 `python -I`，无需 GPU 软件包。

文档构建使用已有 Torch CPU 环境及 Sphinx 依赖；修复上游 `build.rst` 一处标题下划线长度警告后严格构建通过。没有禁用警告或模拟可选依赖。

## 验收边界

本轮没有运行实际 GPU、全库 GPU pytest、性能测量或多卡测试，没有新租赁算力。历史 A100/4090 结果不作为本轮合并代码的硬件验证。上游 PR 的工作流批准和代码审核仍由维护者执行；排队、待批准、跳过或取消均不视为通过。
