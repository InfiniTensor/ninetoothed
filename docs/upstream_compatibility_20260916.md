# 与上游多平台架构的兼容性检查

本轮合入上游 22e74c3afe47e12ee29d7d0bcfaf1de8286f4560（多平台目标架构）。
解释器、差分定位和回放功能继续通过共享 frontend/SSA 工作。

## 接入与冲突处理

- 编译器保留共享的 prepare_application，同时接入上游 TargetContext、平台选项和约束。
- SSA pass 保留来源与结果对应记录，并支持新增的 ssa.validate_target_capabilities 阶段。
- 使用上游等价的嵌套布尔表达式辅助函数；流上下文和 scatter 测试采用上游现已修正的版本。
- jagged 测试保留已验证的显式维度构造及独立 shape/offset 断言，避免把原本可执行的维度改成跳过。

## 本轮 CPU 验证

在无 Torch/Triton 的独立 Python 3.12 环境中：

~~~text
493 passed, 15 deselected in 42.79s
~~~

统一入口为 python scripts/run_cpu_tests.py。本轮 493 项包含上游平台配置测试，
三种后端的默认管线正向检查，以及平台能力拒绝时定位到对应 pass、停止后续阶段、
不虚报数值操作定位的检查。37 项管线/平台专项属于这 493 项的子集，不重复累计。

原项目的 A100 835 passed、2 skipped 结果仍属于旧计算源码 ed33273。
本轮没有执行 GPU 回归，不能把旧 A100 结果当作合入新上游后的完整硬件验收。
维护者可以结合实际 GPU CI 复核编译和运行影响。

文档、风格、独立安装与回放的最终检查记录见 cpu_interpreter_validation.txt；
在线 CPU 工作流另覆盖 Python 3.10 和 3.12。
