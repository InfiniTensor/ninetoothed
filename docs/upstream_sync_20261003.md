# 2026-10-03 上游兼容性维护

## 版本与目的

截止前提交快照仍为 [8cc6b36](https://github.com/a962695448-rgb/ninetoothed/commit/8cc6b36a9cd399d4c3c5cf845af8f47d79a6fa7c)，其计算版本与 A100 验收保持 [150f526](final_gpu_freeze_20260920.md) 的原始范围。本文记录截止后因官方 master 更新产生的兼容性维护，供评审区分版本。

本次受测计算与测试提交为 **960645200f8647dc2f62e992a41e65841642526e**，Git 树 **e50f3020e27d293c3b475d816513bc7811be9b51**。它以原提交 8cc6b36 和官方 **1f9f476fe4f7fc510abaaeff08ce6f00a3474e3c** 为双亲；之后只追加说明文档。上游自 7ffdab8 后的十二项提交全部保留，涉及 libdevice、标量参数、归约坐标与调度、区域副作用、源偏移、SSA 契约和 dtype 来源。

## 合并与兼容修复

- 保留上游的新共享表达式生成器，在该入口调用原有负整数 floor/mod 修正。原广播地址、掩码与逻辑坐标保护保留。
- 上游 #237 的新 SSA 校验与解释器已有的指针写入、完整视图读取、布尔 mask 和 load fallback 形式存在合同差异。本次明确接受这些已支持形式，仍拒绝非法操作数/结果数量、目标类型、非布尔 mask、未知内存操作和非法区域。原访存、OOB、共享存储、stride 与字节依赖断言保留；新增 42 项合同回归。原先将 tensor load 当作非法的上游控制用例改为确实无效的 scalar load。
- 上游 #227 使用第二 SSA 操作数或 dtype_ref 表示动态类型。旧解释器忽略该引用会把 cast 误作 float64、把构造误作 float32。现在使用实际数组/标量的 dtype 元数据；TensorRef 和 Pointer 的类型查询不读取其数据。新增 32 项前后相同的回归在修复前为 **23 failed / 9 passed**，修复后全部通过，联合原字面量和缓存专项 **85 passed**。
- 手写循环 fixture 补齐 index induction 和 iter_args，保持输入、运算和断言。生成代码检查准确选取实际 kernel，避免误选新辅助函数。
- CPU 入口另行排除上游新增的实际 CUDA 动态整数归约用例，保持 Torch/Triton 不存在的 CPU 环境要求。

初次三方合并回归的 **59 failed / 800 passed / 15 deselected** 已保留在本地日志；没有通过删除原行为断言、伪造指针类型或取消严格区域校验获取通过。

## 验证

本地 CPU 环境为 macOS arm64 / Python 3.12.14 / NumPy 2.3.5 / SymPy 1.14.0 / pytest 9.1.1，未安装 Torch/Triton：

~~~text
932 passed, 16 deselected in 27.65s
~~~

统一入口覆盖 35 个解释器与 SSA 模块。16 项排除分别为原 15 个实际 GPU 差分，以及上游新增的动态整数归约 GPU 用例。

隔离 Torch CPU 环境（Torch 2.14.0 / NumPy 2.5.3 / SymPy 1.14.0）另行执行类型别名、Torch 适配、默认 pass、缓存、上游 block reduction、libdevice 和标量参数检查：

~~~text
97 passed, 6 skipped, 1 deselected, 1 warning in 2.35s
~~~

6 项跳过为无 GPU 的设备参数或 CUDA 条件；1 项取消选择为实际 CUDA 张量拒绝检查。首次命令未正确选择这个环境限制项，按原测试明确报告 UNVERIFIED；修正选择表达式后得到上面结果。sparse 输入测试保留 Torch invariant-check 警告。两个选择范围重叠，不将其计数相加。

Ruff、180 文件格式检查和项目贡献风格通过。214 份冻结源码输入在回归前后 SHA256 一致；wheel 中和独立安装后的 70 份 Python 源码逐字节匹配。源码目录外以 python -I 执行 debug 示例和导出的独立 replay，正确定位注入的坏常量。严格 Sphinx -W 文档构建通过。

复现主入口：

~~~bash
python -m pip install -r requirements-cpu.txt
python scripts/run_cpu_tests.py --junitxml /tmp/nine-cpu-results.xml
~~~

## 硬件证据范围

本轮执行 CPU 回归、代码生成检查、安装回放和文档验证，没有租赁服务器或运行新的 GPU 实测。9 月 20 日的 A100 针对性验收和 9 月 19 日的历史全库 1206/2 结果分别保留各自源码范围，不能套用到本轮合并代码。GitHub Actions 的未执行、跳过或取消状态也不计作 GPU 通过。
