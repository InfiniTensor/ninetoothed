# 布局调用与空形状边界修复（2026-09-17）

## 两个已复现的问题

基线：`9de14a3d8effad34799467faa90bef936434c315`。

1. `IndexExpr.parse("int('10', base=16)")` 静默丢掉关键字，渲染为 `int('10')` 并返回 10；`int(**options)` 被改写为 `int()` 并返回 0。结构化 IR 只有位置参数表示，因此现在在解析阶段明确拒绝关键字和 `**` 传参。合法位置调用 `int('10', 16)` 仍返回 16，原有位置语法可往返。
2. `next_power_of_2(0)` 原先返回 2，负整数也被变成正的幂次。相同公式还用于命名的 padded-shape 符号，导致零长度恒等视图被认为长 2，公开 `interpret_program` 的空读取报“没有访问映射”。现在调用与符号绑定共享同一私有函数，非正输入返回 0，正整数继续向上取整。

## 依据与范围

参考 [Triton 3.1.0 的固定源码](https://github.com/triton-lang/triton/blob/cf34004b8a67d290a962da166f5aa2fc66751326/python/triton/__init__.py)。标签 v3.1.0 解析到提交 `cf34004b8a67d290a962da166f5aa2fc66751326`，其六步位扩散实现对零和负的 signed-64-bit 输入返回 0。测试用独立的六步移位循环核对该范围，未导入或执行下载的模块。

本改动修正的是解释器逻辑形状，不调整 GPU 启动块大小或调度参数的默认值。原有 `int(value)` 规范化保持，例如可转换的 NumPy 整数、布尔值及 2.9→2。非正值行为和关键字拒绝是有意修复；不以兼容错误结果为目的。

## 验证

- 同一新测试在原版复现 **21 failed, 46 passed**；修复后 **67 passed**，开启 `-W error`。
- 新增用例包括关键字/展开关键字拒绝、合法位置调用、正负整数及边界、类型规范化、两种 padded-shape 表示、公开形状查询和空读取。
- 完整无 Torch/Triton CPU 范围：**729 passed, 15 deselected in 37.43s**。
- Ruff、format、项目贡献风格检查通过。性能没有作为此轮的采用依据，也没有宣称提速。
- 本轮没有新开 GPU 或声称新的硬件验证；参考源码核对、CPU 行为与后续 GitHub CI 分别记录。先前实机记录继续属于原先的源码范围。

从仓库根目录复现：

```bash
python scripts/run_cpu_tests.py --junitxml /tmp/layout-calls-cpu.xml
```

[原版失败、修复结果、源码与固定参考](https://github.com/a962695448-rgb/ninetoothed/tree/d75de7b29bec4b81bba4a058d3f0ce66ad4f4142/docs/validation/layout-calls-20260917) 已固定归档。
