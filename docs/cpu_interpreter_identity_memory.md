# 解释器恒等布局内存优化（2026-09-17）

## 结果

在旧分支 `3c0ceccf540233fbe2f23656f71b69268dd811e1` 基础上，把未安排布局的张量和无访问映射的恒等布局坐标改为每轴向量的广播视图。坐标临时存储由与「维数 × 元素总数」成正比，降为与「各轴长度之和」成正比。读取结果、活动掩码、原始存储与完整轨迹仍按原合同处理，不缓存输入数据。

三轮新的 Python 进程成对对照均通过预先设置的 20% 峰值下降与每场景不超过 5% 时间退化门槛。以下为第一轮 warm 的执行分配峰值（字节），下降比例取三轮最小值。

| 场景 | 原版峰值 | 候选峰值 | 最小下降 |
|---|---:|---:|---:|
| `identity_257x509_reversed` | 5,764,951 | 3,678,375 | 36.19% |
| `identity_512x512_contiguous` | 11,543,515 | 7,357,707 | 36.26% |
| `identity_8x16x32x32_contiguous` | 9,971,979 | 5,779,099 | 42.05% |

- 指标为 `tracemalloc` 在公开 `interpret_program` 调用期间捕获的峰值，预先分配的输入和输出不计入；不是 RSS，也不代表整个训练任务节省相同比例的内存。
- 原始数据包含 warm/cold 缓存、trace 开关、排列布局 affine/softmax 对照，共 30 条成对记录；全部完整轨迹 SHA-256 相同。开启轨迹的目标场景峰值下降约 10%–17%；没有压缩或删减轨迹。
- 本轮以内存为主要收益。第一轮时间测量部分与本地 CPU 回归重叠，因此不将最大时间加速值作为性能承诺；所有场景的三轮门槛仍通过。

## 同时修复的边界问题

新测试在原版复现了两例零维张量 `write(mask=False)` 错误：空坐标元组仍选中标量，导致将空数组赋给标量而报错。候选在零维存储和零维掩码时使用布尔索引，既保留掩码，也避免正向标量写入的数组到标量弃用警告。原版失败日志完整保留。

## 验证

- 本地纯 CPU 环境，无 Torch/Triton：`577 passed, 15 deselected in 38.21s`。15 条真实 GPU 差分测试随后在 RTX4090D 上独立通过。
- 新增 42 条测试覆盖标量、空轴、连续、反向、切片、转置、全开/全关/交替掩码、两种恒等布局、访问观察器和公开 SSA 别名写入；`-W error` 通过。
- 同一新增测试在远端 NumPy 1.26.4 环境也以 `-W error` 通过：`42 passed in 0.49s`。
- CUDA 接口复测期间 NineToothed 的 187 个文件逐一确认没有改变，因此复用上述 GPU 证据。
- 改动的 Python 源码与测试 Ruff 检查通过。
- 环境：macOS arm64，Python 3.12.14，NumPy 2.3.5。原始结果、逐文件哈希及复现工具见[固定验证档案](https://github.com/a962695448-rgb/ninetoothed/tree/345cacbb6e9ea2a74f45d399d7bc0b746f46e9be/docs/validation/identity-memory-20260917)。

## 文档与最终格式检查

GPU 测量后的收尾仅在 `memory.py` 和新增测试中补齐项目要求的空行，Python AST 逐项核对完全一致；没有修改计算逻辑，记录见[格式前后源码对应](https://github.com/a962695448-rgb/ninetoothed/blob/345cacbb6e9ea2a74f45d399d7bc0b746f46e9be/docs/validation/identity-memory-20260917/checks/format-only.json)。格式后的最终文件再次通过完整 CPU 回归：`577 passed, 15 deselected in 36.93s`。

修改的 CPU 指南及安装说明以严格 Sphinx（`-W --keep-going`）单独构建通过。整个站点的普通 Sphinx 构建在本机纯 CPU 环境因旧有 `ninetoothed.debugging` 自动文档导入需要 Torch 而失败；失败日志保留，不将专项文档检查写成全站构建通过。没有为文档检查向纯 CPU 测试环境安装 Torch。
