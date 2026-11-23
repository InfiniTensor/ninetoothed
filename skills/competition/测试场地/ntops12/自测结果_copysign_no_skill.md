# 自测任务记录：copysign（无 Skill 对照版）

> 对照版（Skill 辅助）：`测试场地/ntops11/自测结果_copysign_with_skill.md`
> 工程目录：`测试场地/ntops12/`
> 对话记录：`测试场地/ntops12/copysign_no_skills.txt`

## 输入任务说明

与 Skill 版完全相同：实现 `copysign(input, other)`，支持 fp32/fp16，正确处理 IEEE 754 ±0.0，
`input` 与 `other` shape 相同。

## AI 智能体执行记录摘要

- **Step 0 — 通用 Explore 子代理勘探**
  - 未激活 Skill，派生通用 Explore 子代理扫 `ntops12/` 工程结构，识别出"三文件模式"
    （kernel / torch / test）。
  - 未发现 Skill 专属的 `references/TAXONOMY.md`、`API_REFERENCE.md`、`OPTIMIZATION_GUIDE.md`、
    `FIX_CARDS.md` 等文档；仅依靠 `mul.py`、`pow.py`、`sign.py`、`signbit.py` 等已有算子反推规范。
  - 未做远端 GPU 连通性检查（无 MCP 调用），全程本地。

- **Step 1 — 隐式分类**
  - 无显式分类步骤。通过 grep 确认 `copysign` 尚未存在后，隐式归为 "binary element-wise"，
    复用 `ntops.kernels.element_wise.arrangement`（模仿 `mul.py`）。

- **Step 2 — 代码生成（三版迭代）**
  - **V1**：bitcast 双 dtype 路径（fp16 用 `uint16` 掩码，fp32 用 `uint32` 掩码），
    使用 `ntl.uint16(0x8000)` / `ntl.uint32(0x80000000)` 构造常量。
  - **V2（自我否定 V1）**：grep 发现 `ntl.uint32(...)` 构造器在整个 codebase 中**零出现**，
    担心是非法语法，推翻 V1。改为 `signbit(other) > 0` + `ntl.where(..., -abs, abs)`。
  - **V3（自我否定 V2）**：写到一半意识到 `< 0.0` 无法识别 `-0.0`（IEEE 754: `-0.0 < 0.0 == False`），
    再改回 bitcast + `>> 15 / >> 31` 取符号位，然后 `ntl.where(other_sign > 0, -abs, abs)`。
  - Wrapper 未做 shape/dtype assert；未指定 block_size（使用 `ninetoothed.block_size()` 默认值）。

- **Step 2.5 — 测试循环（仅本地 parse 验证，未跑 GPU）**
  - 本地 `python -c "import ast; ast.parse(...)"` 三个新文件语法 OK。
  - 本地 `import ntops` 失败（无 `ninetoothed` 依赖），**未执行 pytest**。
  - **未上传到 MetaX C500，正确性未实测**。

- **Step 2.8 — 性能优化**
  - **无**。未生成 benchmark 脚本，未做 tile sweep，未调整 block_size。

## 产出补丁摘要

- **新增文件**：
  - `src/ntops/kernels/copysign.py`（V3 最终版）
  - `src/ntops/torch/copysign.py`
  - `tests/test_copysign.py`
- **修改文件**：
  - `src/ntops/kernels/__init__.py`、`src/ntops/torch/__init__.py`（注册 copysign，按字母序插在 cosh 之后）
  - `.gitignore`
- **关键代码片段（kernel application，V3 最终版）**：

```python
def application(input, other, output):
    if input.dtype is ntl.float16:
        other_bits = ntl.cast(other, ntl.uint16, bitcast=True)
        other_sign = other_bits >> 15
    elif input.dtype is ntl.float32:
        other_bits = ntl.cast(other, ntl.uint32, bitcast=True)
        other_sign = other_bits >> 31

    abs_input = ntl.abs(input)

    output = ntl.where(other_sign > 0, -abs_input, abs_input)  # noqa: F841
```

## Correctness 测试

命令：`python -m pytest tests/test_copysign.py -v`
结果：**未执行**（本地无 GPU，agent 未走 MCP 上传到远端）。

测试覆盖设计（纸上分析）：
- 8 个 `shape × dtype` 参数化（1D–4D × fp32/fp16）
- 2 个 `special_values`（±0.0, ±1.5, ±2.5, nan, inf, -inf）
- 共 10 个用例（Skill 版为 12 个，缺 `test_copysign_signed_zero` 专项）
- 使用 `torch.equal(int_view(ninetoothed), int_view(reference))` 做位精确比较（更严格但正确）

非连续输入：**未涉及**。

## Benchmark

**未执行**。Agent 未生成任何 benchmark 脚本，也未调用 `auto_bench` 或 `bench_compare.py`。

## 失败诊断（纸上分析，未实测）

**已识别的潜在问题**（agent 自纠）：
- V1 失败：`ntl.uint16(0x8000)` / `ntl.uint32(0x80000000)` 构造器在 codebase 中无先例，担心编译失败 → 推翻。
- V2 失败：`other < 0.0` 不能识别 `-0.0`，会违反 IEEE 754 signed-zero 语义 → 推翻。

**未在实测中暴露的潜在风险**（仅事后分析）：
1. **`ntl.where` 对 fp16 的行为**：`-abs_input` 在 fp16 上是否保留 NaN 符号位，未验证。
2. **`other_sign > 0` 的类型兼容性**：`other_sign` 是 `uint16`/`uint32` tile，与 Python int 比较，
   在 ninetoothed 编译后是否成立，未在 GPU 上验证。
3. **Wrapper 未做 shape/dtype assert**：用户传错 shape 时会触发难以定位的 kernel 错误。
4. **性能未优化**：默认 block_size 在大 shape 上可能仅 0.5–0.9x torch（参照 Skill 版 V1 数据）。

## 不支持用例

与 Skill 版相同，外加：
- **性能保证**：无 benchmark 数据，无法给出小 shape / 大 shape 行为结论。
- **实测正确性**：未在 MetaX C500 上验证，所有"正确"结论均为纸面分析。
