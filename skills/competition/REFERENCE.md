# References and Disclosure

## Public References

### NineToothed (九齿) DSL

- **Source repository** — <https://github.com/InfiniTensor/ninetoothed>
  使用范围：`ninetoothed.make`、`ninetoothed.jit`、`@ninetoothed.kernel`、tile 语义、`arrangement` / `application` 分离、`torch_impl` 对拍契约。
- **Examples repository** — <https://github.com/InfiniTensor/ninetoothed-examples>
  使用范围：模式级思路参考（逐元素 tile、归约 + fp32 累加、三级 matmul）。本 Skill `examples/` 目录下的算子均为独立重写实现，未搬运任何官方示例源码。
- **Documentation** — <https://ninetoothed.org/>
  使用范围：tile 形状规则、`ninetoothed.Tile` API、`power_of_two` 常量、符号索引约束。

### PyTorch

- **Documentation** — <https://pytorch.org/docs/>
  作为本 Skill 实现的所有算子的语义基准（`torch.add`、`torch.softmax`、`torch.matmul`、`torch.nn.functional.scaled_dot_product_attention`、`torch.conv2d`、`torch.addmm`、`torch.bmm`、`torch.swiglu`、`torch.rotary_pos_emb`、`torch.max_pool2d` 等）。`scripts/gen_pytorch_oracle.py` 与 `tests/test_examples.py` 中的正确性 oracle 均以 PyTorch 输出为真值。

### Triton

- **Documentation** — <https://triton-lang.org/>
  NineToothed 最终下降到 Triton。`references/FIX_CARDS.md` 中的诊断卡片（编译失败、`tl` dtype 不匹配、atomic RMW、`program_id` 布局）均基于 Triton 语义。

### MetaX GPU (沐曦)

- **Vendor documentation** — MetaX C500 / MX-SMI 手册（由沐曦官方提供）。
  使用范围：远程执行环境（`MACA_PATH`、`LD_LIBRARY_PATH` 注入）、`mx-smi` 监控、`references/OPTIMIZATION_GUIDE.md` 中的双平台 benchmark 说明。

## Competition Materials

- NineToothed `.skill` Innovation Track 赛题说明与规则（由组委会提供）。
- 竞赛仓库中的官方参考 Skill —— 仅参考结构规范，未复制任何参考 Skill 代码。

## AI Assistance Disclosure

本 Skill 包在开发过程中使用了 AI 辅助。辅助范围包括：

- 组织工作流指令与决策树（Step 0 → Step 3、`📋 IMPLEMENTATION PROPOSAL` 契约）。
- 撰写 `references/CODE_TEMPLATES.md` 与 `references/FIX_CARDS.md` 中的可复用模板与诊断卡片。
- 准备 `scripts/` 下的复现与验证脚本。

参赛者对每一条被交付的结论负全责：正确性数据、benchmark 数据、环境细节、参赛者身份均须由参赛者在提交前亲自复现与核验。

## Third-Party Code

本包包含：

- 原创说明文本与工作流定义。
- 原创诊断与验证脚本。
- 对公开 NineToothed 模式的引用（未搬运 NineToothed 源码）。

未包含竞赛未公开数据、隐藏评测答案或针对赛题的 bypass 逻辑（详见 `HONOR_CODE.md`）。
