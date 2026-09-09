# Reference 引用披露

## 引用资料

| 资料 | 来源 | 用途 |
|------|------|------|
| NineToothed DSL 框架 | https://github.com/InfiniTensor/ninetoothed | 核心 DSL 框架，SKILL.md 指导 AI 智能体使用的目标平台 |
| ntops 算子库 | https://github.com/InfiniTensor/ntops | 算子开发目标仓库，模板中引用的 arrangement 模式来源 |
| Triton | https://github.com/triton-lang/triton | NineToothed 的底层编译后端 |
| PyTorch | https://pytorch.org | 精度基线、性能对比基准、torch 层实现 |
| CUDA libdevice | NVIDIA CUDA Toolkit | ntl.libdevice 函数参考（publi c API） |

## 外部代码引用

本 .skill 中所有代码模板（`references/code_templates.md`）基于 NineToothed 公开的 arrangement 模式编写，模板中标注了占位符（`{COMPUTATION}`、`{OP}`）表示需填充的计算逻辑。

`references/ntl_api.md` 基于 NineToothed 公开 API 编写，为原创整理的参考文档。

## 生成式 AI 辅助范围

本 .skill 的开发过程中，使用了 Claude Code (Anthropic) 作为 AI 智能体进行以下辅助：

- 协助编写 SKILL.md 的结构和内容
- 协助调试算子代码（30 个算子的编译错误和精度问题）
- 协助运行 benchmark 和生成报告
- 协助撰写中期报告和最终报告

所有最终提交内容均经过人工审查和验证。

## 第三方依赖

| 依赖 | 版本要求 | License | 用途 |
|------|----------|---------|------|
| Python | ≥3.10 | PSF | 运行环境 |
| PyTorch | ≥2.0 | BSD | CPU 参考实现 + torch 层 |
| Triton | ≥3.0 | MIT | GPU kernel 编译后端 |
| NineToothed | ≥0.25 | 待确认 | DSL 框架 |
| ntops | ≥0.1 | 待确认 | 算子库 |
| pytest | ≥7.0 | MIT | 测试框架（仅自测） |

## 未授权内容

本 .skill 不包含任何未授权的第三方代码、数据或模型权重。
