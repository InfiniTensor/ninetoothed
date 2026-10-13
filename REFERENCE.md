# Reference

## 1. NineToothed 官方示例（内核模式）

| 项目 | 链接 | 使用内容 |
|------|------|----------|
| ninetoothed-examples | https://github.com/InfiniTensor/ninetoothed-examples | `ops/ninetoothed/kernels/add.py`、`silu.py` 的 arrangement/application/make 结构；用于 `scaffold_kernel.py` 模板设计与 `reference.md` |
| 许可证 | Apache-2.0 | 见上游 LICENSE |

**使用方式**：仅借鉴文件结构与 API 用法；本仓库脚本生成的是带 TODO 的骨架，非复制粘贴完整实现。

## 2. 文档

| 资源 | 链接 |
|------|------|
| NineToothed 文档 | https://ninetoothed.org/ |
| ntops | https://github.com/InfiniTensor/ntops |
| 大赛指南 | 2026 春季启元人工智能大赛指南（主办方发布） |

## 3. 无外部代码的组件

- `scripts/preflight.py` — 本人编写（Python ast 标准库）  
- `scripts/pack_submission.py` — 本人编写  
- `skills/ntops-forge/SKILL.md`、`skills/ntops-copilot/SKILL.md` — 本人编写  
- `scripts/forge.py` 及流水线脚本 — 本人编写
