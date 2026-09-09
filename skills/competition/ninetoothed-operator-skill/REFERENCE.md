# Reference Disclosure / 引用披露



## Official Materials / 官方材料



| Source / 来源 | URL / Location | Usage / 用途 |

|---------------|---------------|-------------|

| NineToothed Repository | https://github.com/InfiniTensor/ninetoothed | 参考仓库结构、API 设计、测试写法和算子实现模式 |

| NineToothed Documentation | https://ninetoothed.readthedocs.io/ | 查阅 arrangement、application、build 等官方文档 |

| ntops Operators | https://github.com/InfiniTensor/ntops | 参考已有算子实现风格 |

| ninetoothed-examples | https://github.com/InfiniTensor/ninetoothed-examples | 参考示例项目结构 |

| 比赛规则 v0.8 | 赛道官方 PDF | 确认 skill 包结构要求、自测任务覆盖维度、评分标准 |

| 操作手册 | 赛道官方 DOCX | 了解 skill 编写规范、提交流程和 PR 要求 |

| CONTRIBUTING.md | 仓库根目录 `CONTRIBUTING.md` | 遵循分支命名、commit message、ruff 检查等工程规范 |

| PR Template | `.github/pull_request_template.md` | 参照 PR 描述格式填写必要信息 |



## External Code / 外部代码



| Source | URL / Location | Usage |

|--------|----------------|-------|

| NineToothed tests/test_add.py | 仓库 `tests/test_add.py` | 参考 elementwise 算子测试的 parametrize、device 管理、allclose 写法 |

| NineToothed tests/test_softmax.py | 仓库 `tests/test_softmax.py` | 参考 reduction 算子测试的归一化检查、数值容差设置 |

| NineToothed tests/test_clone.py | 仓库 `tests/test_clone.py` | 参考 layout-sensitive 算子的 non-contiguous、stride、storage offset 测试构造 |

| NineToothed tests/test_debugging.py | 仓库 `tests/test_debugging.py` | 参考失败复现与调试模式 |

| NineToothed src/ninetoothed/tensor.py | 仓库 `src/ninetoothed/tensor.py` | 确认 `tile()`、`permute()`、`offsets()`、`stride()` 等 API 语义 |

| NineToothed docs/source/basics.rst | 仓库 `docs/source/basics.rst` | 理解 arrangement、application、arrange-and-apply 概念 |

| NineToothed docs/source/build.rst | 仓库 `docs/source/build.rst` | 理解 AOT build、`ninetoothed.build`、`premake`、`output_dir` 用法 |



No third-party code outside the official NineToothed repository was used.

除 NineToothed 官方仓库外，未使用任何第三方代码。



## Documentation & Papers / 文档与论文



| Title | Link | Notes |

|-------|------|-------|

| PyTorch Documentation | https://pytorch.org/docs/stable/ | 参考 `torch.allclose`、`torch.softmax`、`torch.cuda.synchronize` 等 API 语义 |

| PyTorch Broadcasting Semantics | https://pytorch.org/docs/stable/notes/broadcasting.html | 确认 broadcast 规则，用于 elementwise 算子契约 |

| Agent Skills Overview | Cursor 官方文档 | 参考 .skill 包结构规范 |



No academic papers were referenced for this skill.

本 skill 未参考学术论文。



## AI Assistance / AI 辅助



| Tool / Model | Purpose / 用途 | Sections Affected / 影响范围 |

|--------------|----------------|------------------------------|

| Codex (OpenAI) | 生成算子实现、测试和 benchmark 代码 | `examples/task-02/` ~ `examples/task-04/` 全部代码文件 |

| Cursor | 生成 task-01 算子代码、文档收尾、合规审查 | `examples/task-01/`；`SKILL.md`、`README.md`、赛题报告、`.gitignore` |

| Kimi (Moonshot AI) | 生成 SKILL.md 和文档内容 | `SKILL.md` 核心工作流初版；Git 工作流指导 |

| WorkBuddy | 生成合规文档、执行文件写入和审查 | `HONOR_CODE.md`、`REFERENCE.md` 初版；目录创建与审查 |



### AI 辅助范围说明



- **代码生成：** Codex 和 Cursor 生成的代码均经本人审核，确认 API 调用符合 NineToothed 仓库实际接口，测试逻辑真实有效。

- **文档生成：** Kimi、Cursor 和 WorkBuddy 生成的文档经本人校对，确认内容与代码实现一致，GPU 环境信息已如实披露。

- **审查工作：** 合规审查报告帮助发现占位文件、缺失目录等问题，修复决策由本人做出。

- **未使用 AI 的部分：** 参赛者信息、最终提交决策、环境限制确认均由本人独立完成。



## Environment / 环境

| Item | Status |
|------|--------|
| Development OS | Windows 11 |
| GPU | NVIDIA RTX 5060 Laptop GPU (8GB) |
| CUDA | 12.8 |
| PyTorch | 2.12.0.dev20260408+cu128 |
| ninetoothed | 0.26.0 |
| pytest execution | All tests PASSED on GPU |
| Benchmark results | Collected on RTX 5060 |



## Other References / 其他引用



- 无其他需披露的引用。

- No other references require disclosure.



## Offline Reproducibility / 离线可复现性



This skill does not depend on any online service for execution. All code references point to the local NineToothed repository. PyTorch is required as the reference implementation for correctness tests. All tests and benchmarks were verified on Windows 11 + RTX 5060 + CUDA 12.8.

本 skill 执行时不依赖任何在线服务。所有代码引用均指向本地 NineToothed 仓库。PyTorch 作为正确性测试的参考实现。所有测试和 benchmark 已在 Windows 11 + RTX 5060 + CUDA 12.8 环境验证。

