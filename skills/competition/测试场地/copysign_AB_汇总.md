# copysign 自测 A/B 汇总（Skill vs 无 Skill）

> A 版（Skill 辅助）：`ntops11/自测结果_copysign_with_skill.md`
> B 版（无 Skill）：`ntops12/自测结果_copysign_no_skill.md`

## 一、关键指标对照表

| 维度 | A. Skill 辅助 (ntops11) | B. 无 Skill (ntops12) |
|---|---|---|
| 首次上传到 GPU 即 PASS | ✅ 12/12 in 14.19 s | ❌ 未上传（仅本地 parse） |
| 实测正确性（MetaX C500）| ✅ fp32/fp16 + ±0.0 + nan/inf | ❌ 无任何 GPU 验证 |
| Benchmark 数据 | ✅ 6 组 shape×dtype | ❌ 无 |
| 性能调优动作 | tile sweep 6 档，block 1024→4096 | 无 |
| 代码迭代轮数 | 1 轮（一次成稿） | 3 轮（V1→V2→V3 两次推翻） |
| 测试用例数 | 12 | 10（缺 signed-zero 专项） |
| Wrapper 边界检查 | shape + dtype 双 assert | 无 |
| IEEE 754 ±0.0 陷阱 | Skill 文档明确禁止 `< 0.0` | Agent 自悟踩坑后自纠 |
| 最终 kernel 风格 | libdevice (fp32) + bitcast (fp16) | bitcast + `ntl.where` 统一 |

## 二、Skill 价值拆解

### 1. 分类与规范（TAXONOMY / CODE_TEMPLATES）
- A 版在 Step 1 明确归类为 `ElementWise`，直接定位 `element_wise.arrangement` 与双输入模板。
- B 版通过 grep 已有算子反推模式，多花了约 30 行 thinking 才确认 `mul.py` 为模板。

### 2. 避免 ±0.0 语义陷阱（API_REFERENCE.md § libdevice.copysign）
- Skill 文档明确写道：
  > `output = ntl.where(y >= 0, ntl.abs(x), -ntl.abs(x))  # copysign 错误！不处理 -0.0`
- A 版**直接绕过该坑**，一步到位走 libdevice/bitcast。
- B 版**先踩再爬出**：V2 用了 `< 0.0`，写到一半自己意识到 IEEE 754 问题再推翻。
  这一轮"自我否定"浪费约 200 行对话上下文。

### 3. dtype 处理策略（OPTIMIZATION_GUIDE）
- A 版依据文档：fp16 走 bitcast（libdevice.copysign 对 fp16 未定义），fp32 走 libdevice。
- B 版因怀疑 `ntl.uint32(...)` 构造器非法而整体推翻，最终选了更绕的 `ntl.where` 路径，
  虽然功能正确但走了弯路。

### 4. 性能优化（OPTIMIZATION_GUIDE / scripts/diag_tile_sweep.py）
- A 版直接复用 `diag_tile_sweep.py` 模式，自写 `bench_tile_sweep_copysign.py`，
  在 6 档 block_size 中找到最优 4096，使 `(4096,4096) fp16` 从 0.50x 提升到 0.98x。
- B 版**完全没有性能阶段**，连 benchmark 脚本都未生成。

### 5. 远端 MCP 工作流
- A 版激活 Skill 后自动 `ping` → `run_python` → `upload_code` → `run_test`，
  完成"本地写、远端测"的闭环。
- B 版未调用任何 MCP 工具，agent 自承"本地无 CUDA，无法验证"，任务处于半完成状态。

## 三、A/B 性能对照（仅 A 版有数据）

| 输入规模 | dtype | A 版 speedup (block=4096) | B 版 |
|---|---|---|---|
| (1024,) | fp32 | 0.18x | — |
| (1024,) | fp16 | 0.17x | — |
| (1024,1024) | fp32 | 0.30x | — |
| (1024,1024) | fp16 | 0.20x | — |
| (4096,4096) | fp32 | **0.99x** ✓ | — |
| (4096,4096) | fp16 | **0.98x** ✓ | — |

## 四、总体结论

1. **完成度**：A 版是端到端闭环（写→测→调→验），B 版停在"写完未测"，需要用户手动推到 GPU。
2. **正确性信心**：A 版实测 12/12 + IEEE 754 专项，B 版仅纸面分析。
3. **性能信心**：A 版有 6 档 tile sweep + 最终 benchmark；B 版无数据。
4. **迭代成本**：A 版 1 轮成稿，B 版 3 轮（2 次自我推翻）。
5. **关键坑规避**：Skill 文档把 `ntl.where(< 0.0, ...)` 显式列为反例，让 A 版直接绕过 IEEE 754 signed-zero 陷阱；B 版靠 agent 自省躲过，但付出了额外的对话轮次与风险。

> **一句话**：对 copysign 这种"看似简单、实则有 IEEE 754 隐藏坑"的算子，Skill 的主要价值
> 不是"写代码更快"，而是**把已知的坑提前告诉 agent**，避免无谓的迭代与未验证就交付。
