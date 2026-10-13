# T3-1-1 — nt-devskill

> 赛题：`T3-1-1`
> 小组：`新建队伍名`
> Skill 名称：`nt-devskill`
> 版本：`1.7`
> 提交日期：`2026-07-12`

---

## 1. Skill 名称与赛题信息

| 字段       | 值                                |
| ---------- | --------------------------------- |
| Skill 名称 | `nt-devskill`                     |
| 赛题编号   | `T3-1-1`                          |
| 小组名称   | `新建队伍名`                      |
| 版本       | `1.7`                             |
| 主文件     | `SKILL.md`（914 行）              |
| 配套 MCP   | `remote-operator-mcp`（1.0.0）    |
| 安装位置   | `skills/competition/nt-devskill/` |

---

## 2. 适用任务范围与不适用范围

### 2.1 适用

- **12 个内置示例算子**：add / silu / softmax / matmul / bmm / addmm / sdpa / fused_rms_norm / swiglu / conv2d / rope / max_pool2d
- **9 族算子分类**：elementwise、reduction、matmul-like、attention、conv、generator、layout/view、scatter、composition
- **正确性测试矩阵**：shape × dtype × layout，含 MERE/MARE 精度指标
- **性能基准**：E2E + GPU-only + overhead breakdown + Roofline 模型
- **生成源码审查**与 **arrangement 调试**
- **非连续 / stride / offset 敏感任务**
- **MetaX C500 + CUDA GPU 双平台**

### 2.2 不适用

- NineToothed 编译器内核改造
- 隐藏评测答案或针对赛题硬编码的 bypass
- 需要私有凭证 / 在线服务的算子
- **完全无数据搬运的纯 view-only 操作**（如 `reshape`、`transpose` 返回非连续 view）：Skill 会在 `Step 0.8` 主动告知 view-only 性质并建议退回 `torch` 原生路径
  > 注：`pixel_unshuffle` 虽属 layout/view 族，但在 MetaX 上已返回 contiguous，必须做真数据搬运 → 已在自测 5 中通过 **Pattern 13 wrapper-heavy 路径**成功用九齿实现（详见 `测试场地/ntops14/自测结果_pixel_unshuffle_with_skill.md`）

---

## 3. 安装与使用方式

### 3.1 安装

把本目录作为只读 Skill 包注册到 Agent（以 Qoder 为例）：

```bash
# 项目级（推荐，跟随仓库）
cp -r nt-devskill/  .qoder/skills/nt-devskill/

# 用户级（全局可用）
cp -r nt-devskill/  ~/.qoder-cn/skills/nt-devskill/
```

> 其他 Agent 的安装指引详见 `README.md` 第 2 节（Cursor / Claude Code / Windsurf / Cline / Codex CLI）。

#### 3.1.1 配套 MCP：`remote-operator-mcp`

为了让 SKILL.md 中的 `Step 0 → Step 3` 闭环能在远程 MetaX GPU 上实跑，本项目配套提供了一个独立 MCP Server（`remote-operator-mcp/`），提供 `upload_code / run_test / run_python / auto_bench / mx_smi / remote_ls / remote_cat / remote_md5 / remote_glob / remote_rm / download_file / ping` 共 12 个工具。Skill 与 MCP 解耦：Skill 不依赖任何特定 MCP 实现，但推荐搭配使用。

**安装步骤**：

```bash
# 1. 进入 MCP 目录
cd skills/competition/remote-operator-mcp/

# 2. 安装依赖（Python 3.12+；使用 uv 管理）
uv sync

# 3. 复制并填写 SSH 连接配置（请勿把真实密码/密钥路径提交到仓库）
cp config.example.json config.json
# 编辑 config.json：host / port / user / password 或 key_filename / remote_root

# 4. 启动测试
uv run python remote_operator_mcp.py --help
```

**注册到常用 Agent**（在 MCP 客户端配置中加一个 stdio server）：

| Agent                       | 配置文件路径                                                   | 配置示例                                                                                                                                                 |
| --------------------------- | -------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Qoder (qoderclicn)**      | `~/.qoder-cn/mcp.json`（用户级）或 `.qoder/mcp.json`（项目级） | `{"mcpServers":{"remote-operator":{"command":"uv","args":["--directory","/abs/path/to/remote-operator-mcp","run","python","remote_operator_mcp.py"]}}}`  |
| **Cursor**                  | Cursor Settings → MCP → `+ Add new MCP server`                 | Type: `command`；Name: `remote-operator`；Command: `uv --directory /abs/path/to/remote-operator-mcp run python remote_operator_mcp.py`                   |
| **Claude Code (Anthropic)** | `~/.claude.json` 或项目 `.mcp.json`                            | `{"mcpServers":{"remote-operator":{"command":"uv","args":["--directory","/abs/path/to/remote-operator-mcp","run","python","remote_operator_mcp.py"]}}}`  |
| **Windsurf (Cascade)**      | Windsurf Settings → MCP → Edit Config                          | 同上 `command/args` stdio 配置                                                                                                                           |
| **Cline / Roo Code**        | VSCode `settings.json` → Cline MCP Settings                    | `{"mcpServers":{"remote-operator":{"command":"uv","args":["--directory","/abs/path/to/remote-operator-mcp","run","python","remote_operator_mcp.py"]}}}`  |
| **OpenAI Codex CLI**        | `~/.codex/config.toml` 或项目 `codex.toml`                     | `[mcp_servers.remote-operator]` `command = "uv"` `args = ["--directory", "/abs/path/to/remote-operator-mcp", "run", "python", "remote_operator_mcp.py"]` |

> 通用规则：所有 Agent 都是 stdio 模式；`--directory` 必须填 MCP 目录的绝对路径；`config.json` 必须在 MCP 目录下且含有效 SSH 凭据（不要把 `config.json` 提交到仓库，`.gitignore` 已排除）。

#### 3.1.2 MCP 工具一览

| 工具            | 作用                                                                             |
| --------------- | -------------------------------------------------------------------------------- |
| `ping`          | 返回 `pong`，确认 MCP 可用                                                       |
| `upload_code`   | 上传本地文件/目录到远端（目录自动 tar.gz）                                       |
| `download_file` | 下载远端文件/目录到本地                                                          |
| `run_test`      | 在远端执行命令；`env="default"` 自动注入 MetaX 环境变量；自动 tee 日志到 `logs/` |
| `run_python`    | 在远端运行内联 Python（base64 传输，避免 shell 转义）                            |
| `auto_bench`    | 一键对比 v0/v1 算子；`runs>1` 时给 median/min/max 聚合                           |
| `remote_ls`     | 列出远端目录                                                                     |
| `remote_cat`    | 读远端文件（支持 `lines` 截断）                                                  |
| `remote_md5`    | 计算远端文件 MD5                                                                 |
| `remote_glob`   | 远端 glob 查找（支持 `**`）                                                      |
| `remote_rm`     | 删除远端文件/目录（内置危险路径保护）                                            |
| `mx_smi`        | 查看 MetaX GPU 状态                                                              |

#### 3.1.3 Skill + MCP 协同验证

在 Qoder 中验证两者已联通：

```text
# 1) 确认 MCP 可用
→ 输入：调用 ping
← 期望：pong

# 2) 确认 Skill 可用
→ 输入：实现一个九齿算子：add(input, other)
← 期望：Agent 自动走 Step 0 → 检测到 MCP → 走"远程 GPU"路径 → upload_code → run_test → 闭环
```

> `Step 0` 的环境探测会检查 `run_test / upload_code / run_python` 等 MCP 工具是否可用；可用则选"MCP 远程"策略（详见 `SKILL.md §Step 0`）。

### 3.2 触发

在 Agent 中输入任意九齿算子需求即可自动触发，例如：

```text
实现一个九齿算子：nansum(input, dim, keepdim)
支持 fp32/fp16，NaN 视为 0，参考 torch.nansum
```

或显式触发：

```text
请使用 nt-devskill 帮我实现一个九齿算子
```

### 3.3 工作流

Agent 会自动按以下步骤执行：

```text
Step 0   分析工程目录 + 探测 GPU/MCP/SSH 环境
Step 0.5 输出 CAPABILITY REPORT（九齿是否能表达）
Step 0.8 输出 📋 IMPLEMENTATION PROPOSAL（分类/方案/风险/备选）
Step 1   按 TAXONOMY.md 路由到 9 族之一
Step 2   生成 arrangement + application + wrapper + test 四件套
Step 2.5 进入 pytest 闭环（失败 → FIX_CARDS → 修复 → 重测）
Step 2.8 Tile sweep + Roofline 性能优化
Step 3   审计笔记（写入自测结果报告）
```

### 3.4 验证命令

```shell
# 环境体检
python scripts/doctor.py

# 全流水线
python scripts/pipeline.py run --op add

# 正确性 / 性能 / 诊断
python scripts/validate.py --op softmax
python scripts/benchmark.py --op matmul
python scripts/diag_overhead.py --op add
python scripts/diag_tile_sweep.py --op add
```

---

## 4. 自测任务 / 自测案例的运行记录

5 个自测任务覆盖 5 个算子族（超出赛题"至少 4 个"的最低要求），完整记录见各 `自测结果_*.md` 文件：

| #   | 算子              | 族                       | 工程目录           | 自测报告                                 |
| --- | ----------------- | ------------------------ | ------------------ | ---------------------------------------- |
| 1   | `copysign`        | elementwise（libdevice） | `测试场地/ntops11` | `自测结果_copysign_with_skill.md`        |
| 2   | `nansum`          | reduction                | `测试场地/ntops7`  | `自测结果_nansum_with_skill.md`          |
| 3   | `addcmul`         | composition（诊断/修复） | `测试场地/ntops9`  | `自测结果_addcmul_diag_and_fix.md`       |
| 4   | `scatter_add`     | scatter（atomic RMW）    | `测试场地/ntops13` | `自测结果_scatter_add_with_skill.md`     |
| 5   | `pixel_unshuffle` | layout/view（重排）      | `测试场地/ntops14` | `自测结果_pixel_unshuffle_with_skill.md` |

**运行记录摘要**：

| 算子            | 提示词                                        | Step 0                                                                 | Step 1              | Step 2                                    | Step 2.5 失败轮数               | Step 2.8                        |
| --------------- | --------------------------------------------- | ---------------------------------------------------------------------- | ------------------- | ----------------------------------------- | ------------------------------- | ------------------------------- |
| copysign        | 实现 copysign，支持 fp32/fp16 + IEEE 754 ±0.0 | MetaX C500 探测                                                        | ElementWise         | libdevice + bitcast 双路径                | **0 轮**                        | tile sweep 6 档                 |
| nansum          | 实现 nansum，NaN 视为 0                       | MetaX C500 探测                                                        | Reduction           | NaN_mask = (x != x)                       | **1 轮**（FC-17）               | tile/warps sweep                |
| addcmul         | 诊断 fp16 精度 bug 并最小修复                 | MetaX C500 探测                                                        | Composition         | FC-15 最小修复                            | **0 轮**（已修复）              | Before/After 对照               |
| scatter_add     | 实现 scatter_add                              | MetaX C500 探测                                                        | Scatter（未知族）   | atomic_add + fp32 buffer                  | **1 轮**（FC-15 延伸）          | 已知性能上限                    |
| pixel_unshuffle | 实现 pixel_unshuffle（3D/4D）                 | MetaX C500 探测；发现 `F.pixel_unshuffle` 在 MetaX 上已返回 contiguous | Layout/view（重排） | Pattern 13 1D copy + wrapper view+permute | **2 轮**（ndim / permute 顺序） | block sweep 7 档；大 shape 反超 |

---

## 5. 自测结果：AI 智能体使用 Skill 前后的对比

### 5.1 正确性对比

| 算子            | 测试总数 | With Skill PASS | With Skill FAIL |
| --------------- | -------: | --------------: | --------------: |
| copysign        |       12 |              12 |               0 |
| nansum          |       46 |              46 |               0 |
| addcmul         |        4 |               4 |               0 |
| scatter_add     |       65 |              65 |               0 |
| pixel_unshuffle |       12 |              12 |               0 |
| **合计**        |  **139** |         **139** |           **0** |

### 5.2 A/B 对照（copysign，唯一做无 Skill 对照的算子）

> 详细报告见 `测试场地/copysign_AB_汇总.md`

| 维度                     | A. With Skill                    | B. Without Skill          |
| ------------------------ | -------------------------------- | ------------------------- |
| 首次上传到 GPU 即 PASS   | ✅ 12/12 in 14.19 s               | ❌ 未上传（仅本地 parse）  |
| 实测正确性（MetaX C500） | ✅ fp32/fp16 + ±0.0 + nan/inf     | ❌ 无任何 GPU 验证         |
| Benchmark 数据           | ✅ 6 组 shape×dtype               | ❌ 无                      |
| 性能调优动作             | tile sweep 6 档，block 1024→4096 | 无                        |
| 代码迭代轮数             | 1 轮（一次成稿）                 | 3 轮（V1→V2→V3 两次推翻） |
| Wrapper 边界检查         | shape + dtype 双 assert          | 无                        |
| IEEE 754 ±0.0 陷阱       | Skill 文档明确禁止 `< 0.0`       | Agent 自悟踩坑后自纠      |
| MCP 远程测试             | ✅ 完成闭环                       | ❌ 未调用                  |

### 5.3 Benchmark 结果

| 算子            | 输入规模                  | dtype |     speedup | 备注                              |
| --------------- | ------------------------- | ----- | ----------: | --------------------------------- |
| copysign        | (4096,4096)               | fp32  | **0.99x** ✓ | memory-bound                      |
| copysign        | (4096,4096)               | fp16  | **0.98x** ✓ | memory-bound                      |
| nansum          | (1024,1024)               | fp32  |       0.47x | Triton + NaN 指令开销（框架上限） |
| addcmul         | (4096,4096)               | fp16  |           — | Before/After 无性能回退           |
| scatter_add     | 2D 2048×512 high-conflict | fp32  |       0.26x | atomic contention（已知上限）     |
| pixel_unshuffle | (2,3,64,64) r=2 4D        | fp32  |       0.26x | launch overhead 框架上限          |
| pixel_unshuffle | (8,64,256,256) r=2 4D     | fp32  | **1.08x** ✓ | memory-bound（nt 反超 torch）     |
| pixel_unshuffle | (8,64,256,256) r=2 4D     | fp16  | **1.09x** ✓ | memory-bound（nt 反超 torch）     |

### 5.4 对照结论

| 维度           | 结论                                                    |
| -------------- | ------------------------------------------------------- |
| **完成度**     | A 版端到端闭环，B 版停在"写完未测"                      |
| **正确性信心** | A 版实测 + IEEE 754 专项，B 版仅纸面分析                |
| **性能信心**   | A 版有 6 档 tile sweep + 最终 benchmark；B 版无数据     |
| **迭代成本**   | A 版 1 轮成稿，B 版 3 轮（2 次自我推翻）                |
| **关键坑规避** | Skill 文档把 `ntl.where(< 0.0, ...)` 列为反例，直接绕过 |

> **一句话**：Skill 的主要价值不是"写代码更快"，而是**把已知的坑提前告诉 Agent**，避免无谓的迭代与未验证就交付。

---

## 6. 配套 MCP：`remote-operator-mcp`

为了让 `Step 0 → Step 3` 的闭环能在远程 MetaX GPU 上实跑，本项目在 `remote-operator-mcp/` 目录下附带一个独立 MCP Server（Python 3.12 + uv 管理；依赖 `mcp>=1.27.2` 与 `paramiko>=5.0.0`）。Skill 与 MCP **解耦**：Skill 不绑定任何特定 MCP 实现，但 SKILL.md 中所有"远程 GPU"路径默认以本 MCP 的工具命名为示例。

### 6.1 12 个 MCP 工具

| 类别         | 工具                                                                    |
| ------------ | ----------------------------------------------------------------------- |
| 连接与传输   | `ping` · `upload_code` · `download_file`                                |
| 测试与运行   | `run_test` · `run_python` · `auto_bench`                                |
| 远端文件操作 | `remote_ls` · `remote_cat` · `remote_md5` · `remote_glob` · `remote_rm` |
| GPU 监控     | `mx_smi`                                                                |

`run_test` 的 `env="default"` 会自动注入 MetaX 必需的环境变量（`MACA_PATH` / `LD_LIBRARY_PATH` 等），Python 命令自动 `tee` 到 `logs/<script>_<timestamp>.log`；`auto_bench` 支持 `runs>1` 输出 median/min/max 聚合；`remote_rm` 内置危险路径保护（禁止删 `remote_root` / `/data` / `/opt`）。

### 6.2 安装到常用 Agent

所有 Agent 均为 stdio 模式，`--directory` 必须填 MCP 目录的绝对路径；`config.json` 必须在 MCP 目录下且含有效 SSH 凭据。

| Agent                        | 配置文件                                                       | 配置片段                                                                                                                                                |
| ---------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Qoder (qoderclicn)**       | `~/.qoder-cn/mcp.json`（用户级）或 `.qoder/mcp.json`（项目级） | `{"mcpServers":{"remote-operator":{"command":"uv","args":["--directory","/abs/path/to/remote-operator-mcp","run","python","remote_operator_mcp.py"]}}}` |
| **Cursor**                   | Cursor Settings → MCP → `+ Add new MCP server`                 | Type `command`；Name `remote-operator`；Command `uv --directory /abs/path/to/remote-operator-mcp run python remote_operator_mcp.py`                     |
| **Claude Code (Anthropic)**  | `~/.claude.json` 或项目 `.mcp.json`                            | 同 Qoder                                                                                                                                                |
| **Windsurf (Cascade)**       | Windsurf Settings → MCP → Edit Config                          | 同上 stdio 配置                                                                                                                                         |
| **Cline / Roo Code**         | VSCode `settings.json` → Cline MCP Settings                    | `{"mcpServers":{"remote-operator":{"command":"uv","args":["--directory","/abs/path/to/remote-operator-mcp","run","python","remote_operator_mcp.py"]}}}` |
| **OpenAI Codex CLI**         | `~/.codex/config.toml` 或项目 `codex.toml`                     | `[mcp_servers.remote-operator]` `command = "uv"` `args = ["--directory","/abs/path/to/remote-operator-mcp","run","python","remote_operator_mcp.py"]`    |
| **GitHub Copilot Workspace** | 在 Workspace MCP 面板中添加 stdio server                       | 同上 command / args                                                                                                                                     |

### 6.3 使用示例（Qoder 内）

```text
# 1. 探测 MCP 是否在线
→ ping
← pong

# 2. 上传算子工程到远端
→ upload_code(src="./ntops7", dst="ntops7/")

# 3. 跑正确性测试（env="default" 自动注入 MetaX 环境变量）
→ run_test(cmd="python tests/test_nansum.py", workdir="ntops7")
← 自动 tee 到 logs/test_nansum_<timestamp>.log

# 4. 一键 v0/v1 对比
→ auto_bench(v0="05TriAttentionFallback.py", v1="05TriAttentionFallback_MateX.py",
             workdir="05_diag", runs=3)
← 输出 PASS/FAIL + speedup + median/min/max

# 5. 查看 GPU
→ mx_smi(options="--show-usage")
```

### 6.4 安全

- `config.json` 含 SSH 密码或密钥路径 → 必须列入 `.gitignore`，不得提交公共仓库
- `remote_rm` 默认只删文件，删目录需显式 `recursive=True`；对 `remote_root` / `/data` / `/opt` 等路径直接拒绝
- `run_test` 与 `run_python` 在 SSH session 内运行，与 MCP 客户端进程隔离

> 完整的安装与工具文档见 `remote-operator-mcp/README.md` 与 `remote-operator-mcp/config.example.json`。

---

## 7. 署名的 HONOR_CODE.md 和 REFERENCE.md

- ✅ **`HONOR_CODE.md`**：已在根目录签署，确认：
  - 不含隐藏答案或针对性 bypass
  - 不含硬编码赛题名或评测特定路径
  - 不含密钥、凭证、私有 token、未授权数据
  - 不指示 Agent 删除测试、伪造 benchmark、绕过验证或隐藏失败
  - 引用已在 `REFERENCE.md` 披露
  - 所有结果可复现

- ✅ **`REFERENCE.md`**：已披露：
  - 公开引用来源（NineToothed / PyTorch / Triton / MetaX 文档）
  - 竞赛材料（赛题陈述与规则）
  - AI 辅助声明（明确 AI 在指令组织、模板创建、脚本编写中的作用）
  - 第三方代码归因（未 vendor NineToothed 源码，仅参考其 examples）

---

## 8. Proposal 与最终赛题报告

| 文档             | 路径                                                                           |
| ---------------- | ------------------------------------------------------------------------------ |
| **最终赛题报告** | `新建队伍名_九齿skill创新挑战_T3-1-1_赛题报告.pdf`                             |
| **proposal**     | `董俊宏_九齿skill创新挑战_proposal.pdf`                                        |
| **PR 描述模板**  | `PR_DESCRIPTION_TEMPLATE.md`（本文件）                                         |
| **自测报告 1**   | `测试场地/ntops11/自测结果_copysign_with_skill.md`                             |
| **自测报告 2**   | `测试场地/ntops7/自测结果_nansum_with_skill.md`                                |
| **自测报告 3**   | `测试场地/ntops9/自测结果_addcmul_diag_and_fix.md`                             |
| **自测报告 4**   | `测试场地/ntops13/自测结果_scatter_add_with_skill.md`                          |
| **自测报告 5**   | `测试场地/ntops14/自测结果_pixel_unshuffle_with_skill.md`（Step 0.8 正面验证） |
| **A/B 对照**     | `测试场地/copysign_AB_汇总.md`                                                 |
| **MCP 文档**     | `remote-operator-mcp/README.md`                                                |

> 最终报告 `新建队伍名_九齿skill创新挑战_T3-1-1_赛题报告.pdf` 包含 8 个完整章节：
> 1. `.skill` 目标、设计原则与包结构
> 2. 核心工作流说明
> 3. 自测算子任务、运行过程与 correctness 结果
> 4. benchmark 设计、输入规模、性能结果与回退分析
> 5. 失败诊断案例、修复过程或规避建议
> 6. 与不使用 `.skill` 的 AI 智能体基线对比
> 7. 安全、依赖、授权与引用披露（**含配套 MCP `remote-operator-mcp` 说明**）
> 8. 后续可维护计划

---

## 9. 合规清单

- [x] `HONOR_CODE.md` 已签署
- [x] `REFERENCE.md` 已披露公开来源与 AI 辅助
- [x] `README.md` 已说明安装与使用（含多 Agent 安装指引）
- [x] `新建队伍名_九齿skill创新挑战_T3-1-1_赛题报告.pdf` 最终报告已完成 8 章
- [x] 5 个自测报告已按 `自测结果模版.md` 完成（覆盖 elementwise / reduction / composition / scatter / layout-view 5 族）
- [x] 所有命令可复现
- [x] 所有失败案例已披露（含 scatter_add 绕过九齿事故 + pixel_unshuffle Step 0.8 正面验证 + pixel_unshuffle permute 顺序错 Case G）
- [x] 无密钥、凭证、隐藏答案、针对性 bypass
- [x] 不修改测试、不放宽 tolerance
- [x] **配套 MCP（`remote-operator-mcp/`）已交付**：`README.md` 含 12 个工具说明 + `config.example.json` 不含真实凭据 + 多 Agent 安装指引已覆盖
- [x] **MCP 安全**：`remote_rm` 含危险路径保护；`config.json` 未提交（已 `.gitignore`）
