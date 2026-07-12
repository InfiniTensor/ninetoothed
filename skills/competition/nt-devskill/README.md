# nt-devskill

AI 驱动的 NineToothed（九齿）GPU 算子开发助手：自动生成、验证、诊断与基准测试。

- 赛题：`T3-1-1`（NineToothed `.skill` Innovation Track）
- 小组：`新建队伍名`
- Skill 名称：`nt-devskill`
- 版本：`1.7`

---

## 1. 概述

`nt-devskill` 是一套面向 **NineToothed DSL** 的工程化 `.skill` 包，把"算子需求 → 可上线九齿 kernel"的全流程封装成可被主流 AI 编程智能体调用的工作流。它提供：

- **9 族算子分类路由**（elementwise / reduction / matmul / attention / conv / generator / layout / scatter / composition）
- **15 种 arrangement 代码模板**（1D tile、2D row tile、3-level matmul、reduce + fp32 accumulator、conv im2col、scatter atomic_add 等）
- **25 张 FIX_CARDS 故障诊断卡**（Triton 编译失败、dtype 属性、精度上溢、非连续 stride、atomic RMW、reward hacking 等）
- **Step 0 → Step 3 的强制工作流**：环境分析 → 能力判定 → 方案简报 → 分类 → 生成 → 测试闭环 → 性能优化 → 审计笔记
- **方案先行确认机制**（`📋 IMPLEMENTATION PROPOSAL`）：任何动笔写代码前必须先向用户披露分类结论、实现方案、是否 view-only、潜在风险与备选路径

---

## 2. 在常见 AI Agent 工具中安装本 Skill

> 把本目录作为只读 Skill 包注册到 Agent，新算子写入用户自己的工程目录。

### 2.1 Qoder（qoderclicn）

```bash
# 方式 A：项目级（推荐，跟随仓库）
cp -r nt-devskill/  .qoder/skills/nt-devskill/

# 方式 B：用户级（全局可用）
cp -r nt-devskill/  ~/.qoder-cn/skills/nt-devskill/

# 触发方式
# 在 Qoder 中直接输入"写一个九齿算子" / "write a ninetoothed kernel"
# 或显式调用：/skill nt-devskill
```

### 2.2 Cursor

```text
1. 在项目根目录新建 .cursor/skills/nt-devskill/，把本目录整个复制进去。
2. 在 .cursorrules 中加入：
     When the user asks for a NineToothed kernel, follow the
     workflow defined in .cursor/skills/nt-devskill/SKILL.md
3. 触发：在 Chat / Composer 中 @nt-devskill 或直接描述算子需求。
```

### 2.3 Claude Code (Anthropic)

```bash
# 项目级 Skill
mkdir -p .claude/skills && cp -r nt-devskill/ .claude/skills/nt-devskill/
# 在 CLAUDE.md 顶部加入：
#   Always consult .claude/skills/nt-devskill/SKILL.md before writing
#   any NineToothed kernel.
```

### 2.4 Windsurf (Cascade)

```text
1. 将本目录复制到 ~/.windsurf/skills/nt-devskill/
2. 在 Windsurf Settings → Cascade → Custom Instructions 中加入：
     Use the nt-devskill skill for any NineToothed / GPU kernel request.
```

### 2.5 Cline / Roo Code

```json
// .cline/skills.json 或 .roo/rules.json
{
  "skills": [
    {
      "name": "nt-devskill",
      "path": "./.cline/skills/nt-devskill/SKILL.md",
      "trigger": ["九齿", "ninetoothed", "kernel", "算子"]
    }
  ]
}
```

### 2.6 OpenAI Codex CLI / GitHub Copilot Workspace

```text
- 把 SKILL.md 加入 AGENTS.md / CODEX.md 顶部 include 列表；
- 在仓库根目录保留本目录（skills/competition/nt-devskill/），让 Codex 自动索引。
```

### 2.7 通用约定

无论哪种 Agent：

| 检查项       | 方法                                                                            |
| ------------ | ------------------------------------------------------------------------------- |
| Skill 可见   | Agent 应能 `cat skills/competition/nt-devskill/SKILL.md` 或列出 `examples/`     |
| 只读约束     | 在 Agent 规则里声明"Skill 目录下禁止写入新算子"                                 |
| MCP 远程 GPU | 若走远程测试，Agent 需具备 `upload_code` / `run_test` / `remote_ls` 等 MCP 工具 |

---

## 3. 适用范围

**适用**：

- 12 个内置示例算子（add / silu / softmax / matmul / bmm / addmm / sdpa / fused_rms_norm / swiglu / conv2d / rope / max_pool2d）
- 9 族算子分类（elementwise、reduction、matmul-like、attention、conv、generator、layout/view、scatter、composition）
- 正确性测试矩阵（shape × dtype × layout，含 MERE/MARE 精度指标）
- 性能基准（E2E + GPU-only + overhead breakdown + Roofline）
- 生成源码审查、arrangement 调试、非连续/stride/offset 敏感任务
- MetaX C500 + CUDA GPU 双平台

**不适用**：

- NineToothed 编译器内核改造
- 隐藏评测答案或针对赛题硬编码的 bypass
- 需要私有凭证 / 在线服务的算子
- 纯 view-only 主操作（如 `pixel_unshuffle`、`reshape` 家族）的"九齿包装"——Skill 会在 `Step 0.8` 主动告知 view-only 性质并建议退回 `torch` 原生路径

---

## 4. 快速使用

```text
用户提示词（示例）：
  实现一个九齿算子：nansum(input, dim, keepdim)
  支持 fp32/fp16，NaN 视为 0，参考 torch.nansum

Agent 会自动：
  Step 0   分析工程目录 + 探测 GPU 环境
  Step 0.5 输出 CAPABILITY REPORT（九齿是否能表达）
  Step 0.8 输出 📋 IMPLEMENTATION PROPOSAL（分类/方案/风险/备选）
  Step 1   按 TAXONOMY.md 路由到 reduction 族
  Step 2   生成 arrangement + application + wrapper + test
  Step 2.5 进入 pytest 闭环（失败→FIX_CARDS→修复→重测）
  Step 2.8 Tile sweep + Roofline 优化
  Step 3   审计笔记（写入自测结果报告）
```

---

## 5. 目录结构

```text
nt-devskill/
├── SKILL.md                     # 工作流主文件（914 行）
├── README.md                    # 本文件
├── REFERENCE.md                 # 公开引用 + AI 辅助披露
├── HONOR_CODE.md                # 竞赛合规声明
├── PR_DESCRIPTION_TEMPLATE.md   # 标准提交描述模板
├── examples/                    # 12 个示例算子（只读）
├── scripts/                     # pipeline / validate / benchmark / doctor / diag_*
├── references/
│   ├── API_REFERENCE.md         # API 速查
│   ├── CODE_TEMPLATES.md        # 15 种 arrangement 模板
│   ├── TAXONOMY.md              # 9 族分类路由
│   ├── FIX_CARDS.md             # 25 张故障诊断卡
│   ├── LAYOUT.md                # 非连续/stride/offset 参考
│   └── OPTIMIZATION_GUIDE.md    # Roofline + tile sweep
├── specs/                       # 12 个算子 YAML 规格卡
├── tests/                       # pytest 正确性 + 性能基准
└── agents/                      # Explore / Plan 子代理配置
```

---

## 6. 验证命令

```shell
# 环境体检
python scripts/doctor.py

# 全流水线（生成 → 测试 → 基准）
python scripts/pipeline.py run --op add

# 正确性
python scripts/validate.py --op softmax

# 性能
python scripts/benchmark.py --op matmul

# 性能诊断
python scripts/diag_overhead.py --op add
python scripts/diag_tile_sweep.py --op add

# arrangement 调试
python scripts/debug_arrangement.py examples.add.kernel:arrangement

# 生成源码审查
python scripts/inspect_generated.py --op add --verbose
```

---

## 7. 关键特性一览

| #   | 特性                         | 说明                                                |
| --- | ---------------------------- | --------------------------------------------------- |
| 1   | 强制迭代测试闭环（Step 2.5） | Write → Test → Fix 循环，禁止跳过                   |
| 2   | 4 轮性能优化（Step 2.8）     | Diagnose → Sweep → Optimize → Verify                |
| 3   | 25 张 FIX_CARDS              | 症状优先的诊断流程图                                |
| 4   | 15 种 arrangement 模板       | 覆盖全部 9 族算子                                   |
| 5   | 远程部署安全                 | MCP/SSH 上传保护 `__init__.py`                      |
| 6   | Reward hacking 防护          | 全零/无操作/常数输出检测                            |
| 7   | MERE/MARE 精度矩阵           | shape × dtype × layout                              |
| 8   | Stop rules                   | 最多 3 轮修复、10 次迭代                            |
| 9   | 方案先行确认（Step 0.8）     | 任何写代码前先输出 `📋 IMPLEMENTATION PROPOSAL`      |
| 10  | 能力可行性判定（Step 0.5）   | 九齿不能表达时输出 `CAPABILITY REPORT` + A/B/C 方案 |

---

## 8. 合规与披露

- ✅ `HONOR_CODE.md` 已签署
- ✅ `REFERENCE.md` 已披露公开来源、AI 辅助、第三方代码
- ✅ 不含密钥、凭证、隐藏答案、针对性 bypass
- ✅ 不修改测试、不放宽 tolerance
- ✅ 所有 benchmark 与正确性结果均可从命令复现

> 详细竞赛提交材料见同目录 `PR_DESCRIPTION_TEMPLATE.md` 与最终报告 `REPORT_nt-devskill.md`。
