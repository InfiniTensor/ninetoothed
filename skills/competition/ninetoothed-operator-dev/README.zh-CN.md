# ninetoothed-operator-dev

[English](README.md) | **中文**

面向 **NineToothed** DSL 算子开发的可复用 `.skill`,指导 coding agent 完成算子的
编写、验证、优化与调试(arrangement + application),所有 API 均对照真实
`ninetoothed>=0.25.0` 源码核验。

为九齿 `.skill` 创新挑战(T3-1-1)构建,目标 agent 为 **Claude Code** 与
**GPT-5.5 Codex**,采用标准 Anthropic Agent Skills 包结构。

## 包含内容

```
ninetoothed-operator-dev/
  SKILL.md                  # 入口:触发 / 工作流 / 终止 / 产出清单 / 约束
  references/               # 按需加载的家族 playbook
    operator-taxonomy.md    # 编码前先分类
    elementwise.md  reduction.md  layout.md  perf-diag.md  common-errors.md
  scripts/                  # 可运行,无 eval/exec/动态 import/shell 拼接
    gen_pytorch_oracle.py       # 生成 PyTorch 参考 + pytest 脚手架(纯模板)
    run_correctness_matrix.py   # 跑 pytest,汇总 shape x dtype x layout 到 CSV(MERE/MARE)
    inspect_generated_source.py # 读 ~/.ninetoothed/<sha256>.py,报告 tile/算子(只读)
    bench_compare.py            # CUDA Event 计时 + Roofline 分类(可 import)
    debug_arrangement.py        # 编译 kernel 前验证 arrangement
    failure_classifier.py       # 失败归因:code_error / guidance_error / dsl_limit
    aot_build_smoke.sh          # 检查 AOT 构建产出 .py + .h
  examples/                 # 4 个完整自测任务(每类一个)
  tests/
    self_test_tasks.md      # 4 个自测任务(对应隐藏任务四类)
    verifier_spec.md        # 与官方 6 子项对齐的通过/不通过规则
  evaluation/               # 自测材料(不属于可安装的 skill 本体)
    proxy_tasks/            # 24 题离线集(含参考解),用于 A/B 与 Stage 3
    skill_eval/             # 稳健 bench + reward-hacking 防护(仅打分/防作弊)
  REFERENCE.md              # 引用与来源
```

以上 `proxy_tasks`/`skill_eval` 是静态的:一份答案离线集 + 确定性的公平性/防作弊
检查,无外部依赖。它们是选手自己的离线自测材料——`proxy_tasks/` 里的参考解仅用于
给 A/B 打分,**任何时候都不会喂给 agent**,也不是官方隐藏评测的答案。真正驱动
agent 跑 A/B 的自动化(`claude -p`/本地模型/GLM API
的 episode 编排、rubric 打分、自进化 loop)在独立的配套仓库
[`ninetoothed-skill-eval-harness`](https://github.com/noCharger/ninetoothed-skill-eval-harness)
中——它会从本 skill 的实际检出路径读取 `evaluation/proxy_tasks` 与
`evaluation/skill_eval`,而不是自己再拷贝一份。

## 安装 / 激活

**Claude Code** —— 把文件夹拷贝或软链到 skills 目录,自动发现:

```bash
mkdir -p .claude/skills
cp -r ninetoothed-operator-dev .claude/skills/
# 或用户级:~/.claude/skills/
```

**GPT-5.5 Codex** —— 把同一文件夹放到 Codex 加载 skill 的目录(包结构为可移植的
Anthropic Agent Skills 格式)。agent 先读 `SKILL.md`,再按需拉取 `references/*`。

安装过程不执行任何代码,激活仅为文件发现。

## 依赖

| 依赖 | 版本 | 用途 |
|------------|---------|----------|
| ninetoothed | >= 0.25.0 | 被开发的 DSL |
| triton | >= 3.0.0 | NineToothed 后端 |
| torch | >= 2.4.0 | oracle / 正确性 / benchmark 基线 |
| numpy, sympy | 随 ninetoothed | 传递依赖 |
| pytest | >= 7 | 正确性 runner |
| CUDA Toolkit | 12.x | NVIDIA GPU 编译 + 运行 |

```bash
pip install "ninetoothed>=0.25.0" "torch>=2.4.0" "triton>=3.0.0" pytest
```

无 CUDA 环境:`bench_compare.py` 退化为墙钟计时并告警;正确性测试
`skipif(not torch.cuda.is_available())` 自动跳过。

## 快速上手(单算子,端到端)

```bash
# 1. 为你的 wrapper 模块 my_ops 生成 PyTorch oracle + 测试脚手架
python scripts/gen_pytorch_oracle.py --op softmax \
    --wrapper-module my_ops --wrapper-fn softmax \
    --shapes 1024 4095 64,128 --dtypes float16 float32 \
    --out test_softmax_correctness.py

# 2. 跑 shape x dtype x layout 矩阵
python scripts/run_correctness_matrix.py test_softmax_correctness.py --csv matrix.csv

# 3. 查看 NineToothed 生成的源码
python scripts/inspect_generated_source.py

# 4. benchmark + Roofline(在你自己的 bench 文件里 import bench_compare)
```

## 安全 / 合规

无密钥、不联网、无隐藏答案、不删测试。脚本只读,或以 list-form argv 的子进程方式
运行(`shell=False`)。详见 `SKILL.md` 第 5 节与 `REFERENCE.md`。

## 状态

v0(手写 baseline)。下一步是自测任务与 A/B(no-skill vs v0)实测,见
`tests/self_test_tasks.md`。
