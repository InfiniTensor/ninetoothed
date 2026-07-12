# proxy_tasks —— 离线评测任务集

[English](README.md) | **中文**

赛题的 8 个隐藏评测任务不可见。proxy 任务集是一组镜像隐藏分布的替身任务,用于在
缺少隐藏任务的情况下提供可重复的离线评测基准:度量 skill 安装前后的增益(A/B),
并为 Stage 3 的 MOO 迭代提供 reward 信号。

## 规模与划分

24 题,覆盖赛题四类,每类 6 题;每类 4 题为 train、2 题为 holdout(合计 train 16、
holdout 8)。

| 类别 | train | holdout |
|---|---|---|
| 逐元素 / 广播 | add, mul_broadcast, relu, gelu | silu, masked_add |
| 归约 / 分块 | sum_last, mean_last, softmax, rms_norm | max_last, l2_norm |
| 布局敏感 | contig_transpose, flip_last, narrow_half, strided_gather | pixel_unshuffle, pixel_shuffle |
| 性能 / 诊断 | softmax_no_maxsub, mean_no_upcast, add_bench_memorybound, inspect_tile_config | noncontig_regression, aot_numwarps_mismatch |

holdout 集刻意采用 train 集没有的算子(如 pixel_unshuffle / pixel_shuffle),通过
holdout 即说明 skill 的增益可泛化到调参集合之外,而非对公开样例过拟合。

## holdout 设计——每题在测什么

每道 holdout 题都按一个明确的方式设计:如果 skill 只教了可照抄的答案而非可迁移的
规则,这道题就会在特定的轴上失败。"泛化轴"一列即设计意图——从 train 锚点必须
迁移过来的东西。

| holdout | train 锚点 | 泛化轴 |
|---|---|---|
| ew05 `silu` | relu, gelu | 未见过的一元组合(`x·sigmoid(x)`)——迁移的是 ntl 数学函数的逐元素模式,不是背下来的公式 |
| ew06 `masked_add` | add + mul_broadcast | 首次**组合**广播与掩码(`other=` / `ntl.where`)——train 只分别展示过两种机制,从未合并 |
| rd05 `max_last` | sum_last, softmax | 换掉折叠原语及其 padding 单位元(0 → −inf)——测单位元推理,不是 sum 的配方 |
| rd06 `l2_norm` | rms_norm | 同一条 fp32 累积纪律套在不同公式上(`sqrt∘sum∘square`)——测数值规则,不是算子本身 |
| ly05/ly06 `pixel_(un)shuffle` | contig_transpose … strided_gather | rank-2 → rank-4 的窗口化重索引;需要组合 train 从未完整展示过的 tile→ravel→flatten 链 |
| pd05 `noncontig_regression` | pd01/pd02 | 根因横跨 wrapper/kernel 边界(布局假设),而非单点的 kernel 内数值 bug |
| pd06 `aot_numwarps_mismatch` | pd03/pd04 | 诊断对象从 JIT 运行时移到 AOT 构建链(`make(caller=...)` 参数) |

协议:holdout 只在最后打分一次,绝不用于调优 skill 文本(Stage 3 只在 train 上
优化)。因此 holdout 通过即证明 skill 教的是可迁移的规则(家族 playbook + 纪律),
而非答案;train/holdout 的分差即过拟合的证据。

## 两种 kind

- **operator**(18 题):要求实现一个 NineToothed 算子。携带 PyTorch 参考实现与
  输入生成器;正确性按 MERE/MARE 与参考对比(阈值见
  `../../scripts/run_correctness_matrix.py`:fp32 1.22e-4、fp16 9.77e-4、
  bf16 7.81e-3)。
- **diagnosis**(6 题):给定一个出错或偏慢的 kernel,要求定位根因并给出最小修复。
  携带 `scenario` 与 `expected_findings` 清单,按命中的 finding 数量打分,需修复的
  另判正确性。

任务的 `prompt` / `scenario` / `expected_findings` 字符串特意保留中文:这是面向
中文赛题的任务内容,非代码。

## 任务来源

- InfiniTensor/ninetoothed 已合并 PR 与 issue 中出现的算子开发场景
- 仓库 `examples/` 与 `ntops` 已有算子
- 为布局敏感与 scatter 覆盖缺口手工补充的用例(公开工作在这两类较弱)

## 文件结构

```
proxy_tasks/
  schema.py       # TaskSpec 定义与校验;randn_inputs 工厂
  elementwise.py  # 6 题
  reduction.py    # 6 题
  layout.py       # 6 题
  perf_diag.py    # 6 题(diagnosis)
  loader.py       # 加载、校验、生成 manifest、可选 torch CPU 自检
  manifest.json   # 由 loader 从模块生成,保证与代码不漂移
  README.md
```

## 使用

```bash
# 校验 schema 与结构不变量(24 题 / 每类 4+2),重建 manifest
python loader.py

# 额外在 CPU 上跑每个 operator 任务的参考实现做数值自检(需 torch)
python loader.py --check
```

模块在未安装 torch 的机器上也可导入(torch 在 reference / make_inputs 内部惰性
导入),因此结构校验与 manifest 生成不依赖 GPU 环境;数值自检在装有 torch 的机器
上运行。

## 与评测闭环的衔接

evaluator(`../skill_eval/`)消费本任务集:对每题在 no-skill 与 v0 两档下让 agent
求解,operator 任务用 MERE/MARE 判正确性、`robust_bench` 测性能并查 reward-hacking,
diagnosis 任务按 `expected_findings` 命中率打分,失败用 `failure_classifier` 归因。
train 集驱动 Stage 3 优化,holdout 集仅用于最终泛化评估。
