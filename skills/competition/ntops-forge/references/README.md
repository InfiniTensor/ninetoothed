# ntops-forge 参考资料索引

Agent 开发九齿算子时按需查阅，**不要复制无关长文档**。

| 资料 | 路径 / 链接 | 用途 |
|------|-------------|------|
| 算子分类路由 | `../taxonomy.md` | family → 模板与验收策略 |
| 失败修复卡 | `../fix_cards.md` | 报错 → FC-xxx 动作 |
| 逐元素公式 | `../../ntops-copilot/formulas.md` | unary/binary formula 写法 |
| 官方文档 | https://ninetoothed.org/ | premake / arrangement / application API |
| ntops 仓库 | https://github.com/InfiniTensor/ntops | 参考内核与 pytest |
| 官方示例 | https://github.com/InfiniTensor/ninetoothed-examples | add/silu 结构范式 |

## 复杂算子（决赛扩展）

| family | 必读参考 | 策略 |
|--------|----------|------|
| reduction | `src/ntops/kernels/softmax.py` | 只读 reference，禁止硬写公式 |
| pooling | `src/ntops/kernels/max_pool2d.py` | 关注 stride/padding 布局 |
| norm | `layer_norm.py` / `rms_norm.py` | 分块与归约路径 |

详见 `docs/FinalsRoadmap.md`。
