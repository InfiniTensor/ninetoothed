# 算子分类路由（Operator Taxonomy）

Forge 根据 `family` 选择模板与验收策略，Agent **不要猜**。

## elementwise_unary

| op | formula 要点 |
|----|-------------|
| silu | `exp` 前 `ntl.cast(..., float32)` |
| relu | `max(0.0, input)` |
| gelu | `ntl.erf` + `ntl.sqrt(2.0)` |
| sigmoid/exp/neg/abs | 见 `ntops-copilot/formulas.md` |

- **pattern**: `unary`
- **模板**: `premake` + `element_wise.arrangement`
- **pytest**: `tests/test_<op>.py`

## elementwise_binary

| op | formula 要点 |
|----|-------------|
| add/sub | `alpha * other` |
| mul/div | 直接二元组合 |

- **pattern**: `binary`
- **模板**: 四 tensor premake（含 `alpha`）
- **pytest**: `tests/test_<op>.py`

## reduction / pooling / norm / attention（只读参考）

| family | 策略 |
|--------|------|
| reduction | 先读 ntops 同名 kernel（如 softmax），禁止从公式硬写 |
| pooling | 先读 max_pool2d 等，关注 stride/padding 布局 |
| norm | 读 `layer_norm` / `rms_norm` 参考实现 |
| attention | 读 `scaled_dot_product_attention` |

Forge 对 complex family 会 **停止 CODEGEN**，输出「先读 reference」计划。

## 路由规则

```
family → pattern → scaffold style (ntops) → pytest file → guard level (strict)
```
