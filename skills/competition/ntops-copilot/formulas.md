# 九齿算子公式速查（ntops application 用）

在 `application()` 里改计算逻辑时，优先用 `ntl`（`import ninetoothed.language as ntl`）。

## 一元算子（unary）

| 算子 | application 参考 |
|------|------------------|
| silu | `output = input / (1 + ntl.exp(-ntl.cast(input, ntl.float32)))` |
| relu | `output = max(0.0, input)` |
| gelu (default) | `output = input * 0.5 * (1 + ntl.erf(input / ntl.sqrt(2.0)))` |
| sigmoid | `output = 1 / (1 + ntl.exp(-ntl.cast(input, ntl.float32)))` |
| exp | `output = ntl.exp(input)` |
| neg | `output = -input` |
| abs | `output = ntl.abs(input)` |

## 二元算子（binary，含 alpha）

| 算子 | application 参考 |
|------|------------------|
| add | `output = input + alpha * other` |
| sub | `output = input - alpha * other` |
| mul | `output = input * other` |
| div | `output = input / other` |

## 复杂算子

不要从公式硬写，先读 ntops 或 ninetoothed-examples 同名实现：

- `rms_norm` / `layer_norm`
- `mm` / `bmm` / `conv2d`
- `scaled_dot_product_attention`

## 常见坑

- 输出赋值后 linter 报未使用：行尾加 `# noqa: F841`
- float16/bfloat16：涉及 `exp/sigmoid` 时参考 silu，先 `ntl.cast(..., ntl.float32)`
- 大赛调试慢：先用 `constexpr`/固定 block_size，别用 `meta=True` 自动调优
