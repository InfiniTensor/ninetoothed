# 15 种常见陷阱及修复方法

| # | 陷阱 | 症状 | 正确做法 |
|---|------|------|----------|
| 1 | 忘记 `# noqa: F841` | lint 警告或编译问题 | output 赋值行必须加 `# noqa: F841` |
| 2 | 累加没用 float32 | float16 精度严重丢失 | `ntl.zeros(..., dtype=ntl.float32)` |
| 3 | squeeze 维度错误 | shape 不匹配、编译失败 | 仔细检查 dtype 链，每层 squeeze 哪个维度 |
| 4 | 缺少 `other` padding | reduction 边界值错误 | reduction 输入需要设 padding fill 值：`other=float("-inf")` 或 `other=0` |
| 5 | `Tensor(0)` 忘记 constexpr | 编译时常量不生效 | 编译时常量必须 `constexpr=True` |
| 6 | premake 没用 `functools.partial` | arrangement 参数未绑定 | 必须 partial arrangement 的配置参数 |
| 7 | 多次 arrangement 没有 `copy.deepcopy` | 同一 tensor 被修改 | 同一 tensor 通过 arrangement 两次必须 deepcopy |
| 8 | torch 层输出 shape 错误 | shape mismatch 错误 | 仔细计算输出 shape 再分配 |
| 9 | float16 直接做 exp/log | NaN/Inf 或精度丢失 | 必须先 `ntl.cast(x, ntl.float32)` |
| 10 | 等式比较直接返回 bool | 类型不匹配 | 须用 `ntl.where(cond, 1, 0)` 转为数值 |
| 11 | application 内引用外部变量（闭包） | `NameError('xxx' is not defined')` | **禁止！** NineToothed 通过源码检查编译 application，闭包变量和外部变量在 triton 编译时不可见。必须通过 `Tensor(0, constexpr=True, value=...)` 传入，或作为 application 函数参数。**注意**：module-level 常量（如 `_MAX_ITER = 64`）同样不可见，需硬编码在函数体内 |
| 12 | kernel 文件缺少 `import ninetoothed` | `NameError('ninetoothed' is not defined)` | 用到 `ninetoothed.float64` 等常量时，必须 `import ninetoothed`（不能只有 `import ninetoothed.language as ntl`） |
| 13 | constexpr Tensor 不传运行时参数 | 参数数量不匹配 | constexpr Tensor 虽然编译时确定值，但 kernel 调用时仍需传入该参数 |
| 14 | 0-dim 标量 Tensor 类型不匹配 | `IncompatibleTypeError: pointer<fp64> and float32` | `Tensor(0, dtype=ninetoothed.float64)` 生成 fp64 指针，与 fp32 输入不兼容。标量系数应改用 constexpr |
| 15 | 固定循环中状态更新未条件化 | 全部输出为 0 或错误常数 | while→range 转换时，**循环体内所有状态变量更新都必须条件化**：`x = ntl.where(converged, x, new_x)`。如果某次迭代算法已收敛，所有赋值必须是 no-op。典型遗漏：只保护了被除数（除零安全），忘了保护被覆盖的变量（如 gcd 中的 `a = t` 应为 `a = ntl.where(t == 0, a, t)`） |

---

## 错误诊断流程

```
遇到错误
├── 编译时错误（Triton 编译失败）
│   ├── NameError → 检查是否引用了闭包变量或缺少 import
│   ├── 类型不匹配 → 检查标量 Tensor 是否用了 fp64 dtype
│   └── shape 不匹配 → 检查 tile/expand/squeeze 链
├── 运行时错误
│   ├── 参数数量不匹配 → 检查 premake Tensor 数量 = kernel 调用参数数量
│   ├── shape mismatch → 检查 torch 层输出分配
│   └── ImportError → 检查 kernel 文件的 import 列表
└── 精度错误
    ├── NaN → 除零或 log 负数，加 epsilon 或 ntl.where 保护
    ├── Inf → exp 溢出（尤其 float16），转 float32 计算
    ├── 全部错误 → arrangement 映射错误，检查 tile/permute 顺序
    └── 边界错误 → 缺少 padding fill，设置正确的 other 参数
```

## 修复优先级

1. **先修编译错误**：确保 kernel 能编译通过
2. **再修运行时错误**：确保 kernel 能运行
3. **最后修精度错误**：确保结果与 CPU 参考一致
4. **性能优化最后做**：精度通过后才考虑性能
