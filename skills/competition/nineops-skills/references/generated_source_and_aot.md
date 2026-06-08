# Generated Source 检查与 AOT Build 诊断

## 为什么要检查 generated source？

Ninetoothed 将 DSL 定义编译为 Triton 代码。检查生成的 Triton source 可以：

1. 验证 arrangement 是否正确展开为预期的 tile/block 模式
2. 发现 load/store 模式的低效问题（如非合并访问）
3. 检查 AOT build 是否有 fallback 路径
4. 确认 autotune 参数是否被正确注入

## 查看 generated source

```python
import ninetoothed
from ninetoothed.codegen.source import get_generated_source

# 假设 kernel 已创建
source = get_generated_source(kernel, ...)
print(source)
```

或在命令行：

```bash
# 使用 inspect 脚本
bash .skill/scripts/inspect_generated_source.sh
```

## 常见检查点

### 1. Load 指令

检查点：
- 是否使用 `mask` 处理边界？
- ptr 计算是否按 stride/offset 正确推导？
- 是否有非合并访问（stride > 1 的维度的连续 load）？

✅ 好的 load：

```python
tl.load(input_ptrs + offsets, mask=mask, other=0.0)
```

❌ 有问题的 load：

```python
# 缺少 mask，可能导致越界
tl.load(input_ptrs + offsets)
```

### 2. Store 指令

检查点：
- 是否有 output 的 store 操作？
- mask 是否正确？

### 3. Tile/Block 循环

检查点：
- 外层循环是否遍历了正确的维度？
- 内层 dot 累加是否在 K 维度循环？

### 4. AOT build 日志

```bash
# 查看 AOT compilation 输出
python -c "
import ninetoothed
ninetoothed.make(..., aot=True)
" 2>&1 | tee aot_build.log
```

常见 AOT build 问题：

| 问题 | 表现 | 解决方案 |
|------|------|----------|
| Triton 编译失败 | `CUDA error: invalid configuration` | 调整 BLOCK_SIZE |
| 内存越界 | `tl.load` 访问超出分配范围 | 检查 mask / other 填充值 |
| dtype 不匹配 | load/store 类型不一致 | 显式 cast |
| Autotune 超时 | 搜索空间过大 | 缩小 block_size 范围 |

## 5. Fallback 诊断

当 detected fallback 时：

```bash
# 检查 fallback 触发条件
NINETOOTHED_DEBUG=1 python test_script.py
```

常见 fallback：
- Triton 不支持的操作 → 回退到 PyTorch
- 未安装 CUDA → 回退到 CPU
- 布局无法展开 → 回退到 element-wise 逐元素
