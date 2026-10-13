# 任务说明：新增 maximum 算子

在 `ntops` 中新增：

```python
ntops.torch.maximum(input, other, *, out=None)
```

要求：

- 使用 NineToothed DSL 编写 kernel；
- 增加 torch wrapper、kernel/torch 导出和 pytest；
- 与 `torch.maximum` 对齐；
- 覆盖 float16、float32、1D 至 4D 非规则 shape；
- 覆盖同 shape 双输入和 0 维 tensor `other`；
- 先跑聚焦测试，再跑相邻逐元素算子回归；
- 记录失败、修复和 benchmark，不修改 NineToothed 编译器核心。

当前不要求支持任意不同 shape 的 PyTorch 广播。公共 `element_wise.arrangement` 只明确处理同维 tensor 和 0 维 tensor，该限制必须写入结果。
