# 任务说明：Softmax 归约

检查 `ntops.torch.softmax(input, dim, dtype=None)`：

- 与 `torch.nn.functional.softmax` 对齐；
- 覆盖 float16/float32 和多维 shape；
- 检查单元素归约维、非末维归约和大值输入；
- 运行 benchmark 和 generated-source 检查。

相关文件：

- `src/ntops/kernels/softmax.py`
- `src/ntops/torch/softmax.py`
- `tests/test_softmax.py`

```bash
cd ntops
python3 -m pytest tests/test_softmax.py -q
```
