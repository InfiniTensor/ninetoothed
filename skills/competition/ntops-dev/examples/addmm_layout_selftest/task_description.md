# 任务说明：AddMM 布局检查

检查 `ntops.torch.addmm(input, mat1, mat2, beta=1, alpha=1, out=None)`：

- 与 `torch.addmm` 对齐；
- 覆盖 float16/float32；
- 增加 transpose 构造的 non-contiguous 输入；
- 运行 benchmark 和 generated-source 检查。

相关文件：

- `src/ntops/kernels/addmm.py`
- `src/ntops/torch/addmm.py`
- `tests/test_addmm.py`
- `tests/test_mm.py`

```bash
cd ntops
python3 -m pytest tests/test_addmm.py -q
```
