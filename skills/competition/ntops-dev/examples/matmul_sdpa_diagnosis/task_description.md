# 任务说明：Matmul / SDPA 失败诊断

运行以下测试并保留真实失败：

- `ntops.torch.matmul`
- `ntops.torch.scaled_dot_product_attention`

检查文件：

- `src/ntops/torch/matmul.py`
- `src/ntops/torch/mm.py`
- `src/ntops/torch/bmm.py`
- `tests/test_matmul.py`
- `tests/test_scaled_dot_product_attention.py`

```bash
cd ntops
python3 -m pytest tests/test_matmul.py -q
python3 -m pytest tests/test_scaled_dot_product_attention.py -q
```

出现数值失败时先固定随机种子；出现导入失败时先核对 PyTorch 版本，不通过跳过测试伪造成功。
