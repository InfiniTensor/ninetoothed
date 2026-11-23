# 补丁摘要

新增文件：

- `src/ntops/kernels/maximum.py`
- `src/ntops/torch/maximum.py`
- `tests/test_maximum.py`

修改文件：

- `src/ntops/kernels/__init__.py`
- `src/ntops/torch/__init__.py`

Kernel 复用公共 `element_wise.arrangement`，application 使用 `ntl.maximum(input, other)`。Wrapper 使用 `_cached_make`，缓存键包含 `input.ndim` 和 `other.ndim`。

0 维 `other` 首次测试时被生成代码当作指针传入 `tl.maximum`，导致 Triton 类型错误。最小修复是在 wrapper 中对 0 维输入使用 `other.item()`，把标量值传给 kernel。没有修改公共 arrangement 或编译器。

该修复保证 correctness，但 `.item()` 会触发主机同步，因此 0 维路径被列为性能限制。
