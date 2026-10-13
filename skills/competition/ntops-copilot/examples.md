# 示例：run_task 开发 silu（ntops 真实流程）

## 1. 环境

```bash
source /root/miniconda3/bin/activate base   # GPU 云常见
python scripts/doctor.py
```

## 2. 一键开工

```bash
python scripts/run_task.py --task silu --ntops-root /root/work/ntops --contest-id T1-1-1
```

生成：
- `src/ntops/kernels/silu.py`（premake 骨架）
- `src/ntops/torch/silu.py`

## 3. 改公式

编辑 `application()`（参考 `formulas.md`）：

```python
def application(input, output):
    output = input / (1 + ntl.exp(-ntl.cast(input, ntl.float32)))  # noqa: F841
```

## 4. 注册

在 `kernels/__init__.py` 和 `torch/__init__.py` 加入 `silu`（照抄相邻算子写法）。

## 5. 测试

```bash
python scripts/preflight.py src/ntops/kernels/silu.py --kernel
pytest tests/ -k silu -q
```

GPU 机实测：**8 passed**。

## 6. PR

```bash
git checkout -b 2026-spring-hongwei-2026-T1-1-1
# commit + push + PR 描述用 templates/PR_DESCRIPTION.md
```
