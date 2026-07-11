# 05 — Layout / stride / offset patterns

## When to use

non-contiguous inputs, transpose views, strided slice, `storage_offset`, manual load from pointer.

## Reference files

| Topic | Path |
|-------|------|
| stride + offsets + `ntl.load` | `tests/test_clone.py` (`application_1`–`_3`) |
| `data_ptr` + `atomic_add` | `tests/test_data_ptr.py` |
| getitem / views | `tests/test_getitem.py` |
| pad | `tests/test_pad.py` |

## Key APIs (verify in repo before use)

From `test_clone.py`:

```python
input.source.data_ptr()
input.offsets(0)[:, None] * input.source.stride(0)
ntl.load(ptr + ...)
input.source[i, j]  # arranged indexing
```

## Test strategy

1. Build torch tensor, then `transpose` / `slice` for non-contiguous  
2. Compare to torch op on **same view**  
3. Run `test_clone.py` / `test_data_ptr.py` as templates  

```bash
pytest tests/test_clone.py -v
pytest tests/test_data_ptr.py -v
```

## Common errors

- Assuming contiguous storage when input is a view  
- Wrong stride axis in manual `ntl.load`  
- Comparing against torch ref on contiguous copy instead of view

## Report points

Document: `is_contiguous()`, `stride()`, `storage_offset()` in task card and final report.
