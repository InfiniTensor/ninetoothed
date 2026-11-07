# 10 — Patch minimality checklist

Before submitting or claiming task complete:

## Diff scope

- [ ] Only files required for the operator/test/benchmark  
- [ ] No reformatting unrelated files  
- [ ] No mass rename / import reorder in `<repo-root>/src/ninetoothed/` unless task demands  
- [ ] No deleted tests or `@pytest.mark.skip` without documented unsupported case  

## git checks (user runs manually)

```bash
git status
git diff --stat
```

Red flags:

- `third_party/` bulk changes  
- `.venv/`, `.pip-cache/` committed  
- `logs/` with fabricated numbers (logs OK if real command output)

## Style alignment

- [ ] Test names match `test_<feature>.py`  
- [ ] Uses `get_available_devices()` and parametrize like neighbors  
- [ ] Kernel uses `# noqa: F841` on output assignment if matching repo style  

## Competition compliance

- [ ] No hidden task names hardcoded  
- [ ] No “comment out assertion” patterns  
- [ ] README/report lists exact commands run
