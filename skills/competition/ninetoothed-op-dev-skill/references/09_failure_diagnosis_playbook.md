# 09 — Failure diagnosis playbook

## Loop (mandatory on any fail)

```text
1. Symptom     — command + error / wrong output
2. Reproduce   — minimal shape/dtype/device
3. Hypothesis  — arrangement vs application vs layout vs dtype
4. Minimal fix — smallest patch
5. Re-verify   — same pytest command
6. Log         — save to logs/ and reference in report
```

## Symptom → likely cause

| Symptom | Check first |
|---------|-------------|
| `CUDA error` / no kernel image | PyTorch build vs GPU arch (sm_120 → nightly cu128) |
| `allclose` fail, small drift | fp16 tolerance; softmax stability |
| `allclose` fail, large error | wrong broadcast tile; wrong dim for reduce; wrong op in application (`*` vs `+`) |
| misleading "broadcast" in assert | read kernel `application` — message may be stale comment |
| shape mismatch | arrangement `expand` / `squeeze` on dtype meta |
| pass on contiguous, fail on view | stride/offset — see `05_layout_stride_offset_patterns.md` |
| first run very slow | JIT compile or autotuning — not necessarily bug |
| AOT fail | nvcc missing; read `test_aot.py` for required env |

## Commands for diagnosis

```bash
pytest <file>::<test> -v --tb=long
python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
```

Optional: inspect `kernel._source` after JIT (`08_generated_source_aot_debugging.md`).

## When to declare unsupported

- No CUDA and task requires GPU kernel execution  
- nvcc missing and task requires AOT binary proof  
- Autotuning timeout — report with suggestion to fix block sizes, not silent skip

## Scorecard reminder (0–10)

Fix correctness before claiming performance points. Document diagnosis for partial credit on process/compliance.
