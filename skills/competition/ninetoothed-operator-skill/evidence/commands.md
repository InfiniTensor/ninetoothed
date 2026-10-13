# Validation Commands

Environment: Google Colab T4 GPU.

Install dependencies:

```bash
pip install -r requirements.txt
```

Correctness validation:

```bash
python scripts/run_correctness.py 2>&1 | tee evidence/terminal-correctness-raw.txt
```

Benchmark / generated-source dump / performance validation:

```bash
python -m pytest tests/ -m benchmark -v -s --tb=short 2>&1 | tee evidence/terminal-benchmark-raw.txt
```

Raw terminal logs:

- `terminal-correctness-raw.txt`
- `terminal-benchmark-raw.txt`

Refresh the raw logs after changing test semantics.

Useful focused checks:

```bash
python scripts/run_correctness.py --op relu
python scripts/run_correctness.py --op softmax
python scripts/run_correctness.py --op rms_norm
python scripts/run_benchmark.py --op softmax
```
