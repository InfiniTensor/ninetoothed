# Benchmark Summary

Hardware: Google Colab T4 GPU.

## ReLU

| Input | Latency | Bandwidth |
|---|---:|---:|
| 1M fp16 | 0.029 ms | 145.0 GB/s |
| 16M fp16 | 0.276 ms | 243.4 GB/s |

## RMSNorm Non-Contiguous Fallback

| Path | Latency |
|---|---:|
| Contiguous | 0.041 ms |
| Non-contiguous fallback | 0.067 ms |
| Overhead | 65.6% |

The fallback number includes the explicit `.contiguous()` copy plus the kernel.

## Softmax Throughput

### `tests/test_softmax.py` benchmark

| Shape | NineToothed | PyTorch |
|---|---:|---:|
| (1024, 512) | 136.8 GB/s (0.031 ms) | 13.8 GB/s (0.303 ms) |
| (1024, 2048) | 225.6 GB/s (0.074 ms) | 208.0 GB/s (0.081 ms) |
| (4096, 4096) | 227.8 GB/s (0.589 ms) | 242.3 GB/s (0.554 ms) |

### `tests/test_softmax_perf.py` separated throughput

| Shape | NineToothed | PyTorch |
|---|---:|---:|
| (512, 256) | 17.5 GB/s | 42.3 GB/s |
| (512, 1024) | 150.6 GB/s | 205.1 GB/s |
| (512, 4096) | 227.8 GB/s | 134.2 GB/s |
| (2048, 4096) | 235.5 GB/s | 145.5 GB/s |

## BLOCK_SIZE Padding Waste

| ncols | BLOCK_SIZE | Latency |
|---:|---:|---:|
| 100 | 128 | 0.028 ms |
| 100 | 1024 | 0.026 ms |

Conclusion: for this small shape on T4, over-padding produced a measurable but
small timing difference in the recorded run. Use the smallest power-of-two
BLOCK_SIZE as the documented default and treat exact microbenchmark ratios as
environment-dependent.
