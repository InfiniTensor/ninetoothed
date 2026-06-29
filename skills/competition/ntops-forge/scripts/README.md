# ntops-forge 可执行脚本

脚本位于仓库根目录 `scripts/`（赛题允许 skill 包引用同级脚本，见 `REFERENCE.md`）。

| 脚本 | 用途 |
|------|------|
| `forge.py` | 五段流水线 + gate |
| `preflight.py` | 结构护栏 |
| `compare_ref.py` | 语义对照 |
| `scaffold_kernel.py` | 内核骨架 |
| `run_ab_suite.py` | A/B 一键 |
| `bench_op.py` | GPU benchmark |
| `doctor.py` | 环境自检 |

安装后工作目录：`/root/work/skill`（GPU）或本仓库根目录（本地）。
