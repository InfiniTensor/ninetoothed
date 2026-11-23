#!/usr/bin/env python3
"""环境检测脚本 — 检查 MetaX GPU 服务器上的依赖就绪情况

用法：
    python scripts/doctor.py
"""

import sys
from pathlib import Path


def check():
    print("=" * 50)
    print("  nt-devskill 环境检测")
    print("=" * 50)
    ok = True

    # Python
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"  [OK] Python {py_ver}")

    # torch
    try:
        import torch
        print(f"  [OK] PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"  [OK] GPU: {name}")
            props = torch.cuda.get_device_properties(0)
            print(f"       Memory: {props.total_memory / 1024**3:.1f} GB")
        else:
            print("  [FAIL] CUDA/GPU not available")
            ok = False
    except ImportError:
        print("  [FAIL] PyTorch not installed")
        ok = False

    # triton
    try:
        import triton
        print(f"  [OK] Triton {triton.__version__}")
    except ImportError:
        print("  [WARN] Triton not directly installed (may be bundled)")

    # ninetoothed
    try:
        import ninetoothed
        ver = getattr(ninetoothed, "__version__", "unknown")
        print(f"  [OK] ninetoothed {ver}")
    except ImportError:
        print("  [FAIL] ninetoothed not installed")
        print("         pip install ninetoothed  (或从官方仓库安装)")
        ok = False

    # pytest
    try:
        import pytest
        print(f"  [OK] pytest {pytest.__version__}")
    except ImportError:
        print("  [WARN] pytest not installed (pip install pytest)")

    # examples 导入测试
    SKILL_ROOT = Path(__file__).resolve().parent.parent
    if str(SKILL_ROOT) not in sys.path:
        sys.path.insert(0, str(SKILL_ROOT))

    ops = ["add", "softmax", "matmul", "fused_rms_norm", "silu", "bmm", "addmm", "scaled_dot_product_attention", "swiglu", "conv2d", "rotary_position_embedding", "max_pool2d"]
    imported = 0
    for op in ops:
        try:
            __import__(f"examples.{op}")
            imported += 1
        except Exception as e:
            print(f"  [FAIL] examples.{op}: {e}")
            ok = False

    print(f"\n  算子导入: {imported}/{len(ops)}")

    if ok:
        print("\n  ✓ 环境就绪，可以开始测试。")
    else:
        print("\n  ✗ 环境存在问题，请先修复上述 [FAIL] 项。")

    return ok


if __name__ == "__main__":
    sys.exit(0 if check() else 1)
