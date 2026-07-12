#!/bin/bash
# ============================================================
# NineToothed Operator Skill — Colab/Cloud GPU Setup Script
# Run this ONCE on a GPU machine (Colab / AutoDL / etc.)
# ============================================================
set -e

echo "========================================"
echo " Step 1/4: Check GPU"
echo "========================================"
nvidia-smi || { echo "NO GPU FOUND. Make sure you selected GPU runtime."; exit 1; }

echo ""
echo "========================================"
echo " Step 2/4: Install dependencies"
echo "========================================"
pip install -q torch ninetoothed pytest triton
echo "Done."

echo ""
echo "========================================"
echo " Step 3/4: Clone NinToothed + install"
echo "========================================"
if [ ! -d "ninetoothed" ]; then
    git clone https://github.com/InfiniTensor/ninetoothed.git
fi
cd ninetoothed
pip install -q -e .
echo "Done."

echo ""
echo "========================================"
echo " Step 4/4: Environment check"
echo "========================================"
python -c "
import torch
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.version.cuda}')
print(f'GPU:      {torch.cuda.get_device_name(0)}')
print(f'GPU RAM:  {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')

try:
    import ninetoothed
    print(f'NineToothed: OK')
except:
    print('NineToothed: MISSING')

try:
    import triton
    print(f'Triton:   {getattr(triton, \"__version__\", \"unknown\")}')
except:
    print('Triton:   MISSING')

try:
    import pytest
    print(f'pytest:      OK')
except:
    print('pytest:      MISSING')
"

echo ""
echo "========================================"
echo " SETUP COMPLETE — ready to run tests."
echo "========================================"
