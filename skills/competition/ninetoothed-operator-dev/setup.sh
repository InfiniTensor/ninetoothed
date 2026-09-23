#!/usr/bin/env bash
# setup.sh — install dependencies for ninetoothed-operator-dev self-tests.
#
# Usage:
#   bash setup.sh               # core deps only (torch/triton/ninetoothed/pytest)
#   bash setup.sh --venv        # create .venv first, then install core
#   bash setup.sh --with-viz    # core + optional visualization (matplotlib)
#   bash setup.sh --venv --with-viz   # venv + core + viz
#
# Requires Python >= 3.10 and a CUDA-capable GPU (CUDA 12.x recommended).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

USE_VENV=false
WITH_VIZ=false
for arg in "$@"; do
  [[ "$arg" == "--venv"     ]] && USE_VENV=true
  [[ "$arg" == "--with-viz" ]] && WITH_VIZ=true
done

if $USE_VENV; then
  echo "[setup] Creating virtual environment at $SCRIPT_DIR/.venv …"
  python3 -m venv "$SCRIPT_DIR/.venv"
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.venv/bin/activate"
  echo "[setup] Activated .venv"
fi

PYTHON="${PYTHON:-python3}"

echo "[setup] Python: $($PYTHON --version)"
echo "[setup] Installing core dependencies …"

$PYTHON -m pip install --upgrade pip --quiet

# Pin ranges from pyproject.toml of ninetoothed 0.25.0
$PYTHON -m pip install \
  "torch>=2.4.0" \
  "triton>=3.0.0" \
  "ninetoothed>=0.25.0" \
  "sympy>=1.13.0" \
  "numpy>=1.26.4" \
  "pytest>=7" \
  --quiet

echo "[setup] Checking CUDA availability …"
$PYTHON - <<'PYEOF'
import sys
try:
    import torch
    if torch.cuda.is_available():
        print(f"[setup] CUDA OK  — {torch.cuda.get_device_name(0)}")
    else:
        print("[setup] WARNING: CUDA not available. Tests will be skipped.")
        sys.exit(0)
    import ninetoothed
    print(f"[setup] ninetoothed {ninetoothed.__version__ if hasattr(ninetoothed,'__version__') else 'installed'}")
except ImportError as e:
    print(f"[setup] Import error: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

if $WITH_VIZ; then
  echo "[setup] Installing optional visualization deps (matplotlib) …"
  $PYTHON -m pip install -r "$SCRIPT_DIR/requirements-optional.txt" --quiet
  echo "[setup] Visualization deps installed."
  echo "[setup] Note: visualization.visualize(save_path=) is headless-safe."
  echo "[setup]       visualization.visualize_arrangement() requires a display (local dev only)."
fi

echo "[setup] Done. Run:  bash run_self_tests.sh"
