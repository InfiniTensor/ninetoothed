#!/usr/bin/env bash
#
# Set up and verify the NineToothed CPU reference interpreter on a rented GPU box.
#
# Run it from the repository root:
#
#     bash setup_on_server.sh
#
# It installs the project, runs the GPU-free test suite, and reports whether a
# CUDA device is reachable so you know whether `cross_validate.py` can run.
#
# See INTERPRETER_TESTING.md for the full walkthrough.

set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"

echo "== 1. Interpreter =="
"$PYTHON" -V

echo
echo "== 2. Environment probe =="
"$PYTHON" - <<'PROBE'
import importlib.util

for name in ("torch", "triton", "numpy"):
    spec = importlib.util.find_spec(name)
    print(f"  {name:8s} {'installed' if spec else 'MISSING'}")

try:
    import torch
except ModuleNotFoundError:
    print("  cuda     no torch, so no CUDA device can be seen")
else:
    if torch.cuda.is_available():
        print(f"  cuda     {torch.cuda.get_device_name(0)}")
        capability = torch.cuda.get_device_capability(0)
        print(f"  arch     sm_{capability[0]}{capability[1]}")
    else:
        print("  cuda     torch is installed but no device is visible")
PROBE

echo
echo "== 3. Install the project =="
"$PYTHON" -m pip install -e . --quiet

echo
echo "== 4. GPU-free test suite =="
if ! "$PYTHON" -m pytest --version >/dev/null 2>&1; then
    echo "  pytest is not installed yet, adding it"
    "$PYTHON" -m pip install pytest --quiet
fi

"$PYTHON" -m pytest -q tests/test_interpret.py

echo
echo "== 5. Which passes and backends this checkout supports =="
PYTHONPATH=src "$PYTHON" -c "
from ninetoothed.interpret import format_support_matrix
print(format_support_matrix())
"

cat <<'NEXT'

== Next steps ==

The GPU-free suite is the part that must stay green. To go further:

  # everything in the repo that does not need torch
  python -m pytest -q --continue-on-collection-errors tests/

  # the interpreter against a real Triton/CUDA kernel (needs a GPU)
  python cross_validate.py

  # the same, against the repository's own test suite (needs a GPU)
  python -m pytest -q tests/test_softmax.py tests/test_matmul.py

NEXT
