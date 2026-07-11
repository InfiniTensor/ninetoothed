#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${SKILL_ROOT}"

echo "Running NineToothed operator skill self-tests..."
echo "Skill root: ${SKILL_ROOT}"

python -m pytest examples/task-01/test_add.py
python -m pytest examples/task-02/test_softmax.py
python -m pytest examples/task-03/test_transpose_add.py

echo "Self-tests completed."
