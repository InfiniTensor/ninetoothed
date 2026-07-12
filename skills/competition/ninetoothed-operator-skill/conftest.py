"""Pytest configuration for the skill's example self-tests.

The example tests reuse the repository's own `tests.utils.get_available_devices`
helper to stay consistent with repo conventions. Because these tests live several
directories below the repository root, that root is not on `sys.path` by default
when the tests are collected directly. This conftest inserts it, so the examples
run with a plain `pytest <path>` from the repository root, without needing
`PYTHONPATH` to be set manually.
"""

import pathlib
import sys

# skills/competition/ninetoothed-operator-skill/conftest.py -> repo root is 3 up.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
