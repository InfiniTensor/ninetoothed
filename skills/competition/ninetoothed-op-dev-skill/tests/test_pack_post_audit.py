"""Post-pack audit: must inspect pack *output*, not only the source workspace."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
PACKER = SCRIPTS / "pack_runtime_skill.py"

sys.path.insert(0, str(SCRIPTS))
from audit_packed_skill import audit_packed_tree  # noqa: E402


def _pack(dst: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(PACKER),
            "--src",
            str(SKILL_ROOT),
            "--dst",
            str(dst),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


@pytest.mark.skipif(not PACKER.is_file(), reason="packer absent in packed install")
def test_pack_then_audit_output(tmp_path: Path):
    """Authoritative source-workspace check: pack → audit the pack output only."""
    dst = tmp_path / "ninetoothed-op-dev-skill-runtime"
    _pack(dst)
    errors = audit_packed_tree(dst)
    assert not errors, "\n".join(errors)


def test_runtime_tree_in_place_audit():
    """Packed install (no packer): audit the installed skill root itself."""
    if PACKER.is_file():
        pytest.skip(
            "source workspace uses test_pack_then_audit_output against pack output"
        )
    errors = audit_packed_tree(SKILL_ROOT)
    assert not errors, "\n".join(errors)
