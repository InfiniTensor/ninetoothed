from pathlib import Path

_EMITTERS = Path(__file__).parents[1] / "src" / "ninetoothed" / "backends" / "emitters"


def test_shared_ssa_emitter_uses_capabilities_instead_of_backend_flags():
    source = (_EMITTERS / "ssa.py").read_text(encoding="utf-8")

    for marker in (
        "is_cuda",
        "is_triton",
        "is_tilelang",
        "Target.CUDA",
        "Target.TRITON",
        "Target.TILELANG",
    ):
        assert marker not in source


def test_backend_strategies_only_use_public_shared_emitter_hooks():
    for name in ("cuda.py", "triton.py", "tilelang.py"):
        source = (_EMITTERS / name).read_text(encoding="utf-8")
        assert "common._" not in source
