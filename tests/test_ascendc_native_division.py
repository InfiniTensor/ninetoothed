"""Explicit Div uses physical lifetimes and keeps the default strict renderer."""

import pytest

import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed.backends import ascendc_resources as resources
from ninetoothed.compiler import DEFAULT_COMPILER, CompileRequest
from ninetoothed.ir import ir_to_dict

PASS = "ssa.ascendc.optimize_schedule"
MARKER = "Native Div with original IEEE sparse repair"


def _flat(x, y, out):
    return x.tile((128,)), y.tile((128,)), out.tile((128,))


def _rows(x, y, out):
    return x.tile((1, 2048)), y.tile((1, 2048)), out.tile((1, 2048))


def _silu(x, y, out):
    out = x / (1.0 + ntl.exp(-x))  # noqa: F841


def _emit(
    application=_silu, *, options=None, row=False, pass_options=None, pipeline=None
):
    shape = (17, 2048) if row else (32768,)

    return DEFAULT_COMPILER.compile(
        CompileRequest(
            arrangement=_rows if row else _flat,
            application=application,
            tensors=tuple(
                nt.Tensor(shape=shape, dtype="float32", name=n, other=0)
                for n in ("x", "y", "out")
            ),
            backend="ascendc",
            backend_options={
                "arch": "dav-c220",
                **({} if row else {"vector_chunk": 9600}),
            },
            kernel_name="native_div_contract",
            max_num_configs=1,
            pass_options={PASS: options or {}}
            if pass_options is None
            else pass_options,
            pipeline=pipeline,
        )
    )


@pytest.fixture
def platform(monkeypatch):
    # Source-only fixture. Actual SDK bytes and device execution are separate gates.
    monkeypatch.setattr(
        resources,
        "native_division_platform_evidence",
        lambda: {"codegen_arch": "dav-c220", "test_only": True},
    )


@pytest.mark.parametrize("row", (False, True))
def test_registered_candidate_is_really_consumed_by_a_proved_plan(platform, row):
    if not row:
        with pytest.raises(ValueError, match="Removed AscendC policy switches"):
            _emit(options={"strict_validity_summary": False})

    original = _emit(row=row)
    candidate = _emit(options={"candidate": "vector-native-division"}, row=row)
    assert MARKER in candidate.artifact.primary_source
    metadata = candidate.artifact.metadata["ascendc_division_strategy"]
    assert metadata["consumed"]
    assert metadata["extra_ub_bytes"] == 0
    assert metadata["route"] == ("typed-row-dag" if row else "basic-vector")
    assert all(p["operands_preserved"] for p in metadata["operand_lifetimes"])
    assert (
        candidate.artifact.metadata["ssa_metadata"]["selected_schedule_candidate"]
        == "vector-native-division"
    )
    assert ir_to_dict(original.launch_plan) == ir_to_dict(candidate.launch_plan)
    assert ir_to_dict(original.launch_abi) == ir_to_dict(candidate.launch_abi)
    assert (
        "Abs(nt_ref0, nt_dag_v" if row else "Abs(nt_ref0, nt_a0"
    ) in candidate.artifact.primary_source
