"""Direct half transfers require full original masks under either accumulation mode."""

from tests.test_cuda_wmma_grouping import _compile


def test_direct_loads_default_to_proved_maps_and_keep_indexed_fallback():
    mode = "ordered"
    options = {"cuda_wmma_accumulation": mode}
    baseline = _compile(backend_options=options).artifact
    selected = baseline
    assert "_dense_wmma" in selected.primary_source
    assert selected.metadata["cuda_wmma_direct_load"]["selected"]
    fast = selected.primary_source.split("_dense_wmma) {", 1)[1].split("} else {", 1)[0]
    assert "__float2half_rn" not in fast
    assert "nt_pred" not in fast
    assert " = x[" in fast and " = y[" in fast
    assert ("__fadd_rn" in selected.primary_source) is (mode == "ordered")
    assert selected.metadata["cuda_wmma_accumulation"]["mode"] == mode
    assert "__float2half_rn" in selected.primary_source
