import ast
import inspect
from pathlib import Path

import pytest
import torch

import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Tensor
from ninetoothed.generation import CodeGenerator
from ninetoothed.language import libdevice
from tests.utils import get_available_devices


def _qualified_isnan(x, output):
    output = ntl.libdevice.isnan(x) != 0  # noqa: F841


def _imported_isnan(x, output):
    output = libdevice.isnan(x) != 0  # noqa: F841


def _qualified_isinf(x, output):
    output = ntl.libdevice.isinf(x) != 0  # noqa: F841


def _arrangement(x, output):
    return x.tile((128,)), output.tile((128,))


@pytest.mark.parametrize(
    "application", (_qualified_isnan, _imported_isnan, _qualified_isinf)
)
def test_libdevice_namespace_is_imported_for_each_backend(
    application, tmp_path, monkeypatch
):
    if application is not _imported_isnan:
        monkeypatch.delitem(application.__globals__, "libdevice")

    arranged = _arrangement(Tensor(1), Tensor(1))
    application.__annotations__ = dict(
        zip(inspect.signature(application).parameters, arranged)
    )
    monkeypatch.setattr("ninetoothed.generation.CACHE_DIR", tmp_path)
    source = CodeGenerator()(
        application,
        caller="torch",
        kernel_name="floating_classification",
        num_warps=4,
        num_stages=1,
        max_num_configs=None,
        prettify=False,
    )
    tree = ast.parse(Path(source).read_text())
    imports = {
        node.module: alias.asname
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
        if alias.name == "libdevice"
    }

    assert set(imports) == {"triton.language.extra", "triton.language.extra.cann"}
    assert len(set(imports.values())) == 1

    classification_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"isnan", "isinf"}
    ]

    assert len(classification_calls) == 2
    assert all(
        isinstance(node.func.value, ast.Name) and node.func.value.id in imports.values()
        for node in classification_calls
    )
    compile(tree, "<floating-classification>", "exec")


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("strided", (False, True))
@pytest.mark.parametrize(
    "application, reference",
    ((_qualified_isnan, torch.isnan), (_qualified_isinf, torch.isinf)),
)
def test_floating_classification_preserves_nan_and_infinity(
    application, reference, strided, dtype, device
):
    values = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), 0.0, -0.0, 1.0, -1.0] * 19,
        dtype=dtype,
        device=device,
    )

    if strided:
        storage = torch.empty((values.numel(), 2), dtype=dtype, device=device)
        storage[:, 0] = values
        values = storage[:, 0]

    output = torch.empty(values.shape, dtype=torch.bool, device=device)
    kernel = ninetoothed.make(
        _arrangement,
        application,
        (Tensor(1), Tensor(1)),
        num_warps=4,
        num_stages=1,
    )
    kernel(values, output)

    assert torch.equal(output, reference(values))
