import ast
import importlib
import sys
import types
from pathlib import Path


def _copy_application(input, output):
    output = input  # noqa: F841


def _scalar_broadcast_application(input, output):
    output = input  # noqa: F841


def _import_with_triton_stub(monkeypatch, module_name):
    package_root = Path(__file__).resolve().parents[1] / "src" / "ninetoothed"

    for name in tuple(sys.modules):
        if name == "ninetoothed" or name.startswith("ninetoothed."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    package = types.ModuleType("ninetoothed")
    package.__path__ = [str(package_root)]
    monkeypatch.setitem(sys.modules, "ninetoothed", package)

    triton = types.ModuleType("triton")
    triton_language = types.ModuleType("triton.language")
    triton_language_extra = types.ModuleType("triton.language.extra")
    triton_language_extra.libdevice = types.SimpleNamespace()
    triton_language.extra = triton_language_extra
    triton.language = triton_language
    triton.runtime = types.SimpleNamespace(
        JITFunction=type("JITFunction", (), {}),
        driver=types.SimpleNamespace(
            active=types.SimpleNamespace(
                get_current_device=lambda: 0,
                utils=types.SimpleNamespace(get_device_properties=lambda device: {}),
            )
        ),
    )

    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.language", triton_language)
    monkeypatch.setitem(sys.modules, "triton.language.extra", triton_language_extra)

    return importlib.import_module(module_name)


def test_literal_true_mask_is_omitted_from_memory_calls(monkeypatch):
    generation = _import_with_triton_stub(monkeypatch, "ninetoothed.generation")
    symbol = importlib.import_module("ninetoothed.symbol")

    node = generation.CodeGenerator._generate_memory_call(
        "load", symbol.Symbol("ptr"), mask=symbol.Symbol(True), other=0
    )

    source = ast.unparse(node)
    assert source == "ninetoothed.language.load(ptr, other=0)"


def test_literal_true_conjunction_is_simplified_from_memory_masks(monkeypatch):
    generation = _import_with_triton_stub(monkeypatch, "ninetoothed.generation")
    symbol = importlib.import_module("ninetoothed.symbol")

    mask = symbol.Symbol(True) & symbol.Symbol("i < n")
    node = generation.CodeGenerator._generate_memory_call(
        "load", symbol.Symbol("ptr"), mask=mask, other=0
    )

    source = ast.unparse(node)
    assert source == "ninetoothed.language.load(ptr, mask=i < n, other=0)"


def test_all_literal_true_conjunction_is_omitted_from_memory_masks(monkeypatch):
    generation = _import_with_triton_stub(monkeypatch, "ninetoothed.generation")
    symbol = importlib.import_module("ninetoothed.symbol")

    mask = symbol.Symbol(True) & symbol.Symbol(True)
    node = generation.CodeGenerator._generate_memory_call(
        "store", symbol.Symbol("ptr"), symbol.Symbol("value"), mask=mask
    )

    source = ast.unparse(node)
    assert source == "ninetoothed.language.store(ptr, value)"


def test_redundant_arange_bounds_are_recognized(monkeypatch):
    tensor = _import_with_triton_stub(monkeypatch, "ninetoothed.tensor")
    symbol = importlib.import_module("ninetoothed.symbol")

    index = symbol.Symbol("ninetoothed.language.arange(0, n)")
    size = symbol.Symbol("n")

    assert tensor._is_known_non_negative_index(index)
    assert tensor._is_strict_upper_bound_check_redundant(index, size)


def test_lower_bound_symbol_indices_are_recognized_as_non_negative(monkeypatch):
    tensor = _import_with_triton_stub(monkeypatch, "ninetoothed.tensor")
    symbol = importlib.import_module("ninetoothed.symbol")

    index = symbol.Symbol("pid_index", lower_bound=0)

    assert tensor._is_known_non_negative_index(index)


def test_pid_grid_upper_bound_mask_is_pruned_but_tail_mask_remains(monkeypatch):
    generation = _import_with_triton_stub(monkeypatch, "ninetoothed.generation")
    tensor = importlib.import_module("ninetoothed.tensor")
    symbol = importlib.import_module("ninetoothed.symbol")

    block_size = symbol.Symbol("block_size", meta=True)
    tiled = tensor.Tensor(1).tile((block_size,))

    generator = generation.CodeGenerator()
    generator._invariants = {}
    generator._args = [tiled]

    source = ast.unparse(generator._generate_load(tiled))

    assert "ninetoothed_tensor_0_index_0 <" not in source
    assert (
        "ninetoothed_tensor_0_index_0 * ninetoothed_meta_prefix_block_size +" in source
    )
    assert " < ninetoothed_ninetoothed_tensor_0_size_0" in source


def test_pid_upper_bound_mask_is_kept_for_non_grid_shapes(monkeypatch):
    generation = _import_with_triton_stub(monkeypatch, "ninetoothed.generation")
    tensor = importlib.import_module("ninetoothed.tensor")
    symbol = importlib.import_module("ninetoothed.symbol")

    block_size = symbol.Symbol("block_size", meta=True)
    grid_tiled = tensor.Tensor(1).tile((block_size,))
    non_grid_tiled = tensor.Tensor(1).tile((block_size,))

    generator = generation.CodeGenerator()
    generator._invariants = {}
    generator._args = [grid_tiled]

    source = ast.unparse(generator._generate_load(non_grid_tiled))

    assert "* ninetoothed_meta_prefix_block_size +" in source
    assert " < ninetoothed_ninetoothed_tensor_" in source


def test_broadcast_expand_uses_zero_offset(monkeypatch):
    tensor = _import_with_triton_stub(monkeypatch, "ninetoothed.tensor")
    symbol = importlib.import_module("ninetoothed.symbol")

    source = tensor.Tensor(shape=(1,))
    expanded = source.expand((4,))

    offset = expanded._offsets((symbol.Symbol("i"),))[0][0]

    assert ast.unparse(symbol.Symbol(offset).node) == "0"


def test_broadcast_expand_preserves_tile_offset_shape(monkeypatch):
    tensor = _import_with_triton_stub(monkeypatch, "ninetoothed.tensor")
    symbol = importlib.import_module("ninetoothed.symbol")

    source = tensor.Tensor(shape=(1,))
    expanded = source.expand((4,))

    index = symbol.Symbol("ninetoothed.language.arange(0, block_size)")
    offset = expanded._offsets((index,))[0][0]

    assert ast.unparse(symbol.Symbol(offset).node) == (
        "ninetoothed.language.arange(0, block_size) * 0"
    )


def test_uniform_scalar_broadcast_load_omits_mask_and_zero_stride(monkeypatch):
    generation = _import_with_triton_stub(monkeypatch, "ninetoothed.generation")
    tensor = importlib.import_module("ninetoothed.tensor")
    symbol = importlib.import_module("ninetoothed.symbol")

    source = tensor.Tensor(shape=(1,))
    expanded_tile = source.expand((symbol.Symbol("n"),)).tile(
        (symbol.Symbol("block_size", meta=True),)
    )

    generator = generation.CodeGenerator()
    generator._invariants = {}

    source = ast.unparse(generator._generate_load(expanded_tile))

    assert source == (
        "ninetoothed.language.load(ninetoothed_tensor_0_pointers, other=None)"
    )


def test_dynamic_scalar_broadcast_load_uses_scalar_nonempty_mask(monkeypatch):
    generation = _import_with_triton_stub(monkeypatch, "ninetoothed.generation")
    tensor = importlib.import_module("ninetoothed.tensor")
    symbol = importlib.import_module("ninetoothed.symbol")

    source = tensor.Tensor(1)
    expanded_tile = source.expand((symbol.Symbol("n"),)).tile(
        (symbol.Symbol("block_size", meta=True),)
    )

    generator = generation.CodeGenerator()
    generator._invariants = {}

    source = ast.unparse(generator._generate_load(expanded_tile))

    assert source == (
        "ninetoothed.language.load(ninetoothed_tensor_0_pointers, "
        "mask=ninetoothed_ninetoothed_tensor_0_size_0 > 0, other=None)"
    )


def test_jit_stride_metadata_is_constexpr_without_launch_arguments(monkeypatch):
    generation = _import_with_triton_stub(monkeypatch, "ninetoothed.generation")
    tensor = importlib.import_module("ninetoothed.tensor")

    _copy_application.__annotations__ = {
        "input": tensor.Tensor(1).tile((128,)),
        "output": tensor.Tensor(1).tile((128,)),
    }

    source_file = generation.CodeGenerator()(
        _copy_application,
        "torch",
        "copy_stride_probe",
        4,
        3,
        None,
        False,
    )
    source = Path(source_file).read_text(encoding="utf-8")

    assert "_s_0: triton.language.constexpr" in source
    assert "_stride_0: triton.language.constexpr" not in source

    launch_line = next(
        line
        for line in source.splitlines()
        if line.startswith("def launch_copy_stride_probe")
    )
    launch_args = launch_line.removeprefix(
        "def launch_copy_stride_probe("
    ).removesuffix("):")

    assert "stride" not in launch_args
    assert "_s_0 = ninetoothed_tensor_" in source
    assert ".stride(0)" in source


def test_uniform_scalar_broadcast_drops_unused_source_stride_constexpr(monkeypatch):
    generation = _import_with_triton_stub(monkeypatch, "ninetoothed.generation")
    tensor = importlib.import_module("ninetoothed.tensor")
    symbol = importlib.import_module("ninetoothed.symbol")

    block_size = symbol.Symbol("block_size", meta=True)
    input_source = tensor.Tensor(1)
    output_source = tensor.Tensor(1)
    input_arg = input_source.expand((output_source.shape[0],)).tile((block_size,))
    output_arg = output_source.tile((block_size,))

    _scalar_broadcast_application.__annotations__ = {
        "input": input_arg,
        "output": output_arg,
    }

    source_file = generation.CodeGenerator()(
        _scalar_broadcast_application,
        "torch",
        "scalar_stride_prune_probe",
        4,
        3,
        None,
        False,
    )
    source = Path(source_file).read_text(encoding="utf-8")

    input_stride = str(input_arg.source.stride_string(0))
    output_stride = str(output_arg.source.stride_string(0))

    assert input_stride not in source
    assert output_stride in source


def test_aot_unparser_handles_excluded_constexpr_stride_arguments(monkeypatch):
    aot = _import_with_triton_stub(monkeypatch, "ninetoothed.aot")

    launch_func = ast.parse(
        """
def launch_probe(input, output):
    probe[input.shape[0]](input, output, input.strides[0])
"""
    ).body[0]

    unparsed = aot._Unparser(
        ("CUstream", "float*", "float*"),
        constexpr_inner_strides=(("input", 0),),
    ).unparse(launch_func)

    assert "return probe[input.shape[0]]((CUstream)stream," in unparsed
    assert "*(float* *)input.data.shape" not in unparsed
    assert "*(float* *)input.data" in unparsed
    assert "input.strides[0]" not in unparsed


def test_aot_unparser_keeps_unspecialized_constexpr_stride_arguments(monkeypatch):
    aot = _import_with_triton_stub(monkeypatch, "ninetoothed.aot")

    launch_func = ast.parse(
        """
def launch_probe(input, output):
    ninetoothed_constexpr_prefix_ninetoothed_ninetoothed_tensor_0_s_0 = input.strides[0]
    probe[input.shape[0]](ninetoothed_constexpr_prefix_ninetoothed_ninetoothed_tensor_0_s_0, input, output)
"""
    ).body[0]

    unparsed = aot._Unparser(
        ("CUstream", "int64_t", "float*", "float*"),
    ).unparse(launch_func)

    assert (
        "(int64_t)ninetoothed_constexpr_prefix_ninetoothed_ninetoothed_tensor_0_s_0"
        in unparsed
    )
    assert (
        "auto ninetoothed_constexpr_prefix_ninetoothed_ninetoothed_tensor_0_s_0 = "
        "input.strides[0]" in unparsed
    )
    assert (
        "*ninetoothed_constexpr_prefix_ninetoothed_ninetoothed_tensor_0_s_0.data"
        not in unparsed
    )


def test_aot_unparser_drops_specialized_constexpr_stride_names(monkeypatch):
    aot = _import_with_triton_stub(monkeypatch, "ninetoothed.aot")

    launch_func = ast.parse(
        """
def launch_probe(input, output):
    ninetoothed_constexpr_prefix_ninetoothed_ninetoothed_tensor_0_s_0 = input.strides[0]
    probe[input.shape[0]](ninetoothed_constexpr_prefix_ninetoothed_ninetoothed_tensor_0_s_0, input, output)
"""
    ).body[0]

    unparsed = aot._Unparser(
        ("CUstream", "float*", "float*"),
        constexpr_inner_strides=(("ninetoothed_tensor_0", 0),),
    ).unparse(launch_func)

    assert (
        "ninetoothed_constexpr_prefix_ninetoothed_ninetoothed_tensor_0_s_0"
        not in unparsed
    )
    assert "*(float* *)input.data" in unparsed
