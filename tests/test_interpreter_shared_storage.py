"""Preserve actual view aliasing through independent comparison and replay."""

import json

import numpy as np
import pytest

from ninetoothed.interpreter import interpret_program
from ninetoothed.interpreter.debugger import (
    _copy_inputs,
    compare_programs,
    export_reproducer,
    load_reproducer,
)
from ninetoothed.ir import ssa


@pytest.mark.parametrize(
    "layout", ("overlap", "reverse", "transpose", "mixed_bytes", "zero_stride")
)
def test_copy_inputs_preserves_alias_bytes_strides_and_write_permissions(layout):
    original = np.arange(16, dtype=np.int32).reshape(4, 4)

    if layout == "overlap":
        first, second = original[:, :3], original[:, 1:]
    elif layout == "reverse":
        first, second = original, original[::-1, ::-1]
    elif layout == "transpose":
        first, second = original, original.T
    elif layout == "mixed_bytes":
        first, second = original, original.view(np.uint8)
    else:
        first, second = original, np.broadcast_to(original[1:2, 1:2], (3, 3))

    second.flags.writeable = False
    inputs = {"first": first, "second": second, "same_object": first}
    before = original.copy()
    copied = _copy_inputs(inputs)
    assert copied["first"] is copied["same_object"]
    assert np.shares_memory(copied["first"], copied["second"])
    assert not np.shares_memory(copied["first"], first)

    for name in ("first", "second"):
        assert copied[name].strides == inputs[name].strides
        assert copied[name].flags.writeable == inputs[name].flags.writeable
        np.testing.assert_array_equal(copied[name], inputs[name])

    copied["first"][...] = 0

    if layout == "overlap":
        np.testing.assert_array_equal(copied["second"][:, :2], 0)
    else:
        np.testing.assert_array_equal(copied["second"], 0)

    np.testing.assert_array_equal(original, before)


def _program():
    type_ = ssa.Type(kind="tensor", dtype="int32", shape=("3",))
    first, reader = (ssa.Value(name=name, type=type_) for name in ("first", "reader"))
    seven = ssa.Value(name="%seven", type=ssa.Type(kind="scalar", dtype="int32"))
    result = ssa.Value(name="%result", type=type_)

    return ssa.Program(
        kind="overlap_replay",
        inputs=(first, reader),
        outputs=(result,),
        blocks=(
            ssa.Block(
                operations=(
                    ssa.Operation(
                        opcode="arith.constant", results=(seven,), attrs={"value": 7}
                    ),
                    ssa.Operation(
                        opcode="mem.store", operands=(seven.name, first.name)
                    ),
                    ssa.Operation(
                        opcode="mem.load", operands=(reader.name,), results=(result,)
                    ),
                )
            ),
        ),
    )


def test_overlapping_views_roundtrip_and_execute_with_original_aliases(tmp_path):
    storage = np.arange(5, dtype=np.int32)
    inputs = {"first": storage[:3], "reader": storage[1:4]}
    report = compare_programs(_program(), _program(), inputs)
    assert report.equal
    np.testing.assert_array_equal(storage, np.arange(5, dtype=np.int32))
    directory = export_reproducer(tmp_path / "views", _program(), inputs)
    metadata = json.loads((directory / "manifest.json").read_text())
    assert metadata["schema"] == 2
    program, restored, options = load_reproducer(directory)
    assert np.shares_memory(restored["first"], restored["reader"])
    result = interpret_program(program, restored, **options)
    np.testing.assert_array_equal(result.outputs["%result"], [7, 7, 3])
    np.testing.assert_array_equal(storage, np.arange(5, dtype=np.int32))


@pytest.mark.parametrize(
    "field,value",
    (
        ("offset", -1),
        ("offset", 99999),
        ("strides", [99999]),
        ("shape", [100000]),
        ("dtype", "object"),
    ),
)
def test_view_manifest_cannot_address_outside_numeric_storage(tmp_path, field, value):
    storage = np.arange(5, dtype=np.int32)
    directory = export_reproducer(
        tmp_path / "invalid", _program(), {"first": storage[:3], "reader": storage[1:4]}
    )
    path = directory / "manifest.json"
    metadata = json.loads(path.read_text())
    metadata["inputs"]["reader"][field] = value
    path.write_text(json.dumps(metadata))

    with pytest.raises((ValueError, TypeError)):
        load_reproducer(directory)


def test_padding_not_named_by_any_input_is_zero_in_the_export(tmp_path):
    allocation = np.arange(10, dtype=np.int32) + 913
    first, second = allocation[::3], allocation[3::3]
    inputs = {"first": first, "second": second}
    values = tuple(
        ssa.Value(
            name=name,
            type=ssa.Type(kind="tensor", dtype="int32", shape=(str(array.size),)),
        )
        for name, array in inputs.items()
    )
    program = ssa.Program(
        kind="strided_archive", inputs=values, outputs=values, blocks=(ssa.Block(),)
    )
    path = export_reproducer(tmp_path / "padding", program, inputs)

    with np.load(path / "inputs.npz", allow_pickle=False) as archive:
        payload = archive["storage_0"]

    visible = np.zeros(40, dtype=bool)

    for index in (0, 3, 6, 9):
        visible[index * 4 : index * 4 + 4] = True

    assert np.all(payload[~visible] == 0)


def test_roundtrip_preserves_negative_strides_readonly_scalar_views_and_alias_identity(
    tmp_path,
):
    allocation = np.arange(5, dtype=np.int32)
    first = allocation[:3]
    reader = allocation[2::-1]
    reader.flags.writeable = False
    inputs = {
        "first": first,
        "reader": reader,
        "same": first,
        "scalar_view": allocation[1:2].reshape(()),
        "factor": np.float32(0.5),
    }
    values = tuple(
        ssa.Value(
            name=name,
            type=ssa.Type(
                kind="tensor" if np.ndim(value) else "scalar",
                dtype=str(np.asarray(value).dtype),
                shape=tuple(map(str, np.shape(value))),
            ),
        )
        for name, value in inputs.items()
    )
    program = ssa.Program(
        kind="mixed_archive", inputs=values, outputs=values, blocks=(ssa.Block(),)
    )
    path = export_reproducer(tmp_path / "mixed", program, inputs)
    _, restored, _ = load_reproducer(path)
    assert restored["same"] is restored["first"]
    assert restored["reader"].strides == (-4,)
    assert not restored["reader"].flags.writeable
    assert isinstance(restored["scalar_view"], np.ndarray)
    assert isinstance(restored["factor"], np.float32)
    restored["first"][1] = 17
    assert restored["scalar_view"] == 17
    np.testing.assert_array_equal(restored["reader"], [2, 17, 0])
    np.testing.assert_array_equal(allocation, np.arange(5, dtype=np.int32))


def test_shared_byte_copy_keeps_non_native_endian_nan_payloads():
    bits = np.array([0x7FA00001, 0x7FC00002, 0x80000000], dtype=">u4")
    copied = _copy_inputs({"floating": bits.view(">f4"), "bits": bits})
    assert copied["floating"].dtype == np.dtype(">f4")
    np.testing.assert_array_equal(copied["bits"], bits)
    assert copied["bits"].tobytes() == bits.tobytes()


def test_shared_storage_failure_replays_its_actual_writer(tmp_path):
    from dataclasses import replace

    from ninetoothed.interpreter.failure import replay_failure
    from ninetoothed.ir.provenance import ProvenancePass, seed_origins

    base = _program()
    nine = ssa.Value(name="%nine", type=ssa.Type(kind="scalar", dtype="int32"))
    operations = list(base.blocks[0].operations)
    operations.insert(
        1, ssa.Operation(opcode="arith.constant", results=(nine,), attrs={"value": 9})
    )
    original = seed_origins(
        replace(base, blocks=(ssa.Block(operations=tuple(operations)),))
    )
    tracker = ProvenancePass(original, "bad_alias_store")
    operations = list(original.blocks[0].operations)
    old = operations[2]
    (operations[2],) = tracker.derive(
        (replace(old, operands=(nine.name, "first")),), (old,)
    )
    candidate = tracker.finish(
        replace(original, blocks=(ssa.Block(operations=tuple(operations)),))
    )
    allocation = np.arange(5, dtype=np.int32)
    report = compare_programs(
        original,
        candidate,
        {"first": allocation[:3], "reader": allocation[1:4]},
        failure_dir=tmp_path / "failure",
    )
    assert not report.equal
    assert report.localization.operation.opcode == "mem.load"
    (edge,) = report.dependency_slice.memory_dependencies
    assert edge.byte_ranges == ((4, 12),)
    assert edge.writer_index == 2
    replay_failure(report.reproducer)
    np.testing.assert_array_equal(allocation, np.arange(5, dtype=np.int32))
