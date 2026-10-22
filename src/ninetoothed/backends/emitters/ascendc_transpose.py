"""Tile a proved physical transpose through exact DMA and private UB gather."""

from dataclasses import dataclass

from ninetoothed.backends.ascendc_resources import ASCENDC_UB_BYTES
from ninetoothed.backends.emitters.base import private_parameter_conflicts
from ninetoothed.backends.emitters.transpose_transfer import plan_transpose_transfer

_TILE = 64
_ELEMENTS = _TILE * _TILE
_UB_BYTES = 5 * _ELEMENTS * 4


@dataclass(frozen=True)
class TransposeBody:
    body: str
    guard: str
    tasks: int
    ub_bytes: int = _UB_BYTES
    tile_rows: int = _TILE
    tile_columns: int = _TILE


def render_transpose_transfer(context, *, budget=ASCENDC_UB_BYTES):
    """Preserve every original predicate, bit pattern and output owner.

    GM-side DataCopyPad strides are bytes; UB-side strides are 32-byte blocks
    after rounding the short row to a block boundary. Gather offsets address
    the private tile in bytes. Only offset arithmetic uses float: its
    values are small exactly representable integers or powers-of-two scalings.
    Input data never undergoes arithmetic or numeric conversion. Rectangular
    tiles preserve the same complete output ownership and proof domain.
    """
    if private_parameter_conflicts(
        context,
        ("nt_",),
        excluded_prefixes=("nt_materialized_", "nt_dot_partials"),
    ):
        return None

    plan = plan_transpose_transfer(context)

    if plan is None:
        return None

    large_matrix = (
        plan.rows >= 128
        and plan.columns >= 64
        and plan.rows * plan.columns >= 512 * 1024
        and 5 * 128 * 64 * 4 + 256 <= budget
    )
    tile_rows, tile_columns = (128 if large_matrix else 64), 64

    elements = tile_rows * tile_columns
    ub_bytes = 5 * elements * 4

    if (
        ub_bytes + 256 > budget
        or max(plan.rows, plan.columns) > ((1 << 32) - 1) // 4
        or (plan.rows > _TILE and plan.rows % 8 != 0)
    ):
        return None

    rows, columns = plan.rows, plan.columns
    column_tiles = (columns + tile_columns - 1) // tile_columns
    tasks = ((rows + tile_rows - 1) // tile_rows) * column_tiles
    source, output = plan.source, plan.output
    src, dst = f"((uintptr_t){source})", f"((uintptr_t){output})"
    disjoint = (
        f"(({src} <= {dst} && {dst} - {src} >= {plan.byte_span}ULL) || "
        f"({dst} <= {src} && {src} - {dst} >= {plan.byte_span}ULL))"
    )
    guard = " && ".join(
        (
            f"({plan.guard})",
            f"(({src} & 3ULL) == 0)",
            f"(({dst} & 31ULL) == 0)",
            disjoint,
        )
    )
    body = f"""    const int64_t nt_transpose_task = (int64_t)GetBlockIdx();
    if (nt_transpose_task >= {tasks}) {{ return; }}
    const int64_t nt_transpose_row = (nt_transpose_task / {column_tiles}) * {tile_rows};
    const int64_t nt_transpose_column = (nt_transpose_task % {column_tiles}) * {tile_columns};
    const int64_t nt_transpose_rows = ({rows} - nt_transpose_row < {tile_rows}) ? ({rows} - nt_transpose_row) : {tile_rows};
    const int64_t nt_transpose_columns = ({columns} - nt_transpose_column < {tile_columns}) ? ({columns} - nt_transpose_column) : {tile_columns};
    TPipe nt_transpose_pipe;
    TQue<TPosition::VECIN, 1> nt_transpose_input_queue;
    TQue<TPosition::VECOUT, 1> nt_transpose_output_queue;
    TBuf<TPosition::VECCALC> nt_transpose_offset_buffer;
    TBuf<TPosition::VECCALC> nt_transpose_row_buffer;
    TBuf<TPosition::VECCALC> nt_transpose_integer_buffer;
    nt_transpose_pipe.InitBuffer(nt_transpose_input_queue, 1, {elements} * sizeof(float));
    nt_transpose_pipe.InitBuffer(nt_transpose_output_queue, 1, {elements} * sizeof(float));
    nt_transpose_pipe.InitBuffer(nt_transpose_offset_buffer, {elements} * sizeof(float));
    nt_transpose_pipe.InitBuffer(nt_transpose_row_buffer, {elements} * sizeof(float));
    nt_transpose_pipe.InitBuffer(nt_transpose_integer_buffer, {elements} * sizeof(int32_t));
    GlobalTensor<float> nt_transpose_source;
    GlobalTensor<float> nt_transpose_output;
    nt_transpose_source.SetGlobalBuffer({source}, {rows * columns}LL);
    nt_transpose_output.SetGlobalBuffer({output}, {rows * columns}LL);
    LocalTensor<float> nt_transpose_input = nt_transpose_input_queue.AllocTensor<float>();
    Duplicate(nt_transpose_input, 0.0f, {elements});
    PipeBarrier<PIPE_ALL>();
    DataCopyExtParams nt_transpose_read((uint16_t)nt_transpose_rows,
        (uint32_t)(nt_transpose_columns * 4LL),
        (uint32_t)(({columns}LL - nt_transpose_columns) * 4LL),
        (uint32_t)({tile_columns // 8} - (nt_transpose_columns + 7) / 8), 0);
    DataCopyPadExtParams<float> nt_transpose_padding(false, 0, 0, 0.0f);
    DataCopyPad(nt_transpose_input,
        nt_transpose_source[nt_transpose_row * {columns}LL + nt_transpose_column],
        nt_transpose_read, nt_transpose_padding);
    nt_transpose_input_queue.EnQue(nt_transpose_input);
    LocalTensor<float> nt_transpose_ready = nt_transpose_input_queue.DeQue<float>();
    LocalTensor<float> nt_transpose_offsets = nt_transpose_offset_buffer.Get<float>();
    LocalTensor<float> nt_transpose_row_offsets = nt_transpose_row_buffer.Get<float>();
    LocalTensor<int32_t> nt_transpose_integer_offsets = nt_transpose_integer_buffer.Get<int32_t>();
    CreateVecIndex(nt_transpose_offsets, 0.0f, {elements});
    PipeBarrier<PIPE_V>();
    Muls(nt_transpose_row_offsets, nt_transpose_offsets, {1.0 / tile_rows}f, {elements});
    PipeBarrier<PIPE_V>();
    Cast(nt_transpose_integer_offsets, nt_transpose_row_offsets, RoundMode::CAST_FLOOR, {elements});
    PipeBarrier<PIPE_V>();
    Cast(nt_transpose_row_offsets, nt_transpose_integer_offsets, RoundMode::CAST_NONE, {elements});
    PipeBarrier<PIPE_V>();
    Muls(nt_transpose_row_offsets, nt_transpose_row_offsets, {float(tile_rows)}f, {elements});
    PipeBarrier<PIPE_V>();
    Sub(nt_transpose_offsets, nt_transpose_offsets, nt_transpose_row_offsets, {elements});
    PipeBarrier<PIPE_V>();
    Muls(nt_transpose_offsets, nt_transpose_offsets, {float(tile_columns * 4)}f, {elements});
    PipeBarrier<PIPE_V>();
    Muls(nt_transpose_row_offsets, nt_transpose_row_offsets, {4.0 / tile_rows}f, {elements});
    PipeBarrier<PIPE_V>();
    Add(nt_transpose_offsets, nt_transpose_offsets, nt_transpose_row_offsets, {elements});
    PipeBarrier<PIPE_V>();
    Cast(nt_transpose_integer_offsets, nt_transpose_offsets, RoundMode::CAST_TRUNC, {elements});
    PipeBarrier<PIPE_V>();
    LocalTensor<float> nt_transpose_values = nt_transpose_output_queue.AllocTensor<float>();
    Gather(nt_transpose_values, nt_transpose_ready,
        nt_transpose_integer_offsets.ReinterpretCast<uint32_t>(), 0, {elements});
    nt_transpose_output_queue.EnQue(nt_transpose_values);
    LocalTensor<float> nt_transpose_written = nt_transpose_output_queue.DeQue<float>();
    DataCopyExtParams nt_transpose_write((uint16_t)nt_transpose_columns,
        (uint32_t)(nt_transpose_rows * 4LL),
        (uint32_t)({tile_rows // 8} - (nt_transpose_rows + 7) / 8),
        (uint32_t)(({rows}LL - nt_transpose_rows) * 4LL), 0);
    DataCopyPad(nt_transpose_output[nt_transpose_column * {rows}LL + nt_transpose_row],
        nt_transpose_written, nt_transpose_write);
    nt_transpose_output_queue.FreeTensor<float>(nt_transpose_written);
    nt_transpose_input_queue.FreeTensor<float>(nt_transpose_ready);"""

    return TransposeBody(body, guard, tasks, ub_bytes, tile_rows, tile_columns)
