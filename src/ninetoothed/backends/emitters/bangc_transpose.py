"""DMA and NRAM transpose for a proved full float32 transpose domain."""

from dataclasses import dataclass

from ninetoothed.backends.emitters.transpose_transfer import plan_transpose_transfer

_TILE = 64


@dataclass(frozen=True)
class TransposeBody:
    body: str
    guard: str
    tasks: int
    nram_bytes: int = _TILE * _TILE * 4 * 2


def render_transpose_transfer(context):
    """Keep exact partial DMA spans and private, initialized padding.

    A false guard retains the caller's complete original source and grid.
    Output row strides divisible by 32 bytes separate concurrent tile writes;
    a shorter matrix has only one tile owner for each complete output row.
    """
    plan = plan_transpose_transfer(context)

    if plan is None or (plan.rows > _TILE and plan.rows % 8 != 0):
        return None

    rows, columns = plan.rows, plan.columns
    row_tiles = (rows + _TILE - 1) // _TILE
    column_tiles = (columns + _TILE - 1) // _TILE
    src, dst = plan.source, plan.output
    src_address, dst_address = f"((uintptr_t){src})", f"((uintptr_t){dst})"
    non_alias = (
        f"(({src_address} <= {dst_address} && {dst_address} - {src_address} >= {plan.byte_span}ULL) || "
        f"({dst_address} <= {src_address} && {src_address} - {dst_address} >= {plan.byte_span}ULL))"
    )
    guard = f"({plan.guard}) && (({dst_address} & 31ULL) == 0) && {non_alias}"
    body = f"""    const int64_t nt_transpose_row = ((int64_t)taskIdX / {column_tiles}) * {_TILE};
    const int64_t nt_transpose_column = ((int64_t)taskIdX % {column_tiles}) * {_TILE};
    const uint32_t nt_transpose_rows = (uint32_t)(({rows} - nt_transpose_row) < {_TILE} ? ({rows} - nt_transpose_row) : {_TILE});
    const uint32_t nt_transpose_columns = (uint32_t)(({columns} - nt_transpose_column) < {_TILE} ? ({columns} - nt_transpose_column) : {_TILE});
    __nram__ float nt_transpose_input[{_TILE * _TILE}];
    __nram__ float nt_transpose_output[{_TILE * _TILE}];
    __bang_write_zero(nt_transpose_input, {_TILE * _TILE});
    __memcpy(nt_transpose_input, {src} + nt_transpose_row * {columns} + nt_transpose_column, nt_transpose_columns * sizeof(float), GDRAM2NRAM, {_TILE} * sizeof(float), {columns} * sizeof(float), nt_transpose_rows - 1);
    __bang_transpose(nt_transpose_output, nt_transpose_input, {_TILE}, {_TILE});
    __memcpy({dst} + nt_transpose_column * {rows} + nt_transpose_row, nt_transpose_output, nt_transpose_rows * sizeof(float), NRAM2GDRAM, {rows} * sizeof(float), {_TILE} * sizeof(float), nt_transpose_columns - 1);"""

    return TransposeBody(body, guard, row_tiles * column_tiles)
