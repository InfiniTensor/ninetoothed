import pytest
import torch

import ntops
from tests.skippers import skip_if_cuda_not_available


def _torch_rotary_position_embedding(input, sin_table, cos_table, interleaved=True):
    batch_size, seq_len, num_heads, emb_dim = input.shape

    assert emb_dim % 2 == 0, "The embedding dimension must be even."

    sin_table = sin_table[None, :, None, :]
    cos_table = cos_table[None, :, None, :]

    if interleaved:
        pair_wise_input = input.view(batch_size, seq_len, num_heads, emb_dim // 2, 2)
        input_0, input_1 = pair_wise_input[..., 0], pair_wise_input[..., 1]
        input_0_rotated = input_0 * cos_table - input_1 * sin_table
        input_1_rotated = input_0 * sin_table + input_1 * cos_table

        return torch.stack((input_0_rotated, input_1_rotated), dim=-1).view(input.shape)
    else:
        input_0 = input[..., : input.shape[-1] // 2]
        input_1 = input[..., input.shape[-1] // 2 :]
        input_0_rotated = input_0 * cos_table - input_1 * sin_table
        input_1_rotated = input_0 * sin_table + input_1 * cos_table

        return torch.cat((input_0_rotated, input_1_rotated), dim=-1)


def _generate_sin_and_cos_tables(
    seq_len, emb_dim, base=10000, dtype=torch.float32, device="cuda"
):
    assert emb_dim % 2 == 0, "The embedding dimension must be even."

    theta = base ** (
        -2 * (torch.arange(emb_dim // 2, dtype=dtype, device=device) / emb_dim)
    )

    positions = torch.arange(seq_len, dtype=dtype, device=device).unsqueeze(1)
    sin_table = torch.sin(positions * theta)
    cos_table = torch.cos(positions * theta)

    return sin_table, cos_table


@skip_if_cuda_not_available
@pytest.mark.parametrize("device", ("cuda",))
@pytest.mark.parametrize(
    "dtype, rtol, atol", ((torch.float32, 0, 0.001), (torch.float16, 0.001, 0.001))
)
@pytest.mark.parametrize("inplace", (False, True))
@pytest.mark.parametrize("interleaved", (False, True))
@pytest.mark.parametrize("emb_dim", (32, 64))
@pytest.mark.parametrize("num_heads", (1, 8))
@pytest.mark.parametrize("seq_len", (1, 128))
@pytest.mark.parametrize("batch_size", (1, 4))
def test_rotary_position_embedding(
    batch_size,
    seq_len,
    num_heads,
    emb_dim,
    interleaved,
    inplace,
    dtype,
    device,
    rtol,
    atol,
):
    input = torch.randn(
        batch_size, seq_len, num_heads, emb_dim, dtype=dtype, device=device
    )
    sin_table, cos_table = _generate_sin_and_cos_tables(
        seq_len, emb_dim, dtype=dtype, device=device
    )

    ninetoothed_output = ntops.torch.rotary_position_embedding(
        input.clone() if inplace else input,
        sin_table,
        cos_table,
        interleaved=interleaved,
        inplace=inplace,
    )
    reference_output = _torch_rotary_position_embedding(
        input, sin_table, cos_table, interleaved=interleaved
    )

    assert torch.allclose(ninetoothed_output, reference_output, rtol=rtol, atol=atol)
