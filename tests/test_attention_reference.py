import pytest
import torch

from tests.test_attention import _reference_attention


@pytest.mark.parametrize("is_causal", (False, True))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
def test_sequence_length_one_returns_value(dtype, is_causal):
    q = torch.tensor([[[[1, 2, 3, 4]]]], dtype=dtype)
    k = torch.tensor([[[[4, 3, 2, 1]]]], dtype=dtype)
    v = torch.tensor([[[[1, -2, 3, -4]]]], dtype=dtype)

    output = _reference_attention(q, k, v, is_causal)

    torch.testing.assert_close(output, v)


@pytest.mark.parametrize("is_causal", (False, True))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16))
@pytest.mark.parametrize("seq_len", (1, 31, 32, 33, 64, 65))
def test_zero_scores_return_expected_mean(seq_len, dtype, is_causal):
    q = torch.zeros((1, 2, seq_len, 8), dtype=dtype)
    k = torch.zeros_like(q)
    v = (
        torch.arange(2 * seq_len * 8, dtype=torch.float64).reshape(1, 2, seq_len, 8)
        / 16
    ).to(dtype)

    output = _reference_attention(q, k, v, is_causal)

    if is_causal:
        divisor = torch.arange(1, seq_len + 1, dtype=torch.float64).reshape(
            1, 1, seq_len, 1
        )
        expected = v.double().cumsum(dim=-2) / divisor
    else:
        expected = v.double().mean(dim=-2, keepdim=True).expand_as(v)

    torch.testing.assert_close(output, expected.to(dtype))
