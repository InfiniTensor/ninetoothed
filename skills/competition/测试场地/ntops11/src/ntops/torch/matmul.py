import ntops


def matmul(input, other, *, out=None):
    assert input.ndim in (2, 3) and other.ndim in (2, 3), (
        "Currently, only 2D and 3D tensors are supported."
    )

    if input.ndim == 2 and other.ndim == 2:
        return ntops.torch.mm(input, other, out=out)

    if input.ndim < 3:
        input = input.unsqueeze(0)

    if other.ndim < 3:
        other = other.unsqueeze(0)

    return ntops.torch.bmm(input, other, out=out)
