import torch

from examples.fused_rms_norm.kernel import kernel


def fused_rms_norm(x, w, eps=None):
    if eps is None:
        eps = torch.finfo(x.dtype).eps

    x_2d = x.view(-1, x.shape[-1])
    w_2d = w.expand_as(x_2d)
    y_2d = torch.empty_like(x_2d)

    kernel(x_2d, w_2d, eps, y_2d, BLOCK_SIZE=x.shape[-1])

    return y_2d.view(x.shape)
