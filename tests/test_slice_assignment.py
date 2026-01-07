import torch

import ninetoothed as nt


def assign(lhs, rhs, BLOCK_SIZE):
    @nt.jit
    def add_kernel(
        lhs: nt.Tensor(1)[:BLOCK_SIZE],
        rhs: nt.Tensor(1)[:BLOCK_SIZE],
    ):
        lhs = rhs  # noqa: F841

    add_kernel(lhs, rhs)


if __name__ == "__main__":
    DEV = "cuda"
    x = torch.randn(20, device=DEV)
    y = torch.zeros_like(x)
    print("before assign")
    print(y)
    assign(y, x, BLOCK_SIZE=10)
    print("after assign")
    print(y)
