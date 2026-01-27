import torch
import triton
import triton.language as tl

from ninetoothed.auto_tuner import AutoTuner

if __name__ == "__main__":

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 64}, num_stages=2, num_warps=2),
            triton.Config({"BLOCK_SIZE": 128}, num_stages=3, num_warps=4),
        ],
        key=["N"],
    )
    @triton.jit
    def kernel(x_ptr, y_ptr, z_ptr, N, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(x_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        tl.store(z_ptr + offs, x + y, mask=mask)

    def func_1(x, y, z):
        N = x.shape[0]

        def grid(meta):
            return (triton.cdiv(N, meta["BLOCK_SIZE"]),)

        kernel[grid](x, y, z, N)

    def func_2(x, y, z):
        torch.add(x, y, out=z)

    device = "cuda"
    size = 1024 * 512
    x = torch.rand(size, device=device, dtype=torch.float32)
    y = torch.rand(size, device=device, dtype=torch.float32)
    z = torch.empty_like(x)

    tuner = AutoTuner(funcs=[func_1, func_2], keys=["add1_1024*512", "add2_1024*512"])
    tuner.run(x, y, z)
    expected = x + y
    print(z)
    print(expected)
    print(f"Max difference: {torch.max(torch.abs(z - expected))}")
    tuner.run(x, y, z)
    expected = x + y
    print(f"Max difference: {torch.max(torch.abs(z - expected))}")
    tuner.run(x, y, z)
    expected = x + y
    print(f"Max difference: {torch.max(torch.abs(z - expected))}")
