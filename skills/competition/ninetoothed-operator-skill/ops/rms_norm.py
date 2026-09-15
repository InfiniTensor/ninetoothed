"""
Operator: rms_norm (no learnable weight, eps hardcoded)
Pattern: tile((1, BLOCK_SIZE)) — same as examples/softmax

Root cause diagnosis (recorded for SKILL.md Step 8):
  Tensor(0) scalars (rank-0 tensors) are passed as pointer<fp32> in the
  generated Triton kernel signature. Using them in arithmetic (+ eps)
  produces:
    IncompatibleTypeErrorImpl('invalid operands of type pointer<fp32>
    and triton.language.float32')
  This affects ninetoothed==0.26.0. relu and softmax are unaffected because
  they have NO Tensor(0) arguments.

Fix: remove the Tensor(0) eps argument entirely. Hardcode eps=1e-6 as a
Python literal in the application body. NineToothed substitutes Python
float literals correctly into the generated Triton IR.

Limitation: eps is fixed at 1e-6. If a different eps is needed, the
kernel must be re-declared with a different literal.
"""
import torch
import ninetoothed
import ninetoothed.language as ntl
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(input, output, BLOCK_SIZE=BLOCK_SIZE):
    return (
        input.tile((1, BLOCK_SIZE)),
        output.tile((1, BLOCK_SIZE)),
    )


def application(input, output):
    input_fp32 = ntl.cast(input, ntl.float32)
    # input.shape[-1] → (1, BLOCK_SIZE)[-1] → BLOCK_SIZE constexpr — correct
    # 1e-6 is a Python literal, substituted as-is into Triton IR — correct
    # No Tensor(0) pointer — avoids pointer<fp32> arithmetic error
    output = input_fp32 * ntl.rsqrt(  # noqa: F841
        ntl.sum(input_fp32 * input_fp32) / input.shape[1] + 1e-6
    )


tensors = (Tensor(2), Tensor(2))

kernel = ninetoothed.make(arrangement, application, tensors)


def rms_norm(input: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Only eps=1e-6 is supported because eps is embedded as a kernel literal.
    Non-contiguous inputs are handled with an explicit .contiguous() fallback.
    """
    if eps != 1e-6:
        raise ValueError("Only eps=1e-6 is currently supported")
    if not input.is_contiguous():
        input = input.contiguous()
    output = torch.empty_like(input, dtype=torch.float32)
    kernel(input.float(), output, BLOCK_SIZE=input.shape[-1])
    return output.to(input.dtype)
