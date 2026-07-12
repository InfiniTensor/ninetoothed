import ninetoothed
from ninetoothed import Symbol, Tensor

BLOCK_SIZE = Symbol("BLOCK_SIZE", constexpr=True)


def arrangement(input, other, output, BLOCK_SIZE=BLOCK_SIZE):
    input_arranged = input.tile((BLOCK_SIZE,))
    other_arranged = other.tile((BLOCK_SIZE,))
    output_arranged = output.tile((BLOCK_SIZE,))

    return input_arranged, other_arranged, output_arranged


def application(input, other, output):
    output = input + other


tensors = tuple(Tensor(1) for _ in range(3))

kernel = ninetoothed.make(arrangement, application, tensors)
