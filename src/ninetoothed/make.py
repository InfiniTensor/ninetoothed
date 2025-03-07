import inspect

from ninetoothed.aot import aot
from ninetoothed.jit import jit


def make(
    arrangement,
    application,
    tensors,
    caller="torch",
    kernel_name=None,
    output_dir=None,
    num_warps=4,
    num_stages=3,
):
    """Integrate the arrangement and the application of the tensors.

    :param arrangement: The arrangement of the tensors.
    :param application: The application of the tensors.
    :param tensors: The tensors.
    :param caller: Who will call the compute kernel.
    :param kernel_name: The name for the generated kernel.
    :param output_dir: The directory to store the generated files.
    :param num_warps: The number of warps to use.
    :param num_stages: The number of pipeline stages.
    :return: A handle to the compute kernel.
    """

    params = inspect.signature(application).parameters
    types = arrangement(*tensors)
    annotations = {param: type for param, type in zip(params, types)}
    application.__annotations__ = annotations

    if caller == "torch":
        return jit(application, caller=caller, kernel_name=kernel_name)

    return aot(
        application,
        caller=caller,
        kernel_name=kernel_name,
        output_dir=output_dir,
        num_warps=num_warps,
        num_stages=num_stages,
    )
