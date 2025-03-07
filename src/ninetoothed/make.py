import inspect

from ninetoothed.jit import jit


def make(arrangement, application, tensors, caller="torch", kernel_name=None):
    """Integrate the arrangement and the application of the tensors.

    :param arrangement: The arrangement of the tensors.
    :param application: The application of the tensors.
    :param tensors: The tensors.
    :param caller: Who will call the compute kernel.
    :param kernel_name: The name for the generated kernel.
    :return: A handle to the compute kernel.
    """

    params = inspect.signature(application).parameters
    types = arrangement(*tensors)
    annotations = {param: type for param, type in zip(params, types)}
    application.__annotations__ = annotations

    return jit(application, caller=caller, kernel_name=kernel_name)
