"""Shared arrangement and SSA preparation for compilation and interpretation."""

import copy
import inspect
from dataclasses import dataclass

from ninetoothed.frontend.layout import tensor_specs
from ninetoothed.frontend.python import LoweringError, from_application


@dataclass(frozen=True)
class PreparedApplication:
    """The target-neutral result of arranging and lowering an application."""

    parameters: tuple
    arranged: tuple
    tensors: tuple
    program: object


def prepare_application(
    application, *, arrangement=None, tensors=(), tensor_dtypes=None, kernel_name=None
):
    """Run exactly the frontend preparation shared by all execution routes."""
    parameters = tuple(inspect.signature(application).parameters)

    if arrangement is None:
        annotations = inspect.get_annotations(application, eval_str=False)

        try:
            arranged = tuple(copy.deepcopy(annotations[name]) for name in parameters)
        except KeyError as exc:
            raise LoweringError(
                f"Cannot lower `{application.__name__}`: parameter `{exc.args[0]}` "
                "does not have a Tensor annotation."
            ) from exc
    else:
        arranged = arrangement(*copy.deepcopy(tuple(tensors)))
        arranged = arranged if isinstance(arranged, tuple) else (arranged,)

    if len(arranged) != len(parameters):
        raise LoweringError(
            f"Cannot lower `{application.__name__}`: arrangement returned "
            f"{len(arranged)} values for {len(parameters)} parameters."
        )

    for name, tensor in zip(parameters, arranged):
        dtype = (tensor_dtypes or {}).get(name)

        if dtype is not None:
            getattr(tensor, "source", tensor).dtype = dtype

    specs = tensor_specs(parameters, arranged)

    try:
        program = from_application(
            application, specs, kind=kernel_name or application.__name__, strict=True
        )
    except LoweringError as exc:
        raise LoweringError(
            f"Cannot lower `{application.__name__}` through the SSA backend path: {exc}."
        ) from exc

    if program is None:
        raise LoweringError(
            f"Cannot lower `{application.__name__}` through the SSA backend path: "
            "source inspection did not produce ssa.Program."
        )
    return PreparedApplication(parameters, arranged, specs, program)
