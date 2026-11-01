"""Shared arrangement and SSA preparation for compilation and interpretation."""

from dataclasses import dataclass

from ninetoothed.compiler.driver import DEFAULT_COMPILER, CompileRequest


@dataclass(frozen=True)
class PreparedApplication:
    """The target-neutral result of arranging and lowering an application."""

    parameters: tuple
    meta_defaults: dict
    tensors: tuple
    program: object


def prepare_application(
    application, *, arrangement=None, tensors=(), tensor_dtypes=None, kernel_name=None
):
    """Run exactly the frontend preparation shared by all execution routes."""
    module = DEFAULT_COMPILER.frontend(
        CompileRequest(
            application=application,
            arrangement=arrangement,
            tensors=tuple(tensors),
            tensor_dtypes=tensor_dtypes,
            kernel_name=kernel_name,
        )
    )

    return PreparedApplication(
        module.public_args, module.meta_defaults, module.tensors, module.program
    )
