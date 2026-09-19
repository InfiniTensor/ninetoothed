"""Built-in interpreter operation handlers.

Importing this package registers every built-in operation.  Additional
operations can be added without touching the interpreter by importing a module
that calls :func:`ninetoothed.interpret.registry.register`.
"""

from . import control, elementwise, memoryops, reduce, tensorops  # noqa: F401

__all__ = ["control", "elementwise", "memoryops", "reduce", "tensorops"]
