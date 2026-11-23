"""Built-in interpreter operation handlers.

Importing this package registers every built-in operation.  A module that calls
:func:`ninetoothed.interpret.registry.register` adds more without touching the
interpreter.
"""

from . import control, elementwise, memoryops, reduce, tensorops  # noqa: F401

__all__ = ["control", "elementwise", "memoryops", "reduce", "tensorops"]
