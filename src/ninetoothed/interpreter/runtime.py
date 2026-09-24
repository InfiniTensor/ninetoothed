"""Runtime state of the CPU reference interpreter.

The state carries everything the interpreter needs while it walks one SSA
program instance: the SSA value environment, the CPU memory buffers, the program
instance index, the active mask, and the resolved symbols used by layout
expressions.
"""

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ninetoothed.interpreter.errors import InvalidProgramError

_ALLOWED_BUILTINS = {
    "abs": abs,
    "bool": bool,
    "ceil": math.ceil,
    "float": float,
    "floor": math.floor,
    "int": int,
    "len": len,
    "max": max,
    "min": min,
    "pow": pow,
    "round": round,
}


@dataclass
class RuntimeState:
    """Mutable interpreter state for one program instance.

    :param symbols: The resolved layout symbols, such as source sizes and strides.
    :param environment: The SSA value environment.
    :param memory: The CPU buffers of the tensor arguments.
    :param accesses: The access mappings of the tensor arguments.
    :param sources: The symbolic source shapes of the tensor arguments.
    :param program_id: The linearized program instance index.
    :param program_shape: The shape of the program-instance domain.
    :param mask: The mask of the most recent memory access.
    """

    symbols: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)
    memory: dict = field(default_factory=dict)
    accesses: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    program_id: int = 0
    program_shape: tuple = ()
    mask: Any = None

    def bind(self, name: str, value) -> None:
        """Bind an SSA value in the environment."""
        self.environment[name] = value

    def lookup(self, name: str):
        """Return the value bound to an SSA name."""
        try:
            return self.environment[name]
        except KeyError as exc:
            raise InvalidProgramError(
                f"SSA value `{name}` was used before it was bound."
            ) from exc

    def child(self) -> "RuntimeState":
        """Return a nested state that inherits the layout and memory tables."""
        return RuntimeState(
            symbols=self.symbols,
            environment=dict(self.environment),
            memory=self.memory,
            accesses=self.accesses,
            sources=self.sources,
            program_id=self.program_id,
            program_shape=self.program_shape,
            mask=self.mask,
        )

    def access(self, name: str):
        """Return the access mapping of a tensor argument."""
        try:
            return self.accesses[name]
        except KeyError as exc:
            raise InvalidProgramError(
                f"Tensor `{name}` is not an interpreted tensor argument."
            ) from exc

    def source_shape(self, name: str) -> tuple:
        """Return the resolved source shape of a tensor argument."""
        shape = self.sources.get(name)

        if shape is None:
            raise InvalidProgramError(f"Tensor `{name}` has no recorded source shape.")

        return tuple(int(self.evaluate(dim)) for dim in shape)

    def evaluate(self, text) -> Any:
        """Evaluate a symbolic layout or index expression recorded in the SSA.

        :param text: The expression text, for example ``ninetoothed_..._size_0``,
            ``output.shape``, or ``2 * BLOCK_SIZE``.
        :return: The evaluated expression.
        """
        namespace = dict(self.symbols)
        namespace.update(self.environment)
        namespace["np"] = np
        namespace["math"] = math

        try:
            return eval(str(text), {"__builtins__": _ALLOWED_BUILTINS}, namespace)
        except Exception as exc:
            raise InvalidProgramError(
                f"Cannot evaluate the layout expression `{text}`: {exc}."
            ) from exc

    def evaluate_shape(self, shape) -> tuple:
        """Resolve an SSA shape, which is recorded as text or as a shape value.

        :param shape: A shape tuple, a shape text such as ``output.shape``, or a
            single dimension text.
        :return: The resolved shape as a tuple of integers.
        """
        if shape is None:
            return ()

        if isinstance(shape, str):
            resolved = self.evaluate(shape)

            if isinstance(resolved, (tuple, list)):
                return tuple(int(dim) for dim in resolved)

            return (int(resolved),)

        if isinstance(shape, (tuple, list)):
            return tuple(
                int(self.evaluate(dim) if isinstance(dim, str) else dim)
                for dim in shape
            )

        return (int(shape),)


__all__ = ["RuntimeState"]
