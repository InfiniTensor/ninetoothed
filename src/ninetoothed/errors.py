"""Shared diagnostics for invalid compiler inputs and semantics."""


class LoweringError(ValueError):
    """Raised when an input cannot be represented by the supported IR."""


__all__ = ["LoweringError"]
