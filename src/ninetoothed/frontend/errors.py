"""Frontend lowering exceptions."""


class LoweringError(ValueError):
    """Raised when a Python construct is outside the SSA frontend subset."""


__all__ = ["LoweringError"]
