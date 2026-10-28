import ast

from ninetoothed.symbol import Symbol

__all__ = ["libdevice"]  # noqa: F822 - exported lazily through __getattr__.

LANGUAGE = "ninetoothed.language"


def __getattr__(name):
    """Load optional GPU math bindings only when a caller uses them."""
    if name == "libdevice":
        from triton.language.extra import libdevice

        return libdevice

    raise AttributeError(f"Module {__name__!r} has no attribute {name!r}.")


def call(func, *args, **kwargs):
    return Symbol(
        ast.Call(
            func=attribute(func).node,
            args=[Symbol(arg).node for arg in args],
            keywords=[
                ast.keyword(arg=kwarg, value=Symbol(kwargs[kwarg]).node)
                for kwarg in kwargs
            ],
        )
    )


def attribute(attr):
    return Symbol(ast.parse(f"{LANGUAGE}.{attr}", mode="eval").body)
