"""Namespace helpers for the NineToothed language surface.

The module deliberately keeps the Triton import lazy.  The Python frontend and
the CPU reference interpreter only need the ``ninetoothed.language`` namespace
string, so importing this module must not pull in a GPU toolchain.
"""

import ast

from ninetoothed.symbol import Symbol

__all__ = ["attribute", "call", "libdevice"]  # noqa: F822 - provided by __getattr__

LANGUAGE = "ninetoothed.language"


def __getattr__(name):
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
