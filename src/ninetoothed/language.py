import ast

from ninetoothed.symbol import Symbol

LANGUAGE = "ninetoothed.language"


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
