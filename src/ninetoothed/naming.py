import re


def make_constexpr(name):
    return _add_prefix(name, _CONSTEXPR)


def make_meta(name):
    return _add_prefix(name, _META)


def is_constexpr(name):
    return _CONSTEXPR in _find_prefixes(name) or is_meta(name)


def is_meta(name):
    return _META in _find_prefixes(name)


def remove_prefixes(name):
    return _PREFIX_PATTERN.sub("", name)


_CONSTEXPR = "constexpr"

_META = "meta"

_PREFIX_PATTERN = re.compile(r"ninetoothed_((?!_).*?)_prefix_")


def _add_prefix(name, string):
    return f"{_make_prefix(string)}{name}"


def _make_prefix(string):
    return f"ninetoothed_{string}_prefix_"


def _find_prefixes(name):
    return set(_PREFIX_PATTERN.findall(name))
