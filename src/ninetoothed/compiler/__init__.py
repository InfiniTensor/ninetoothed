"""Public compiler driver contracts."""

from ninetoothed.compiler.driver import (
    DEFAULT_COMPILER,
    Compilation,
    Compiler,
    CompileRequest,
    aot,
    compile_kernel,
    lower,
    make,
    resolve_compile_target,
    resolve_target,
)
from ninetoothed.compiler.jit import jit
from ninetoothed.compiler.runtime import load_built_artifact
from ninetoothed.compiler.singleton_dot_autotune import autotune_singleton_dot
from ninetoothed.targets import PlatformProfile, PlatformRegistry, TargetContext

__all__ = [
    "Compiler",
    "Compilation",
    "CompileRequest",
    "DEFAULT_COMPILER",
    "aot",
    "autotune_singleton_dot",
    "compile_kernel",
    "jit",
    "lower",
    "load_built_artifact",
    "make",
    "PlatformProfile",
    "PlatformRegistry",
    "TargetContext",
    "resolve_compile_target",
    "resolve_target",
]
