"""Discover the RVNE SDK and construct RISC-V cross-compilation commands."""

import os
import platform
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

RVNE_ISA = "rv64imafcvzne"


@dataclass(frozen=True)
class RvneToolchain:
    root: Path
    compiler: Path
    emulator: Path
    gcc_toolchain: Path
    sysroot: Path


def find_rvne_toolchain(root=None) -> RvneToolchain:
    """Locate the SDK without executing its installation or build scripts."""
    root = root or os.environ.get("NINETOOTHED_RVNE_TOOLCHAIN")

    if not root:
        raise RuntimeError(
            "RVNE AOT requires the RVNE SDK; set NINETOOTHED_RVNE_TOOLCHAIN "
            "or the toolchain_root backend option."
        )

    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise RuntimeError(
            "The RVNE SDK requires Linux x86_64; use a Linux host or WSL."
        )

    sdk = Path(root).expanduser().resolve()
    sdk = sdk / "toolchain" if (sdk / "toolchain").is_dir() else sdk
    compiler = sdk / "llvm" / "bin" / "clang++"

    if not compiler.is_file():
        compiler = sdk / "llvm" / "bin" / "clang"

    emulator = sdk / "qemu" / "bin" / "qemu-riscv64"
    gcc_toolchain = sdk / "xuantie-gnu-toolchain"
    sysroot = gcc_toolchain / "sysroot"

    for path in (compiler, emulator):
        if not path.is_file() or not os.access(path, os.X_OK):
            raise RuntimeError(
                f"RVNE SDK executable is missing or not executable: `{path}`."
            )

    if not sysroot.is_dir():
        raise RuntimeError(f"RVNE SDK sysroot is missing: `{sysroot}`.")

    return RvneToolchain(sdk, compiler, emulator, gcc_toolchain, sysroot)


def rvne_compiler_identity(root=None, *, required=False):
    """Return compiler identity and the target ISA for binary cache isolation."""
    try:
        toolchain = find_rvne_toolchain(root)
    except RuntimeError:
        if required:
            raise
        return {
            "available": False,
            "isa": RVNE_ISA,
            "root": str(root or os.environ.get("NINETOOTHED_RVNE_TOOLCHAIN", "")),
        }

    compiler = toolchain.compiler.resolve(strict=True)
    stat = compiler.stat()

    return {
        "available": True,
        "isa": RVNE_ISA,
        "compiler": str(compiler),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "version": _compiler_version(str(compiler), stat.st_size, stat.st_mtime_ns),
        "sysroot": str(toolchain.sysroot.resolve()),
        "gcc_toolchain": str(toolchain.gcc_toolchain.resolve()),
        "emulator": str(toolchain.emulator.resolve()),
        "command": rvne_compile_command(toolchain, "<source>", "<output>"),
    }


@lru_cache(maxsize=16)
def _compiler_version(path, size, mtime_ns):
    del size, mtime_ns

    try:
        result = subprocess.run(
            [path, "--version"], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Cannot query RVNE compiler `{path}`.") from exc

    version = (result.stdout + result.stderr).strip()

    if not version:
        raise RuntimeError(f"RVNE compiler `{path}` returned an empty version.")
    return version


def rvne_compile_command(toolchain, source, output):
    """Build an executable for the SDK's RVNE user-mode emulator."""
    return (
        str(toolchain.compiler),
        "-x",
        "c++",
        "-std=c++17",
        "-O2",
        "-fwrapv",
        "--target=riscv64-unknown-linux-gnu",
        f"-march={RVNE_ISA}",
        "-fuse-ld=lld",
        "-static",
        f"--sysroot={toolchain.sysroot}",
        f"--gcc-toolchain={toolchain.gcc_toolchain}",
        str(source),
        "-lm",
        "-o",
        str(output),
    )
