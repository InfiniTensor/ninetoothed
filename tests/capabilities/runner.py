"""Run external capability probes in isolated Python processes."""

import importlib
import json
import subprocess
import sys

from tests.capabilities.model import CapabilityResult


def run_python_probe(name: str, source: str, timeout: float = 60) -> CapabilityResult:
    try:
        completed = subprocess.run(
            [sys.executable, "-c", source],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"Capability probe {name!r} timed out after {timeout} seconds."
        ) from error

    if completed.returncode != 0:
        reason = f"{name} probe exited with code {completed.returncode}"
        diagnostic = completed.stderr.strip()

        if diagnostic:
            reason = f"{reason}: {diagnostic}"

        return CapabilityResult(supported=False, reason=reason)

    lines = tuple(line for line in completed.stdout.splitlines() if line.strip())

    if not lines:
        raise RuntimeError(f"Capability probe {name!r} returned invalid empty output.")

    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Capability probe {name!r} returned invalid JSON output."
        ) from error

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Capability probe {name!r} returned an invalid non-object payload."
        )

    status = payload.get("status")

    if status == "supported":
        reason = payload.get("reason", "")

        if not isinstance(reason, str):
            raise RuntimeError(f"Capability probe {name!r} returned an invalid reason.")

        return CapabilityResult(supported=True, reason=reason)

    if status in {"unavailable", "error"}:
        reason = payload.get("reason")

        if not isinstance(reason, str) or not reason:
            raise RuntimeError(f"Capability probe {name!r} returned an invalid reason.")

        if status == "error":
            raise RuntimeError(f"Capability probe {name!r} failed: {reason}.")

        return CapabilityResult(supported=False, reason=f"{name}: {reason}")

    raise RuntimeError(f"Capability probe {name!r} returned an invalid status.")


def resolve_probe(reference: str) -> CapabilityResult:
    module_name, separator, function_name = reference.partition(":")

    if not separator or not module_name or not function_name or ":" in function_name:
        raise ValueError(
            "Capability probe references must use the fully qualified "
            "`module:function` form."
        )

    module = importlib.import_module(module_name)
    probe = getattr(module, function_name)

    if not callable(probe):
        raise TypeError(f"Capability probe {reference!r} is not callable.")

    result = probe()

    if not isinstance(result, CapabilityResult):
        raise TypeError(
            f"Capability probe {reference!r} did not return a CapabilityResult."
        )

    return result
