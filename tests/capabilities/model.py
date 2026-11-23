"""Capability probe result types."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CapabilityResult:
    supported: bool
    reason: str = ""
