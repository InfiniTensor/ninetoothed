"""Language and platform target contracts for NineToothed compilation."""

import json
import os
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping

from ninetoothed.backends.core import Target, normalize_target


class UnsupportedTargetCapabilityError(ValueError):
    """Raised when a resolved target explicitly rejects an IR capability."""


@dataclass(frozen=True, kw_only=True)
class PlatformProfile:
    """One concrete platform profile without imposing a family hierarchy."""

    name: str
    aliases: tuple[str, ...] = ()
    compute_arch: str | None = None
    device_types: tuple[str, ...] = ("cuda",)
    backend_modes: Mapping[str, frozenset[str]] = field(default_factory=dict)
    supported_capabilities: frozenset[str] = frozenset()
    unsupported_capabilities: frozenset[str] = frozenset()
    backend_supported_capabilities: Mapping[str, frozenset[str]] = field(
        default_factory=dict
    )
    backend_unsupported_capabilities: Mapping[str, frozenset[str]] = field(
        default_factory=dict
    )
    constraints: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = platform_id_for(self.name)
        aliases = tuple(platform_id_for(alias) for alias in self.aliases)
        device_types = tuple(
            dict.fromkeys(
                str(device_type).strip().lower() for device_type in self.device_types
            )
        )
        supported_capabilities = frozenset(
            capability_id_for(value) for value in self.supported_capabilities
        )
        unsupported_capabilities = frozenset(
            capability_id_for(value) for value in self.unsupported_capabilities
        )
        overlap = supported_capabilities & unsupported_capabilities
        backend_modes = {
            backend_id_for(backend): frozenset(
                materialization_mode_for(mode) for mode in modes
            )
            for backend, modes in dict(self.backend_modes).items()
        }
        backend_supported_capabilities = {
            backend_id_for(backend): frozenset(
                capability_id_for(value) for value in capabilities
            )
            for backend, capabilities in dict(
                self.backend_supported_capabilities
            ).items()
        }
        backend_unsupported_capabilities = {
            backend_id_for(backend): frozenset(
                capability_id_for(value) for value in capabilities
            )
            for backend, capabilities in dict(
                self.backend_unsupported_capabilities
            ).items()
        }

        if not device_types:
            raise ValueError(f"Platform profile `{name}` must accept a device type.")

        if any(not device_type for device_type in device_types):
            raise ValueError(f"Platform profile `{name}` has an empty device type.")

        if overlap:
            capabilities = ", ".join(sorted(overlap))
            raise ValueError(
                f"Platform profile `{name}` both supports and rejects: {capabilities}."
            )

        capability_backends = set(backend_supported_capabilities) | set(
            backend_unsupported_capabilities
        )
        unknown_backends = capability_backends - set(backend_modes)

        if unknown_backends:
            backends = ", ".join(sorted(unknown_backends))
            raise ValueError(
                f"Platform profile `{name}` declares capabilities for unsupported "
                f"backends: {backends}."
            )

        for backend in capability_backends:
            backend_overlap = (
                supported_capabilities
                | backend_supported_capabilities.get(backend, frozenset())
            ) & (
                unsupported_capabilities
                | backend_unsupported_capabilities.get(backend, frozenset())
            )

            if backend_overlap:
                capabilities = ", ".join(sorted(backend_overlap))
                raise ValueError(
                    f"Platform profile `{name}` backend `{backend}` both supports "
                    f"and rejects: {capabilities}."
                )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "device_types", device_types)
        object.__setattr__(
            self,
            "backend_modes",
            MappingProxyType(backend_modes),
        )
        object.__setattr__(
            self,
            "compute_arch",
            None
            if self.compute_arch is None
            else compute_arch_id_for(self.compute_arch),
        )
        object.__setattr__(
            self,
            "supported_capabilities",
            supported_capabilities,
        )
        object.__setattr__(
            self,
            "unsupported_capabilities",
            unsupported_capabilities,
        )
        object.__setattr__(
            self,
            "backend_supported_capabilities",
            MappingProxyType(backend_supported_capabilities),
        )
        object.__setattr__(
            self,
            "backend_unsupported_capabilities",
            MappingProxyType(backend_unsupported_capabilities),
        )
        object.__setattr__(
            self,
            "constraints",
            _freeze_mapping(self.constraints),
        )
        object.__setattr__(
            self,
            "metadata",
            _freeze_mapping(self.metadata),
        )

    def as_metadata(self) -> Mapping[str, Any]:
        """Return the stable, serializable portion of this profile."""
        return {
            "name": self.name,
            "compute_arch": self.compute_arch,
            "device_types": self.device_types,
            "backend_modes": {
                backend: tuple(sorted(modes))
                for backend, modes in self.backend_modes.items()
            },
            "supported_capabilities": tuple(sorted(self.supported_capabilities)),
            "unsupported_capabilities": tuple(sorted(self.unsupported_capabilities)),
            "backend_supported_capabilities": {
                backend: tuple(sorted(capabilities))
                for backend, capabilities in self.backend_supported_capabilities.items()
            },
            "backend_unsupported_capabilities": {
                backend: tuple(sorted(capabilities))
                for backend, capabilities in self.backend_unsupported_capabilities.items()
            },
            "constraints": _metadata_value(self.constraints),
            "metadata": _metadata_value(self.metadata),
        }


@dataclass(frozen=True, kw_only=True)
class TargetContext:
    """Resolved language and platform inputs shared by compiler stages."""

    backend: Target
    platform: PlatformProfile
    compute_arch: str | None = None

    def __post_init__(self) -> None:
        backend = normalize_target(self.backend)
        compute_arch = (
            None
            if self.compute_arch is None
            else compute_arch_id_for(self.compute_arch)
        )

        if (
            compute_arch is not None
            and self.platform.compute_arch is not None
            and compute_arch != self.platform.compute_arch
        ):
            raise ValueError(
                f"Compute architecture `{compute_arch}` conflicts with platform "
                f"`{self.platform.name}` architecture "
                f"`{self.platform.compute_arch}`."
            )

        if backend.value not in self.platform.backend_modes:
            raise ValueError(
                f"Backend `{backend.value}` is not supported by platform "
                f"`{self.platform.name}`."
            )

        object.__setattr__(self, "backend", backend)
        object.__setattr__(
            self,
            "compute_arch",
            compute_arch or self.platform.compute_arch,
        )

    @property
    def device_types(self) -> tuple[str, ...]:
        return self.platform.device_types

    def capability_report(self, required: tuple[str, ...]) -> Mapping[str, Any]:
        """Classify requested capabilities without guessing unknown support."""
        required_set = frozenset(capability_id_for(value) for value in required)
        supported_capabilities = self.platform.supported_capabilities | (
            self.platform.backend_supported_capabilities.get(
                self.backend.value, frozenset()
            )
        )
        unsupported_capabilities = self.platform.unsupported_capabilities | (
            self.platform.backend_unsupported_capabilities.get(
                self.backend.value, frozenset()
            )
        )
        supported = required_set & supported_capabilities
        unsupported = required_set & unsupported_capabilities
        unresolved = required_set - supported - unsupported

        return {
            "required": tuple(sorted(required_set)),
            "supported": tuple(sorted(supported)),
            "unsupported": tuple(sorted(unsupported)),
            "unresolved": tuple(sorted(unresolved)),
        }

    def validate_capabilities(self, required: tuple[str, ...]) -> Mapping[str, Any]:
        """Reject capabilities known to be unsupported by this target."""
        report = self.capability_report(required)
        unsupported = tuple(report["unsupported"])

        if unsupported:
            capabilities = ", ".join(f"`{name}`" for name in unsupported)
            raise UnsupportedTargetCapabilityError(
                f"Target `{self.platform.name}` does not support {capabilities}."
            )
        return report

    def validate_materialization(self, mode: str) -> None:
        """Reject materialization modes not implemented for this profile."""
        mode = materialization_mode_for(mode)
        supported = self.platform.backend_modes[self.backend.value]

        if mode not in supported:
            raise ValueError(
                f"Backend `{self.backend.value}` on platform "
                f"`{self.platform.name}` does not support `{mode}` materialization."
            )

    def as_metadata(self) -> Mapping[str, Any]:
        return {
            "backend": self.backend.value,
            "platform": self.platform.name,
            "compute_arch": self.compute_arch,
            "device_types": self.device_types,
            "profile": self.platform.as_metadata(),
        }


class PlatformRegistry:
    """Explicit platform registry with replaceable aliases and profiles."""

    def __init__(self) -> None:
        self._profiles: MutableMapping[str, PlatformProfile] = {}
        self._aliases: MutableMapping[str, str] = {}

    def register(self, profile: PlatformProfile, *, replace: bool = False) -> None:
        names = (profile.name, *profile.aliases)
        conflicts = {
            name: name if name in self._profiles else self._aliases.get(name)
            for name in names
            if name in self._profiles or name in self._aliases
        }

        if conflicts and (
            not replace or any(owner != profile.name for owner in conflicts.values())
        ):
            names_text = ", ".join(f"`{name}`" for name in conflicts)
            raise ValueError(f"Platform id or alias already registered: {names_text}.")

        if replace:
            self._remove_profile(profile.name)

        self._profiles[profile.name] = profile

        for alias in profile.aliases:
            self._aliases[alias] = profile.name

    def get(self, name: str | None) -> PlatformProfile:
        platform_id = platform_id_for("generic" if name is None else name)
        platform_id = self._aliases.get(platform_id, platform_id)

        try:
            return self._profiles[platform_id]
        except KeyError as exc:
            available = ", ".join(sorted(self._profiles))
            raise ValueError(
                f"Unsupported platform `{platform_id}`. Available platforms: "
                f"{available}."
            ) from exc

    def profiles(self) -> tuple[PlatformProfile, ...]:
        return tuple(self._profiles.values())

    def _remove_profile(self, name: str) -> None:
        self._profiles.pop(name, None)

        for alias, target in tuple(self._aliases.items()):
            if target == name:
                del self._aliases[alias]


def platform_id_for(name: str) -> str:
    platform_id = str(name).strip().lower().replace("_", "-")

    if not platform_id:
        raise ValueError("Platform id must not be empty.")
    return platform_id


def compute_arch_id_for(name: str) -> str:
    compute_arch = str(name).strip().lower()

    if not compute_arch:
        raise ValueError("Compute architecture must not be empty.")
    return compute_arch


def capability_id_for(name: str) -> str:
    capability = str(name).strip().lower()

    if not capability:
        raise ValueError("Capability id must not be empty.")
    return capability


def backend_id_for(name: Target | str) -> str:
    return normalize_target(name).value


def materialization_mode_for(mode: str) -> str:
    mode = str(mode).strip().lower()

    if mode not in {"jit", "aot"}:
        raise ValueError(f"Unsupported materialization mode `{mode}`.")
    return mode


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    items = sorted(
        ((str(key), _freeze_value(item)) for key, item in dict(value).items()),
        key=lambda item: item[0],
    )

    return MappingProxyType(dict(items))


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)

    if isinstance(value, (tuple, list)):
        return tuple(_freeze_value(item) for item in value)

    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)

    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    raise TypeError(
        f"Unsupported platform profile metadata value of type `{type(value).__name__}`."
    )


def _metadata_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _metadata_value(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }

    if isinstance(value, (set, frozenset)):
        items = (_metadata_value(item) for item in value)

        return tuple(
            sorted(
                items,
                key=lambda item: json.dumps(
                    item,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        )

    if isinstance(value, (tuple, list)):
        return tuple(_metadata_value(item) for item in value)
    return value


def _backend_modes(
    backends: tuple[Target, ...],
    *,
    modes: tuple[str, ...] = ("jit", "aot"),
) -> Mapping[str, frozenset[str]]:
    supported_modes = frozenset(modes)

    return {backend.value: supported_modes for backend in backends}


def create_default_platform_registry() -> PlatformRegistry:
    """Create the current concrete profiles without defining family inheritance."""
    registry = PlatformRegistry()

    for profile in (
        PlatformProfile(
            name="generic",
            device_types=("cuda",),
            backend_modes=_backend_modes(tuple(Target)),
            metadata={"legacy_default": True},
        ),
        PlatformProfile(
            name="nvidia-a100",
            aliases=("a100",),
            compute_arch="sm_80",
            device_types=("cuda",),
            backend_modes=_backend_modes(tuple(Target)),
            unsupported_capabilities=frozenset({"dtype.fp8"}),
        ),
        PlatformProfile(
            name="nvidia-h100",
            aliases=("h100",),
            compute_arch="sm_90",
            device_types=("cuda",),
            backend_modes=_backend_modes(tuple(Target)),
            supported_capabilities=frozenset({"dtype.fp8"}),
        ),
        PlatformProfile(
            name="cambricon-mlu590",
            aliases=("mlu590",),
            compute_arch="mlu590",
            device_types=("mlu",),
            backend_modes=_backend_modes(
                (Target.TRITON, Target.BANGC), modes=("jit", "aot")
            ),
            constraints={"compiler_options": {"triton": {"max_num_configs": 1}}},
            metadata={
                "triton_grid_limit": 65535,
                "bangc": {"arch": "mlu590"},
            },
        ),
        PlatformProfile(
            name="ascend-910b3",
            compute_arch="ascend910b3",
            device_types=("npu",),
            backend_modes=_backend_modes((Target.TRITON,), modes=("jit",)),
            unsupported_capabilities=frozenset({"math.pow"}),
            backend_unsupported_capabilities={
                Target.TRITON.value: frozenset({"dtype.fp8"})
            },
            constraints={"compiler_options": {"triton": {"max_num_configs": 1}}},
            metadata={"triton_block_size": 512},
        ),
        PlatformProfile(
            name="ascend-910b4",
            compute_arch="ascend910b4",
            device_types=("npu",),
            backend_modes=_backend_modes((Target.TRITON,), modes=("jit",)),
            unsupported_capabilities=frozenset({"math.pow"}),
            constraints={"compiler_options": {"triton": {"max_num_configs": 1}}},
            metadata={"triton_block_size": 512},
        ),
        PlatformProfile(
            name="tianshu-t150",
            compute_arch="ivcore11",
            device_types=("cuda",),
            backend_modes=_backend_modes((Target.TRITON,), modes=("jit",)),
        ),
        PlatformProfile(
            name="tianshu-t200",
            compute_arch="ivcore20",
            device_types=("cuda",),
            backend_modes=_backend_modes((Target.TRITON,), modes=("jit",)),
            constraints={"compiler_options": {"triton": {"fixed_num_stages": 1}}},
            metadata={"triton_dot_operand_coercions": {"float32": "float16"}},
        ),
        PlatformProfile(
            name="muxi-c550",
            compute_arch="c550",
            device_types=("cuda",),
            backend_modes={
                Target.TRITON.value: frozenset(("jit",)),
                Target.CUDA.value: frozenset(("jit",)),
            },
            backend_unsupported_capabilities={
                Target.CUDA.value: frozenset({"dtype.fp8"})
            },
            constraints={
                "compiler_options": {
                    "triton": {
                        "max_num_configs": 8,
                        "num_stages": (1, 2),
                    }
                }
            },
            metadata={
                "cuda": {
                    "arch": "native",
                    "wmma": False,
                }
            },
        ),
        PlatformProfile(
            name="muxi-x203",
            compute_arch="hpcc-80",
            device_types=("cuda",),
            backend_modes={},
        ),
        PlatformProfile(
            name="moore-s5000",
            compute_arch="musa-31",
            device_types=("musa",),
            backend_modes=_backend_modes((Target.TRITON,), modes=("jit",)),
            backend_unsupported_capabilities={
                Target.TRITON.value: frozenset({"reduction.min"})
            },
        ),
        PlatformProfile(
            name="hygon-bw1000",
            compute_arch="gfx936",
            device_types=("cuda",),
            backend_modes={
                Target.TRITON.value: frozenset(("jit",)),
                Target.CUDA.value: frozenset(("jit",)),
            },
            metadata={
                "cuda": {
                    "arch": "native",
                    "wmma": False,
                }
            },
        ),
        PlatformProfile(
            name="kunlun-p800",
            compute_arch="xpu3",
            device_types=("xpu",),
            backend_modes={},
        ),
    ):
        registry.register(profile)
    return registry


_DEFAULT_PLATFORM_REGISTRY: PlatformRegistry | None = None


def default_platform_registry() -> PlatformRegistry:
    global _DEFAULT_PLATFORM_REGISTRY

    if _DEFAULT_PLATFORM_REGISTRY is None:
        _DEFAULT_PLATFORM_REGISTRY = create_default_platform_registry()
    return _DEFAULT_PLATFORM_REGISTRY


def resolve_target_context(
    backend: Target | str | None,
    *,
    platform: str | None = None,
    compute_arch: str | None = None,
    registry: PlatformRegistry | None = None,
) -> TargetContext:
    """Resolve language and current concrete platform without runtime probing."""
    backend_name = normalize_target(
        os.environ.get("NINETOOTHED_BACKEND") if backend is None else backend
    )
    platform_name = (
        os.environ.get("NINETOOTHED_PLATFORM") if platform is None else platform
    )
    profile = (registry or default_platform_registry()).get(platform_name)
    requested_arch = (
        os.environ.get("NINETOOTHED_COMPUTE_ARCH")
        if compute_arch is None
        else compute_arch
    )
    requested_arch = (
        None if requested_arch is None else compute_arch_id_for(requested_arch)
    )

    return TargetContext(
        backend=backend_name,
        platform=profile,
        compute_arch=requested_arch,
    )


def target_backend_options(
    target: TargetContext,
    options: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Bind backend toolchain options to an explicit concrete target."""
    normalized = dict(options)

    if target.backend == Target.CUDA:
        return _cuda_backend_options(target, normalized)

    if target.backend == Target.BANGC:
        return _bangc_backend_options(target, normalized)

    return normalized


def _cuda_backend_options(target: TargetContext, normalized: Mapping[str, Any]):
    if target.compute_arch is None and target.platform.name == "generic":
        return normalized

    if target.compute_arch is None:
        raise ValueError(
            f"CUDA platform `{target.platform.name}` must define a compute architecture."
        )

    from ninetoothed.backends.toolchain import (
        cuda_compute_capability,
        normalize_cuda_arch,
    )

    platform_cuda = target.platform.metadata.get("cuda", {})
    platform_arch = platform_cuda.get("arch", target.compute_arch)

    if platform_arch == "native":
        requested_arch = normalized.get("arch")

        if requested_arch not in {None, "native", target.compute_arch}:
            raise ValueError(
                f"CUDA architecture `{requested_arch}` conflicts with target "
                f"architecture `{target.compute_arch}` through its native "
                "vendor toolchain."
            )

        if normalized.get("compute_capability") is not None:
            raise ValueError(
                f"CUDA compute capability is not defined for target "
                f"architecture `{target.compute_arch}`."
            )

        normalized["arch"] = "native"
        normalized.pop("compute_capability", None)

        return normalized

    target_arch = normalize_cuda_arch(platform_arch)
    requested_arch = normalize_cuda_arch(normalized.get("arch", target_arch))

    if requested_arch not in {"native", target_arch}:
        raise ValueError(
            f"CUDA architecture `{requested_arch}` conflicts with target "
            f"architecture `{target_arch}`."
        )

    compute_capability = cuda_compute_capability(target_arch)
    requested_capability = normalized.get("compute_capability")

    if (
        requested_capability is not None
        and str(requested_capability) != compute_capability
    ):
        raise ValueError(
            f"CUDA compute capability `{requested_capability}` conflicts with "
            f"target architecture `{target_arch}` capability "
            f"`{compute_capability}`."
        )

    normalized["arch"] = target_arch
    normalized["compute_capability"] = compute_capability

    return normalized


def _bangc_backend_options(target: TargetContext, normalized: Mapping[str, Any]):
    from ninetoothed.backends.toolchain import normalize_bangc_arch

    platform_bangc = target.platform.metadata.get("bangc", {})

    if not isinstance(platform_bangc, Mapping) or not platform_bangc:
        return normalized

    platform_arch = normalize_bangc_arch(platform_bangc.get("arch", "native"))
    requested_arch = normalize_bangc_arch(normalized.get("arch", "native"))

    if (
        requested_arch != "native"
        and platform_arch != "native"
        and (requested_arch != platform_arch)
    ):
        raise ValueError(
            f"BangC architecture `{requested_arch}` conflicts with target "
            f"architecture `{platform_arch}`."
        )

    if platform_arch != "native":
        normalized["arch"] = platform_arch

    platform_chunk = platform_bangc.get("task_chunk")

    if platform_chunk is not None and "task_chunk" not in normalized:
        normalized["task_chunk"] = int(platform_chunk)

    return normalized


def target_device_types(value: Any) -> tuple[str, ...]:
    """Read accepted device types from a compilation or artifact-like value."""
    context = getattr(value, "target", None)

    if isinstance(context, TargetContext):
        return context.device_types

    artifact = getattr(value, "artifact", value)
    metadata = getattr(artifact, "metadata", {})
    target = dict(metadata.get("target", {})) if isinstance(metadata, Mapping) else {}
    raw_device_types = target.get("device_types", ())
    device_types = (
        (str(raw_device_types),)
        if isinstance(raw_device_types, str)
        else tuple(str(item) for item in raw_device_types)
    )

    return device_types or ("cuda",)


# Device types each backend binds tensors to, independent of the platform
# profile: BangC always runs on MLU devices, Triton follows its platform
# profile, and the remaining backends use CUDA-style streams.  Adding a
# hardware backend means adding its device tuple here once.
_BACKEND_DEVICE_TYPES = {
    Target.BANGC.value: ("mlu",),
}


def runtime_device_types(value: Any) -> tuple[str, ...]:
    """Resolve the device types a materialized backend accepts."""
    context = getattr(value, "target", None)

    if isinstance(context, TargetContext):
        backend = context.backend.value
    else:
        artifact = getattr(value, "artifact", value)
        backend = getattr(getattr(artifact, "backend", None), "value", None)

    if backend in _BACKEND_DEVICE_TYPES:
        return _BACKEND_DEVICE_TYPES[backend]

    if backend == Target.TRITON.value:
        return target_device_types(value)

    return ("cuda",)


def validate_artifact_materialization(value: Any, *, mode: str) -> None:
    """Validate a persisted artifact using its embedded profile snapshot."""
    artifact = getattr(value, "artifact", value)
    backend = getattr(getattr(artifact, "backend", None), "value", None)
    metadata = getattr(artifact, "metadata", {})

    if not isinstance(metadata, Mapping) or "target" not in metadata:
        return

    raw_target = metadata["target"]

    if not isinstance(raw_target, Mapping) or not raw_target:
        raise ValueError("Artifact target metadata must be a non-empty mapping.")

    target = dict(raw_target)
    raw_profile = target.get("profile")

    if not isinstance(raw_profile, Mapping) or not raw_profile:
        raise ValueError("Artifact target profile metadata is missing or malformed.")

    raw_backend_modes = raw_profile.get("backend_modes")

    if not isinstance(raw_backend_modes, Mapping) or not raw_backend_modes:
        raise ValueError("Artifact target backend modes are missing or malformed.")

    backend_modes = dict(raw_backend_modes)

    if backend is None or str(backend) not in backend_modes:
        raise ValueError(
            f"Artifact target profile does not declare backend `{backend}`."
        )

    declared_backend = target.get("backend")

    if declared_backend is None:
        raise ValueError("Artifact target backend metadata is missing.")

    if backend_id_for(declared_backend) != str(backend):
        raise ValueError(
            f"Artifact target backend `{declared_backend}` does not match "
            f"artifact backend `{backend}`."
        )

    supported = tuple(backend_modes.get(str(backend), ()))
    normalized_mode = materialization_mode_for(mode)

    if normalized_mode not in supported:
        raise ValueError(
            f"Backend `{backend}` on platform `{target.get('platform')}` does not "
            f"support `{normalized_mode}` materialization."
        )


__all__ = [
    "PlatformProfile",
    "PlatformRegistry",
    "TargetContext",
    "UnsupportedTargetCapabilityError",
    "create_default_platform_registry",
    "default_platform_registry",
    "resolve_target_context",
    "runtime_device_types",
    "target_backend_options",
    "target_device_types",
    "validate_artifact_materialization",
]
