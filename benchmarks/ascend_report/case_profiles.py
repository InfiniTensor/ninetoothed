"""Opt-in case contracts shared by the pinned baseline and ntops kernels.

``csv`` returns an independent copy of the original plan. ``common_contracts_v1``
keeps all41 operator categories, removes product reductions and p=1 dropout,
and narrows six operator groups to their common contracts. On the pinned full
matrix this yields453 cases, including99 cases in the six affected groups.
Those99 cases receive distinct IDs and retain their original definitions.

The prepare/verify wrappers accept the unwrapped baseline functions explicitly.
They do not monkeypatch a module or implement a timed numerical substitute:
clamp/dropout calls still dispatch to the baseline's registered operators.
Reference construction and verification remain outside measurement.
"""

import copy
import hashlib
import json

DEFAULT_PROFILE = "csv"
COMMON_CONTRACTS_PROFILE = "common_contracts_v1"
AFFECTED_OPERATORS = frozenset(
    {"clamp", "dropout", "max_pool2d", "reduction", "addmm", "conv2d"}
)


def apply_profile(plan, profile=DEFAULT_PROFILE):
    """Return a new plan; never mutate or alias original case definitions.

    Changed IDs end in ``__common_v1``. All retained cases of an affected
    operator receive the profile marker, including its already-supported cases,
    so the whole paired group can be remeasured without mixing experiments.
    Removed cases remain in the original plan and are not relabeled as passes.
    """
    if profile not in (DEFAULT_PROFILE, COMMON_CONTRACTS_PROFILE):
        raise ValueError(f"Unknown case profile: {profile!r}")
    if profile == DEFAULT_PROFILE:
        return copy.deepcopy(plan)

    selected = []
    for original in plan:
        if original.get("case_profile") not in (None, DEFAULT_PROFILE):
            raise ValueError("Apply a case profile to the original plan only.")
        operator = original["operator"]
        parameters = original.get("parameters", {})
        if operator == "reduction" and parameters.get("kind") == "prod":
            continue
        if operator == "dropout" and (
            original.get("label") == "p_one" or parameters.get("p") == 1
        ):
            continue

        case = copy.deepcopy(original)
        if operator not in AFFECTED_OPERATORS:
            selected.append(case)
            continue

        case.update(
            id=original["id"] + "__common_v1",
            case_profile=COMMON_CONTRACTS_PROFILE,
            original_id=original["id"],
            original_case=copy.deepcopy(original),
            original_case_sha256=hashlib.sha256(
                json.dumps(
                    original, sort_keys=True, separators=(",", ":"), allow_nan=False
                ).encode("utf-8")
            ).hexdigest(),
        )
        if operator == "clamp":
            case["parameters"] = {"min": -1.0, "max": 1.0}
        elif operator == "dropout":
            probability = parameters.get("p", 0.25)
            case["parameters"] = {
                "p": probability,
                "training": parameters.get("training", True)
                if probability == 0
                else False,
                "return_mask": False,
            }
        elif operator == "max_pool2d":
            case.setdefault("parameters", {})["return_indices"] = False
        elif operator == "addmm":
            case.setdefault("parameters", {}).update(alpha=1.0, beta=1.0)
        elif operator == "conv2d" and original.get("label") == "grouped_dilated":
            case.setdefault("parameters", {}).update(
                stride=1, padding=1, dilation=1, groups=1
            )
            case["weight_shape"][1] = case["shape"][1]
        selected.append(case)

    return selected


def _is_common(case):
    profile = case.get("case_profile", DEFAULT_PROFILE)
    if profile not in (DEFAULT_PROFILE, COMMON_CONTRACTS_PROFILE):
        raise ValueError(f"Unknown case profile: {profile!r}")
    return profile == COMMON_CONTRACTS_PROFILE


def prepare(case, device, original_prepare):
    """Prepare actual inputs/parameters for a profile using baseline operators.

    The pinned baseline hardcodes clamp bounds and dropout mask/RNG arguments;
    override those only for explicitly profiled cases. Its max-pool preparation
    already honors return_indices=False. Other operators use it unchanged.
    """
    common = _is_common(case)
    operator = case["operator"]
    prepared = original_prepare(case, device)
    if not common:
        return prepared

    adjusted = dict(prepared)
    adjusted["case_profile_metadata"] = {
        "profile": COMMON_CONTRACTS_PROFILE,
        "original_id": case["original_id"],
        "output_contract": "tensor",
    }
    if operator == "max_pool2d":
        adjusted["case_profile_metadata"].update(
            baseline_internal_indices_writes_retained=True,
            note="The original baseline still computes/writes indices internally; only its returned contract is values-only.",
        )
    if operator not in {"clamp", "dropout"}:
        return adjusted

    parameters = dict(case["parameters"])
    if operator == "clamp":
        if parameters != {"min": -1.0, "max": 1.0}:
            raise ValueError("The common clamp contract requires bounds [-1, 1].")
    elif (
        set(parameters) != {"p", "training", "return_mask"}
        or parameters["return_mask"] is not False
        or type(parameters["training"]) is not bool
        or not isinstance(parameters["p"], (int, float))
        or isinstance(parameters["p"], bool)
        or not 0 <= parameters["p"] < 1
        or (parameters["training"] and parameters["p"] != 0)
    ):
        raise ValueError("The common dropout contract is a values-only identity.")

    import triton_ascend_ops

    function = triton_ascend_ops.get_operator(operator)
    adjusted["parameters"] = parameters
    args = prepared["args"]
    adjusted["call"] = lambda: function(*args, **parameters)
    if operator == "clamp":
        adjusted["reference"] = args[0].detach().cpu().float().clamp(**parameters)
    else:
        adjusted["case_profile_metadata"].update(
            numeric_scope="deterministic_identity_p0"
            if parameters["training"]
            else "inference_identity",
            baseline_internal_mask_writes_retained=True,
            random_generation=False,
            note="The original baseline identity kernel still allocates/writes a mask internally; no RNG or stochastic training is measured.",
        )
    # Baseline dropout's reference is already the CPU float input. The selected
    # p=0/inference contracts preserve it exactly and require no random mask.
    return adjusted


def verify(case, prepared, output, phase, original_verify):
    """Use baseline tolerances with tensor-only dropout and pooling outputs."""
    if _is_common(case) and case["operator"] in {"dropout", "max_pool2d"}:
        value_case = dict(case, operator="abs")
        return original_verify(value_case, prepared, output, phase)
    return original_verify(case, prepared, output, phase)
