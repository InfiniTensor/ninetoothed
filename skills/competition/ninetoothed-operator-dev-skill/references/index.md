# Reference Index

Open only the guide needed for the current task.

| Need | Guide |
| --- | --- |
| Find repository structure and nearby patterns | [Repository reading routes](repo_reading_routes.md) |
| Elementwise, broadcasting, masks, or dtype behavior | [Elementwise and broadcast](elementwise_broadcast_guide.md) |
| Reduction domains, stability, or block design | [Reduction and blocking](reduction_blocking_guide.md) |
| Strides, offsets, pooling, or non-contiguous input | [Layout, stride, and offset](layout_stride_offset_guide.md) |
| PyTorch references, tolerance, and test design | [Correctness testing](testing_correctness_guide.md) |
| Correctness-gated timing and regression analysis | [Performance benchmarking](performance_benchmark_guide.md) |
| Generated source, AOT output, or InfiniCore dispatch | [Generated source, AOT, and integration](generated_source_aot_integration_guide.md) |
| Reproduction, root cause, repair, and rerun | [Failure diagnosis](failure_diagnosis_playbook.md) |
| LF patches and clean apply-check | [Patch applicability](patch_applicability_guide.md) |
| Claim status, safety, and handoff evidence | [Evidence and compliance](evidence_and_compliance_policy.md) |

Do not load all references by default. Start with `SKILL.md`, identify the
operator family and current failure gate, then open one or two guides.
