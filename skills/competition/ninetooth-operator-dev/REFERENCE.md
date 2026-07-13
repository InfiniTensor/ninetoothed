# References and Disclosure

## Local Competition Materials

- `九齿 .skill 创新挑战赛道赛题与规则.pdf`: local competition rule reference.
- `NineToothed_skills_Innovation_Competition_Track_Regulations.md`: local
  Markdown conversion of the same competition rules.

These files were used as local reference material only and are not included in
the submitted skill package.

## Project Materials

- `proposal.md`: public project proposal, scope summary, evaluation plan, risk
  boundaries, and compliance plan.
- `README.md` and `README_zh.md`: package installation and usage.
- `FINAL_REPORT.md`: final handoff status and verification evidence.
- `HONOR_CODE.md`: signed competition compliance statement.

## Upstream NineToothed Reference

- Repository: `https://github.com/InfiniTensor/ninetoothed.git`.
- Local clone path is user-configured with `NINETOOTHED_REPO`.
- Commit inspected during this work: `c9ebd4950a185beed8d4c1db9ff4a1fd133934ae`.

The skill references upstream tests and implementation files as navigation
anchors. It does not copy large upstream source files into the skill package.
The standalone development repository is
`https://github.com/LaiQuan-conquer/NineToothed-OperatorSkills`. When submitted
to NineToothed, the package is covered by the upstream Apache-2.0 license.

## Runtime and Tooling References

- Python standard library scripts in `skills/competition/ninetooth-operator-dev/scripts/`.
- Pytest-style command templates in the self-test task files.
- Triton benchmark helper usage mirrored from local NineToothed test patterns.

No online service, paid API, private account, or additional runtime dependency
is required by the skill package itself.

## AI Assistance Disclosure

Generative AI assistance was used to:

- expand reference files and examples across the iteration sequence;
- write mechanical lint rules and regression tests;
- prepare final handoff materials;
- run local validation commands and record evidence during development.

All generated material was kept in the repository and validated with the local
structure checks before being treated as project state.
