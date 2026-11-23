# Submission Checklist

## Machine-Verifiable

- [x] Final skill tree exists under `skills/competition/ninetoothed-operator-dev-skill/`.
- [x] `SKILL.md` frontmatter and trigger description validate.
- [x] Package unit tests pass in the final directory.
- [x] Package unit tests pass after clean archive extraction.
- [x] Python scripts compile and every CLI `--help` succeeds.
- [x] Secret scan passes.
- [x] Unsupported-claim scan passes.
- [x] Markdown-link scan passes.
- [x] Four required self-test families and the implementation case are present.
- [x] Two raw correctness-gated benchmark CSV records are present.
- [x] All four comparison patches use LF and pass clean strict apply-check.
- [x] No cache, bytecode, VCS metadata, virtual environment, or credential file is packaged.
- [x] Final report PDF is present and visually checked.
- [x] ZIP root, SHA-256 manifest, and clean-extraction inventory validate.

## Required Human Completion

- [x] Fill participant name, team name, GitHub ID, and signature in `HONOR_CODE.md`.
- [x] Rename the report PDF to the required team-based filename.
- [x] Replace reviewed identity values in `PR_DESCRIPTION.md` and submission commands.
- [ ] Review `REFERENCE.md`, evidence boundaries, and all benchmark wording.
- [ ] Review the final ZIP manually before upload.
- [ ] Push the branch and create the pull request only after review.
- [ ] Upload the required package/report materials through the current official channel.
- [ ] Confirm any pay-as-you-go server instance remains stopped.
