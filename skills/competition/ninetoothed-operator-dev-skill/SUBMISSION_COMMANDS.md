# Submission Commands

Identity used for this signed release:

```bash
GitHubId="Dreamt-Deer-Waking-Fish"
TeamName="123123"
ParticipantName="刘李宏"
SubmissionBranch="2026-spring-Dreamt-Deer-Waking-Fish-T3-1-1"
OfficialReport="123123_九齿skill创新挑战_T3-1-1_赛题报告.pdf"
```

Run validation from the skill directory before creating a commit:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests -p "test_*.py" -v
PYTHONPYCACHEPREFIX="$(mktemp -d)" python -m compileall -q scripts tests
python scripts/validate_skill_package.py .
python scripts/validate_skill_package.py --strict-identity .
python scripts/check_no_secrets.py .
python scripts/check_false_verified_claims.py .
python scripts/check_markdown_links.py .
python scripts/verify_manifest.py .
```

Inspect changes:

```bash
git diff --check
git diff --stat
git diff --name-only
git status --short
```

Create the local commit only after reviewing every file:

```bash
git switch -c "${SubmissionBranch}"
git add skills/competition/ninetoothed-operator-dev-skill
git commit -m "feat(skill): add NineToothed operator development skill"
```

Suggested competition title:

```text
[2026春季][T3-1-1] Dreamt-Deer-Waking-Fish
```

CI-safe Conventional Commits alternative:

```text
feat(skill): add NineToothed operator development skill
```

Push and pull request creation remain manual actions. Use the confirmed personal fork URL from GitHub before pushing; it was not derivable from this local non-Git workspace.

```bash
git push -u origin "${SubmissionBranch}"
```
