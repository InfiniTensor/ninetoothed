# Patch Applicability Guide

## Generate

Run from the target repository root:

```bash
git diff --check
git diff --stat
git diff --name-only
git diff --binary > final.patch
```

Keep LF line endings, repository-relative `a/` and `b/` paths, and only the
files required by the task. Do not use absolute Windows or Unix user paths.

## Validate

Record the target revision, then run against a clean checkout of that revision:

```bash
git apply --check final.patch
```

For strict whitespace validation:

```bash
git apply --check --whitespace=error-all final.patch
```

Use `scripts/validate_patch_artifact.py` to scan line endings, changed paths,
and clean apply-check status without applying the patch.

## Apply And Test

Apply only inside an isolated workspace, inspect `git diff`, and run the
focused correctness command. Keep the original repository unchanged when the
task is an evaluation or handoff.

## Diagnose

- Wrong repository root: regenerate from the actual target root.
- Wrong strip level: inspect `a/` and `b/` prefixes.
- Revision mismatch: restore the recorded target revision or rebase the change.
- Stale context: regenerate from known source instead of editing hunk numbers.
- CRLF: normalize the patch to UTF-8 LF and repeat strict apply-check.
- Already applied: compare the target file before changing the patch.

Record local remediation separately from historical server results. A later
local pass does not retroactively change an earlier server failure.
