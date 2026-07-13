# Subagent Orchestration

Use subagents to keep the parent context focused on decisions, not raw logs.
The parent should receive the result and the shortest useful evidence trail.

## Mandatory Subagent Scenarios

Delegate when the work is scoped, result-oriented, and likely to generate noisy
command output:

- operator implementation for a bounded file set;
- script creation or repair;
- test execution and failure triage;
- benchmark execution and result summarization;
- generated source inspection;
- AOT build diagnosis;
- environment-error investigation;
- repository scouting for existing patterns;
- independent patch or self-test review.

Keep work in the parent when the next step depends on direct design judgment or
when the user still needs to answer a question.

## Parent-Agent Contract

Before delegation, create `request.md` with the exact contract:

- Objective: one concrete outcome.
- Parent Context: why this is delegated and what is already decided.
- Allowed Write Scope: exact files or directories the subagent may edit.
- Forbidden Scope: files, directories, or actions the subagent must avoid.
- Expected Result: the artifact shape or diagnosis format expected on return.
- Validation Command: the exact command or command set to rerun before handoff.
- Return Summary Level: the shallowest useful level the parent needs, from `L0`
  to `L3`.

The request must be self-contained. Do not rely on chat history, hidden
conclusions, or desired answers unless the task is explicitly a proposal review.

## Subagent Tool Boundary

The subagent may use ordinary file, shell, search, and test tools. It must not
spawn nested subagents. This keeps session records readable and prevents
recursive delegation.

## Session Folder

Create one folder per delegated task:

```text
subagent-sessions/
  YYYYMMDD-HHMM-<slug>/
    request.md
    summary.md
    actions.md
    artifacts.md
    errors.md
```

Use the scaffold command:

```bash
python scripts/scaffold_subagent_session.py "short task slug"
```

If the folder was created only for a dry-run validation, delete it after the
check unless it is intentionally preserved as evidence.

## Progressive Summary Levels

Every subagent session records layered summaries. The parent sees only the
highest useful layer by default.

### L0 Parent Return

At most five lines:

- Result;
- Changed files;
- Validation;
- Risks;
- Session.

### L1 Session Summary

One screen:

- Objective;
- Approach;
- Files read;
- Files changed;
- Commands run;
- Final status;
- Unresolved risks.

### L2 Action Log

Structured but concise:

- chronological actions;
- command categories, not full raw logs;
- important error snippets;
- how each error was resolved;
- failed attempts worth preserving.

### L3 Raw Evidence

Raw logs stay inside the session folder as artifacts. Do not paste L3 into the
parent context unless directly requested.

## Return Format

Subagents return this shape to the parent:

```text
Result:
Changed files:
Validation:
Risks:
Session:
```

If the task fails, return the same shape with `Result: failed` and summarize the
smallest reproducible blocker.
