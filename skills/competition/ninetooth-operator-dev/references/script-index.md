# Script Index

Scripts must be boring, offline, and deterministic. Error messages should tell
the next agent how to fix the issue.

## `lint_skill_structure.py`

Checks the required skill package files, reference maps, example task frames,
script executability, and forbidden artifacts.

## `collect_repo_map.py`

Prints a small repository map for a cloned NineToothed checkout.

## `scaffold_selftest.py`

Emits a self-test task skeleton to stdout.

## `scaffold_subagent_session.py`

Creates a timestamped subagent session folder from the templates in
`subagent-sessions/_template/`.
