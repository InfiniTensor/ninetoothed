# Contributing to NineToothed

Thank you for your interest in contributing to NineToothed! This document provides guidelines and instructions for contributing to this project.

> **Note:** This guide is still being actively improved.

## Reporting Issues

If you find a bug or have a feature request, please open an issue on [GitHub Issues](https://github.com/InfiniTensor/ninetoothed/issues). When reporting a bug, include:

- A clear description of the problem.
- Steps to reproduce the issue.
- Expected vs. actual behavior.
- Your environment (Python version, OS, GPU, etc.).

## Development Environment Setup

1. Fork and clone the repository:

   ```bash
   git clone https://github.com/InfiniTensor/ninetoothed.git
   cd ninetoothed
   ```

2. Install the package in editable mode with all optional dependencies:

   ```bash
   pip install -e .[all]
   ```

3. Install development dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up Git hooks:

   ```bash
   git config core.hooksPath .githooks
   ```

   This enables the project's commit message validation and branch name checks.

## Code Development Workflow

We follow [GitHub flow](https://docs.github.com/en/get-started/using-github/github-flow). The main steps are:

1. Create a branch.
2. Make changes.
3. Create a pull request (PR).
4. Address review comments.
5. Merge the PR.
6. Delete the branch.

### Branch Naming

Use kebab-case (lowercase letters, numbers, and hyphens) for branch names, with a maximum of 50 characters. This is enforced by the `pre-push` hook.

- **Valid:** `develop-visualization`, `fix-123-memory-leak`, `add-conv2d-support`
- **Invalid:** `Develop_Visualization`, `fix_memory_leak`, `myBranch`

### Commit Messages and PR Titles

The following rules apply to both commit messages and PR titles:

- **Capitalize** the first letter.
- **Do not** end with punctuation (`.`, `!`, `?`, etc.).
- **Use imperative mood** (e.g., "Add feature" not "Added feature").

These rules are enforced by the `commit-msg` hook.

**Valid examples:**

- `Add user authentication`
- `Fix memory leak in tensor allocation`
- `Refactor code generation module`

**Invalid examples:**

- `add user authentication` (lowercase)
- `Fix memory leak.` (trailing punctuation)
- `Added user authentication` (past tense)

### Pull Request Requirements

Before merging a PR, you must provide the `pytest` output in the PR description to confirm that all tests pass with your latest changes. The PR template includes a section for this. See [#30](https://github.com/InfiniTensor/ninetoothed/pull/30) for a reference example.

## Code Style Guide

Follow [PEP 8](https://peps.python.org/pep-0008/) as the primary style guide. For anything PEP 8 does not cover in detail, refer to the [GDScript style guide](https://docs.godotengine.org/en/stable/tutorials/scripting/gdscript/gdscript_styleguide.html)—while it targets a different language, its non-syntax conventions are still applicable.

Run [Ruff](https://docs.astral.sh/ruff/) before every commit:

```bash
ruff format && ruff check
```

This is also enforced by the `commit-msg` hook.

### Additional Rules

1. **Comments** should be complete English sentences, starting with a capital letter and ending with punctuation. Use Markdown syntax when referencing code within comments.

2. **Error messages and framework conventions:** When a framework has an established convention (e.g., `pytest.skip` messages are typically lowercase without a trailing period), follow that convention. Otherwise, use the same rules as comments.

3. **Function signatures:** If a function has no docstring or comment, do not add a blank line between the function signature and the function body.

4. **Blank lines around control flow:** Add a blank line before and after `if`, `for`, and similar statements.

5. **Return statements:** Add a blank line before a `return` statement, unless it directly follows a control flow statement like `if` or `for`.

6. **Docstrings:** Follow [PEP 257](https://peps.python.org/pep-0257/) conventions.

## Running Tests and Linting

Run the test suite:

```bash
pytest
```

Run the linter:

```bash
ruff check
```

Run the formatter:

```bash
ruff format
```

To run a full local CI check before pushing:

```bash
ruff format && ruff check && pytest
```

## Version Release Process

> This section is primarily for maintainers.

1. Create an `increment-version-number` branch and update the `version` field in `pyproject.toml`. Follow [Semantic Versioning 2.0.0](https://semver.org/) for version numbers.

2. Merge the branch via the standard code development workflow.

3. Create a tag on the `master` branch using the newly merged commit:

   ```bash
   git checkout master
   git pull origin master
   git tag -a v<major>.<minor>.<patch> -m "NineToothed version <major>.<minor>.<patch>"
   ```

4. Push the tag to the remote repository:

   ```bash
   git push origin v<major>.<minor>.<patch>
   ```

5. Verify the release by checking:
   - GitHub Actions workflow results.
   - The [Tags](https://github.com/InfiniTensor/ninetoothed/tags) and [Releases](https://github.com/InfiniTensor/ninetoothed/releases) pages.
   - The [PyPI page](https://pypi.org/project/ninetoothed/).
